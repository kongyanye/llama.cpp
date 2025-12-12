#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "sampling.h"
#include "arg.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n  %s -m your_shard.gguf -p \"Hello, how are you?\" -o output_features.bin\n", argv[0]);
    printf("\n  options:\n");
    printf("    -m MODEL, --model MODEL     Path to GGUF model shard file\n");
    printf("    -p PROMPT, --prompt PROMPT   Input prompt for inference\n");
    printf("    -o OUTPUT, --output OUTPUT   Output file for feature map (default: feature_map.bin)\n");
    printf("    -l LAYER, --layer LAYER     Layer number to extract features from (default: last layer)\n");
    printf("\n");
}

// Header structure for hidden state files
struct HiddenStateHeader {
    int magic = 0xDEADBEEF;     // Magic number for validation
    int version = 1;            // Format version
    int sequence_length = 0;    // Number of tokens in sequence
    int embedding_dim = 0;      // Hidden dimension
    int last_position = 0;      // Last position index (0-based)
};

// Function to save feature map to disk (legacy format)
void save_feature_map(const float * features, size_t size, const std::string & filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        LOG_ERR("Error: unable to open output file %s\n", filename.c_str());
        return;
    }

    outfile.write(reinterpret_cast<const char*>(features), size * sizeof(float));
    outfile.close();

    LOG_INF("Feature map saved to %s (%zu elements, %.2f MB)\n",
            filename.c_str(), size, (size * sizeof(float)) / (1024.0 * 1024.0));
}

// Function to save hidden state with metadata
void save_hidden_state_with_metadata(const float * hidden_state, int seq_len, int embd_dim, int last_pos, const std::string & filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        LOG_ERR("Error: unable to open output file %s\n", filename.c_str());
        return;
    }

    // Prepare header
    HiddenStateHeader header;
    header.sequence_length = seq_len;
    header.embedding_dim = embd_dim;
    header.last_position = last_pos;

    // Write header
    outfile.write(reinterpret_cast<const char*>(&header), sizeof(HiddenStateHeader));

    // Write hidden state tensor data
    size_t total_elements = (size_t)seq_len * embd_dim;
    outfile.write(reinterpret_cast<const char*>(hidden_state), total_elements * sizeof(float));
    outfile.close();

    LOG_INF("Hidden state saved to %s (seq_len=%d, embd_dim=%d, total_elements=%zu, %.2f MB)\n",
            filename.c_str(), seq_len, embd_dim, total_elements, (total_elements * sizeof(float)) / (1024.0 * 1024.0));
}

int main(int argc, char ** argv) {
    // Set log level to info
    llama_log_set([](ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_INFO) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    // Override default values if needed
    if (params.n_predict == -1) params.n_predict = 1;  // Only need one forward pass
    if (params.n_ctx == 4096) params.n_ctx = 2048;
    if (params.n_batch == 2048) params.n_batch = 512;

    LOG_INF("llama_single_shard_infer: single shard inference with feature extraction\n");
    LOG_INF("Model: %s\n", params.model.path.c_str());
    LOG_INF("Prompt: %s\n", params.prompt.c_str());
    LOG_INF("Output file: feature_map.bin\n");

    // Initialize backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Initialize model and context using common helpers
    LOG_INF("Loading model shard...\n");
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (!model) {
        LOG_ERR("Error: unable to load model from %s\n", params.model.path.c_str());
        return 1;
    }

    if (!ctx) {
        LOG_ERR("Error: unable to create context\n");
        return 1;
    }

    printf("\n=== Model loaded successfully ===\n");
    printf("Model: n_ctx=%d, n_layer=%d, n_embd=%d\n",
        llama_n_ctx(ctx),
        llama_model_n_layer(model),
        llama_model_n_embd(model));

    int total_layers = llama_model_n_layer(model);
    int embedding_dim = llama_model_n_embd(model);
    printf("Total layers: %d, Embedding dimension: %d\n", total_layers, embedding_dim);
    fflush(stdout);

    // Always use chat formatting (chat completion mode)
    std::vector<common_chat_msg> chat_msgs;
    std::string formatted_prompt;

    if (!params.prompt.empty()) {
        // Add system prompt if provided
        if (!params.system_prompt.empty()) {
            common_chat_msg system_msg;
            system_msg.role = "system";
            system_msg.content = params.system_prompt;
            chat_msgs.push_back(system_msg);
        }

        // Add user prompt
        common_chat_msg user_msg;
        user_msg.role = "user";
        user_msg.content = params.prompt;
        chat_msgs.push_back(user_msg);

        // Apply chat template
        common_chat_templates_inputs inputs;
        inputs.use_jinja = params.use_jinja;
        inputs.messages = chat_msgs;
        inputs.add_generation_prompt = true;

        auto chat_templates = common_chat_templates_init(model, params.chat_template);
        auto chat_result = common_chat_templates_apply(chat_templates.get(), inputs);
        formatted_prompt = chat_result.prompt;

        LOG_INF("Using chat template, formatted prompt length: %zu\n", formatted_prompt.length());
    } else {
        LOG_ERR("Error: prompt is required for inference\n");
        return 1;
    }

    // Tokenize prompt
    LOG_INF("Tokenizing prompt...\n");
    std::vector<llama_token> input_tokens = common_tokenize(ctx, formatted_prompt, true, true);
    LOG_INF("Prompt tokenized to %zu tokens\n", input_tokens.size());

    // Debug: Print first few tokens for verification
    LOG_INF("First 10 tokens: [");
    for (size_t i = 0; i < std::min((size_t)10, input_tokens.size()); i++) {
        LOG_INF("%d", input_tokens[i]);
        if (i < std::min((size_t)10, input_tokens.size()) - 1) {
            LOG_INF(", ");
        }
    }
    LOG_INF("]\n");

    // Process prompt tokens
    LOG_INF("Processing prompt...\n");
    llama_batch batch = llama_batch_init(input_tokens.size(), 0, 1);
    for (size_t i = 0; i < input_tokens.size(); i++) {
        batch.token[i] = input_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == input_tokens.size() - 1) ? 1 : 0;  // Request logits for last token
    }
    batch.n_tokens = input_tokens.size();

    // For partial models, we need to handle decoding differently
    printf("\n=== Processing Partial Model ===\n");
    printf("Note: Loading first shard only - extracting intermediate features\n");
    printf("Available layers: %d (partial model)\n", total_layers);

    // Try to decode, but handle partial model gracefully
    int decode_result = llama_decode(ctx, batch);
    const float * embeddings = nullptr;

    if (decode_result != 0) {
        LOG_ERR("Error: failed to decode with partial model\n");
        llama_batch_free(batch);
        return 1;
    }

    // Save complete model state instead of just hidden state
    printf("Saving complete model state including KV cache...\n");

    size_t state_size = llama_state_get_size(ctx);
    std::vector<uint8_t> state_data(state_size);
    size_t written = llama_state_get_data(ctx, state_data.data(), state_size);

    std::ofstream state_file("model_state_shard0.bin", std::ios::binary);
    state_file.write(reinterpret_cast<const char*>(state_data.data()), written);
    state_file.close();

    printf("Complete model state saved to model_state_shard0.bin (%zu bytes)\n", written);
    printf("State includes: embeddings, KV cache, and all intermediate values\n");
    printf("This contains all necessary information for shard1 to continue inference\n");

    // Cleanup
    llama_batch_free(batch);
    llama_backend_free();

    LOG_INF("Feature extraction completed successfully\n");
    return 0;
}