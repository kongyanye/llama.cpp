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



int main(int argc, char ** argv) {
    // Set log level to info
    llama_log_set([](ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
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

    // Process prompt tokens
    llama_batch batch = llama_batch_init(input_tokens.size(), 0, 1);
    for (size_t i = 0; i < input_tokens.size(); i++) {
        batch.token[i] = input_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;  // Request logits for ALL tokens to preserve hidden states
    }
    batch.n_tokens = input_tokens.size();

    int decode_result = llama_decode(ctx, batch);

    if (decode_result != 0) {
        LOG_ERR("Error: failed to decode with partial model\n");
        llama_batch_free(batch);
        return 1;
    }

    for (int tok_idx = 0; tok_idx < 2; tok_idx++) {
        // Extract hidden state after the last layer (layer 7)
        printf("Extracting hidden state after layer 7...\n");
        const float * hidden_state = llama_get_hidden_state(ctx);
        if (!hidden_state) {
            // LOG_ERR("Error: unable to get hidden state\n");
            llama_batch_free(batch);
            return 1;
        }
        // printf("First 10 elements of hidden_state:\n");
        // for (int i = 0; i < 10; i++) {                                                                                                                                          
        //     printf("  hidden_state[%d] = %.6f\n", i, hidden_state[i]);                                                                                                          
        // }                                                                                                                                                                       
        // printf("\n");

        int tok_len;
        if (tok_idx == 0) {
            tok_len = input_tokens.size();
        } else {
            tok_len = 1;
        }
        // Save hidden state for Shard1 (format: [embd_dim, seq_len])
        std::string hidden_file_name = "hidden_state_shard" + std::to_string(tok_idx) + ".bin";
        std::ofstream hidden_file(hidden_file_name, std::ios::binary);
        hidden_file.write(reinterpret_cast<const char*>(hidden_state),
                        tok_len * embedding_dim * sizeof(float));
        hidden_file.close();

        printf("Hidden state saved to %s (%u tokens, %d dims)\n",
            hidden_file_name.c_str(), tok_len, embedding_dim);

        llama_token next_token(9906);
        llama_batch batch = llama_batch_get_one(&next_token, 1);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("\nError: failed to decode token\n");
            break;
        }
    }

    // Cleanup
    llama_batch_free(batch);
    llama_backend_free();

    LOG_INF("Feature extraction completed successfully\n");
    return 0;
}