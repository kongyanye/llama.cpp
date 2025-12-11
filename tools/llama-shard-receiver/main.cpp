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
    printf("\n  %s -m your_shard.gguf -i hidden_state.bin\n", argv[0]);
    printf("\n  options:\n");
    printf("    -m MODEL, --model MODEL     Path to GGUF model shard file (receiver shard)\n");
    printf("    -i INPUT, --input INPUT     Input file with hidden state from previous shard\n");
    printf("    -t TEMPERATURE, --temp TEMPERATURE  Sampling temperature (default: 0.8)\n");
    printf("\nNote: This tool generates exactly ONE token from the injected hidden state.\n");
    printf("      For multi-token generation, alternate with shard-0 extraction.\n");
    printf("      Each generation round requires returning to shard-0 for the next token.\n");
    printf("\n");
}

// Header structure for hidden state files (must match Shard 0)
struct HiddenStateHeader {
    int magic = 0xDEADBEEF;     // Magic number for validation
    int version = 1;            // Format version
    int sequence_length = 0;    // Number of tokens in sequence
    int embedding_dim = 0;      // Hidden dimension
    int last_position = 0;      // Last position index (0-based)
};

// Forward declarations
std::vector<float> load_feature_map(const std::string & filename, size_t expected_size);
std::vector<float> load_hidden_state_with_metadata(const std::string & filename, HiddenStateHeader & header);

// Function to load feature map from disk (legacy format)
std::vector<float> load_feature_map(const std::string & filename, size_t expected_size) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        LOG_ERR("Error: unable to open input file %s\n", filename.c_str());
        return {};
    }

    // Get file size
    infile.seekg(0, std::ios::end);
    size_t file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(float);
    if (num_elements != expected_size) {
        LOG_ERR("Error: file size mismatch. Expected %zu elements, got %zu\n", expected_size, num_elements);
        return {};
    }

    std::vector<float> features(num_elements);
    infile.read(reinterpret_cast<char*>(features.data()), file_size);
    infile.close();

    LOG_INF("Feature map loaded from %s (%zu elements, %.2f MB)\n",
            filename.c_str(), num_elements, (file_size) / (1024.0 * 1024.0));

    return features;
}

// Function to load hidden state with metadata from disk
std::vector<float> load_hidden_state_with_metadata(const std::string & filename, HiddenStateHeader & header) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        LOG_ERR("Error: unable to open input file %s\n", filename.c_str());
        return {};
    }

    // Read and validate header
    infile.read(reinterpret_cast<char*>(&header), sizeof(HiddenStateHeader));
    if (infile.gcount() != sizeof(HiddenStateHeader)) {
        LOG_ERR("Error: unable to read header from %s\n", filename.c_str());
        return {};
    }

    // Validate magic number
    if (header.magic != 0xDEADBEEF) {
        LOG_ERR("Error: invalid file format. Magic number mismatch.\n");
        return {};
    }

    // Validate version
    if (header.version != 1) {
        LOG_ERR("Error: unsupported file version %d\n", header.version);
        return {};
    }

    // Calculate data size
    size_t total_elements = (size_t)header.sequence_length * header.embedding_dim;
    size_t data_size = total_elements * sizeof(float);

    // Read hidden state tensor data
    std::vector<float> hidden_state(total_elements);
    infile.read(reinterpret_cast<char*>(hidden_state.data()), data_size);
    if (infile.gcount() != data_size) {
        LOG_ERR("Error: unable to read complete tensor data from %s\n", filename.c_str());
        return {};
    }

    infile.close();

    LOG_INF("Hidden state loaded from %s (seq_len=%d, embd_dim=%d, total_elements=%zu, %.2f MB)\n",
            filename.c_str(), header.sequence_length, header.embedding_dim, total_elements, data_size / (1024.0 * 1024.0));

    return hidden_state;
}

int main(int argc, char ** argv) {
    // Set log level to info
    llama_log_set([](ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_INFO) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    common_params params;

    // Set default parameters for receiver shard
    params.n_predict = 1;  // Only generate one token per call
    params.sampling.temp = 0.8f;

    // Extract custom parameters before common_params_parse
    std::string hidden_state_file = "hidden_state_shard0.bin";  // default

    // Create a new argv without our custom parameter
    std::vector<char*> new_argv;
    new_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) {
                hidden_state_file = argv[i + 1];
                i++; // Skip the next argument
            }
        } else {
            new_argv.push_back(argv[i]);
        }
    }

    int new_argc = new_argv.size();
    char** new_argv_data = new_argv.data();

    if (!common_params_parse(new_argc, new_argv_data, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    LOG_INF("llama_shard_receiver: receiving shard with hidden state injection\n");
    LOG_INF("Model: %s\n", params.model.path.c_str());
    LOG_INF("Hidden state input: %s\n", hidden_state_file.c_str());
    if (!params.prompt.empty()) {
        LOG_INF("Context prompt: %s\n", params.prompt.c_str());
    }

    // Initialize backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Initialize model and context using common helpers
    LOG_INF("Loading receiver shard...\n");
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

    printf("\n=== Receiver Model loaded successfully ===\n");
    printf("Model: n_ctx=%d, n_layer=%d, n_embd=%d\n",
        llama_n_ctx(ctx),
        llama_model_n_layer(model),
        llama_model_n_embd(model));

    int total_layers = llama_model_n_layer(model);
    int embedding_dim = llama_model_n_embd(model);
    printf("Receiver layers: %d, Embedding dimension: %d\n", total_layers, embedding_dim);
    fflush(stdout);

    // Load hidden state with metadata from file
    HiddenStateHeader header;
    std::vector<float> hidden_state = load_hidden_state_with_metadata(hidden_state_file, header);
    if (hidden_state.empty()) {
        LOG_ERR("Error: failed to load hidden state from %s\n", hidden_state_file.c_str());
        return 1;
    }

    // Validate dimensions match
    if (header.embedding_dim != embedding_dim) {
        LOG_ERR("Error: embedding dimension mismatch. Expected %d, got %d\n", embedding_dim, header.embedding_dim);
        return 1;
    }

    printf("\n=== Hidden State Loaded Successfully ===\n");
    printf("Available sequence length: %d token(s)\n", header.sequence_length);
    printf("Original input sequence: %d tokens\n", header.last_position + 1);
    printf("Embedding dimension: %d\n", header.embedding_dim);
    printf("Last processed position: %d\n", header.last_position);
    printf("Total tensor size: %zu elements (%.2f MB)\n", hidden_state.size(), (hidden_state.size() * sizeof(float)) / (1024.0 * 1024.0));

    // Print sample values for verification
    printf("Sample values (hidden state, first 10 dims): ");
    for (int i = 0; i < std::min(10, embedding_dim); i++) {
        printf("%.6f ", hidden_state[i]);
    }
    printf("\n");
    fflush(stdout);

    printf("\n=== Injecting Hidden State and Generating Continuation ===\n");
    printf("Creating batch with embeddings for continuation...\n");

    // For single position hidden state, we need to create a batch that continues from the last position
    // We'll create a batch with just the hidden state as the embedding for continuation
    int sequence_length = 1;  // We only have one position of hidden state
    llama_batch batch = llama_batch_init(sequence_length, embedding_dim, 1);
    if (batch.embd == nullptr) {
        LOG_ERR("Error: failed to allocate batch with embedding support\n");
        return 1;
    }

    // Copy hidden state to batch.embd for the current position
    memcpy(batch.embd, hidden_state.data(), embedding_dim * sizeof(float));

    // Set position to 0 for the receiver shard (fresh context with injected embeddings)
    batch.pos[0] = 0;  // Start fresh position for receiver shard
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;  // Request logits for this position
    batch.n_tokens = sequence_length;

    printf("Hidden state injected into batch.embd (position 0 in receiver shard)\n");
    printf("Note: Original sequence had %d tokens, now continuing from hidden state\n", header.last_position + 1);
    printf("Running forward pass through receiver shard...\n");

    // Process the injected hidden state through shard 1
    int decode_result = llama_decode(ctx, batch);
    if (decode_result != 0) {
        LOG_ERR("Error: failed to decode with hidden state injection\n");
        llama_batch_free(batch);
        return 1;
    }

    printf("Forward pass completed successfully!\n");
    printf("Generating next token from injected hidden state...\n");
    printf("Temperature: %.2f\n", params.sampling.temp);

    // Initialize sampler
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("Error: failed to initialize sampler\n");
        llama_batch_free(batch);
        return 1;
    }

    // Generate exactly ONE token from the injected hidden state
    llama_token new_token = common_sampler_sample(smpl, ctx, batch.n_tokens - 1);

    // Get vocabulary for token conversion
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (new_token == llama_vocab_eos(vocab)) {
        printf("[Generated EOS token - end of sequence]\n");
    } else {
        // Convert token to string
        char token_str[256];
        int n = llama_token_to_piece(vocab, new_token, token_str, sizeof(token_str), 0, true);
        if (n > 0) {
            printf("Generated token: '%.*s'\n", n, token_str);
        }

        // Accept the token in sampler
        common_sampler_accept(smpl, new_token, true);
    }

    common_sampler_free(smpl);
    llama_batch_free(batch);

    printf("\n=== Shard 1 Processing Complete ===\n");
    printf("Next step: Return to Shard 0 with token ");
    if (new_token == llama_vocab_eos(vocab)) {
        printf("<EOS>");
    } else {
        printf("<TOKEN:%d>", new_token);
    }
    printf(" for hidden state extraction\n");

    // Cleanup
    llama_backend_free();

    LOG_INF("Receiver shard processing completed\n");
    return 0;
}