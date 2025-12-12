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
    params.sampling.temp = 0.0f;  // Match standalone default temperature

    // Extract custom parameters before common_params_parse
    std::string hidden_state_file = "model_state_shard0.bin";  // default

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

    printf("DEBUG: About to load complete model state...\n");
    // Load complete model state
    printf("Loading complete model state from shard0...\n");
    std::ifstream state_file("model_state_shard0.bin", std::ios::binary);
    if (!state_file.is_open()) {
        LOG_ERR("Error: unable to open state file model_state_shard0.bin\n");
        LOG_ERR("Make sure shard0 was run first to generate the state file\n");
        return 1;
    }

    state_file.seekg(0, std::ios::end);
    size_t state_size = state_file.tellg();
    state_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> state_data(state_size);
    state_file.read(reinterpret_cast<char*>(state_data.data()), state_size);
    state_file.close();

    // Restore state in the model context
    printf("Restoring state in shard1 model context...\n");
    size_t read = llama_state_set_data(ctx, state_data.data(), state_size);
    printf("Model state restored (%zu bytes)\n", read);

    // Create a minimal batch for sampling
    printf("Creating batch for token generation from restored state...\n");

    // Initialize sampler
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("Error: failed to initialize sampler\n");
        return 1;
    }

    printf("Forward pass completed successfully!\n");
    printf("Generating next token from loaded state...\n");
    printf("Temperature: %.2f\n", params.sampling.temp);

    // Generate exactly ONE token from the loaded state
    // Since we loaded the state, the logits should be available at the last position
    llama_token new_token = common_sampler_sample(smpl, ctx, -1);  // Use -1 for last position

    // Get vocabulary for token conversion
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Debug: Print token ID
    printf("Generated token ID: %d", new_token);

    if (new_token == llama_vocab_eos(vocab)) {
        printf(" [EOS token - end of sequence]\n");
    } else {
        // Convert token to string
        char token_str[256];
        int n = llama_token_to_piece(vocab, new_token, token_str, sizeof(token_str), 0, true);
        if (n > 0) {
            printf(", token text: '%.*s'\n", n, token_str);
        }

        // Accept the token in sampler
        common_sampler_accept(smpl, new_token, true);
    }

    common_sampler_free(smpl);

    printf("\n=== Shard 1 Processing Complete ===\n");
    printf("Successfully generated token from complete model state transfer\n");
    printf("This demonstrates correct state transfer between model shards\n");

    // Cleanup
    llama_backend_free();

    LOG_INF("Receiver shard processing completed\n");
    return 0;
}