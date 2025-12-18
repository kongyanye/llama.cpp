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

    // Set default parameters for receiver shard
    params.n_predict = 1;  // Only generate one token per call
    params.sampling.temp = 0.0f;  // Match standalone default temperature
    params.warmup = false;
    params.n_gpu_layers = 0;

    // Extract custom parameters before common_params_parse
    
    LOG_INF("llama_shard_receiver: receiving shard with hidden state injection\n");
    LOG_INF("Model: %s\n", params.model.path.c_str());
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

    // Initialize sampler
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("Error: failed to initialize sampler\n");
        return 1;
    }

    llama_batch batch;

    printf("Temperature: %.2f\n", params.sampling.temp);

    for (int tok_idx = 0; tok_idx < 2; tok_idx++) {
        std::string hidden_state_file = "/home/sig/files/ModelFlow/llama.cpp/build/hidden_state_shard" + std::to_string(tok_idx) + ".bin";

        // Load hidden state from Shard0
        printf("Loading hidden state from %s...\n", hidden_state_file.c_str());
        std::ifstream hidden_file(hidden_state_file, std::ios::binary);
        if (!hidden_file.is_open()) {
            LOG_ERR("Error: unable to open hidden_state_shard0.bin\n");
            LOG_ERR("Make sure shard0 was run first to generate the hidden state file\n");
            return 1;
        }

        // Calculate sequence length from hidden state file size
        // The file contains seq_len * embedding_dim float values
        hidden_file.seekg(0, std::ios::end);
        size_t file_size = hidden_file.tellg();
        hidden_file.seekg(0, std::ios::beg);

        // Calculate sequence length: total_floats / embedding_dim
        int seq_len = static_cast<int>(file_size / (sizeof(float) * embedding_dim));
        int tok_len;
        if (tok_idx == 0) {
            tok_len = seq_len - 1;
        } else {
            tok_len++;
        }

        printf("Hidden state file: %zu bytes, embedding_dim: %d, calculated seq_len: %d\n",
            file_size, embedding_dim, seq_len);

        std::vector<float> hidden_state(file_size / sizeof(float));
        hidden_file.read(reinterpret_cast<char*>(hidden_state.data()), file_size);
        hidden_file.close();

        printf("Hidden state loaded: %zu floats (%.2f KB)\n",
            hidden_state.size(), (file_size) / (1024.0));

        printf("First 10 elements of hidden_state:\n");
        for (int i = 0; i < 10; i++) {                                                                                                                                          
            printf("  hidden_state[%d] = %.6f\n", i, hidden_state[i]);                                                                                                          
        }                                                                                                                                                                       
        printf("\n");

        // Create batch for embedding input
        printf("Creating batch with all hidden state embeddings...\n");

        if (tok_idx == 0) {
            batch = llama_batch_init(seq_len, embedding_dim, 1);
            batch.n_tokens = seq_len;
            batch.token = nullptr;  // CRITICAL: Must be NULL when using embeddings!
            printf("DEBUG: Copying embeddings directly - shard0 already saves in [%d][%d] format\n", embedding_dim, seq_len);
            memcpy(batch.embd, hidden_state.data(), file_size);
        } else {
            batch = llama_batch_init(1, embedding_dim, 1);
            batch.n_tokens = 1;
            batch.token = nullptr;  // CRITICAL: Must be NULL when using embeddings!
            memcpy(batch.embd, hidden_state.data(), file_size);
        }

        // Set positions and sequence info
        if (tok_idx == 0) {
            // Full sequence processing
            for (int i = 0; i < seq_len; i++) {
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == seq_len - 1) ? 1 : 0;  // Only request logits for last token
            }
        } else {
            // Single token processing (next token generation)
            batch.pos[0] = tok_len;  // Position for next token in sequence
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;  // Request logits for this token
        }

        // Debug: Verify embedding data
        printf("DEBUG: First 5 embedding values: %.6f, %.6f, %.6f, %.6f, %.6f\n",
            batch.embd[0], batch.embd[1], batch.embd[2], batch.embd[3], batch.embd[4]);
        printf("DEBUG: batch.token = %p\n", (void*)batch.token);
        printf("DEBUG: batch.embd = %p\n", (void*)batch.embd);

        // Run forward pass through Shard1 layers (original 8-15)
        printf("Running forward pass through Shard1 layers...\n");
        int decode_result = llama_decode(ctx, batch);
        if (decode_result != 0) {
            LOG_ERR("Error: failed to decode\n");
            llama_batch_free(batch);
            return 1;
        }
        printf("Forward pass completed - ready for sampling\n");

        // Get logits for the last token and print shape + first elements
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits != NULL) {
            // Get vocabulary size (logits tensor dimension)
            const struct llama_vocab * vocab = llama_model_get_vocab(model);
            int32_t n_vocab = llama_vocab_n_tokens(vocab);

            printf("Logits shape: [%d] (vocabulary size)\n", n_vocab);
            // printf("Logits pointer: %p\n", (void*)logits);
            // printf("First 10 logits:\n");
            // for (int i = 0; i < 10 && i < n_vocab; i++) {
            //     printf("  logits[%d] = %.6f\n", i, logits[i]);
            // }

            // Find token with highest probability (max logit)
            int max_idx = 0;
            float max_logit = logits[0];
            for (int i = 1; i < n_vocab; i++) {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    max_idx = i;
                }
            }
            printf("Max logit: %.6f at token %d\n", max_logit, max_idx);
        } else {
            printf("Error: Failed to get logits\n");
        }

    
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
            // Convert token to string using common API for consistency
            std::string token_text = common_token_to_piece(ctx, new_token);
            printf(", token text: '%s'\n", token_text.c_str());
        }
    }

    common_sampler_free(smpl);
    llama_batch_free(batch);

    printf("\n=== Shard 1 Processing Complete ===\n");
    printf("Successfully generated token from complete model state transfer\n");
    printf("This demonstrates correct state transfer between model shards\n");

    // Cleanup
    llama_backend_free();

    LOG_INF("Receiver shard processing completed\n");
    return 0;
}