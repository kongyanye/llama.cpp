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

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n  chat completion: %s -m your_model.gguf -p \"Hello, how are you?\"\n", argv[0]);
    printf("\n  system prompt:   %s -m your_model.gguf -sys \"You are a helpful assistant\" -p \"Hello\"\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    // Set log level to error (only error messages will be shown)
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
    if (params.n_predict == -1) params.n_predict = 128;
    if (params.n_ctx == 4096) params.n_ctx = 2048;
    if (params.n_batch == 2048) params.n_batch = 512;

    // Force chat completion mode (like llama-cli default behavior)
    params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;

    LOG_INF("llama_shard_chain: chat completion inference\n");
    LOG_INF("Model: %s\n", params.model.path.c_str());
    LOG_INF("Prompt: %s\n", params.prompt.c_str());
    if (!params.system_prompt.empty()) {
        LOG_INF("System prompt: %s\n", params.system_prompt.c_str());
    }
    LOG_INF("Max tokens: %d\n", params.n_predict);

    // Initialize backend
    llama_backend_init();
    llama_numa_init(params.numa);

    // Initialize model and context using common helpers
    LOG_INF("Loading model...\n");
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

    // Initialize chat templates
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    printf("\n=== Model loaded successfully ===\n");
    printf("Model: n_ctx=%d, n_layer=%d, n_embd=%d\n",
        llama_n_ctx(ctx),
        llama_model_n_layer(model),
        llama_model_n_embd(model));
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

        auto chat_result = common_chat_templates_apply(chat_templates.get(), inputs);
        formatted_prompt = chat_result.prompt;

        LOG_INF("Using chat template, formatted prompt length: %zu\n", formatted_prompt.length());
    } else {
        LOG_ERR("Error: prompt is required for chat completion\n");
        return 1;
    }

    // Tokenize prompt
    LOG_INF("Tokenizing prompt...\n");
    std::vector<llama_token> input_tokens = common_tokenize(ctx, formatted_prompt, true);
    LOG_INF("Prompt tokenized to %zu tokens\n", input_tokens.size());

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

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Error: failed to decode prompt\n");
        llama_batch_free(batch);
        // ctx and model are automatically freed by smart pointers in llama_init when it goes out of scope
        return 1;
    }
    llama_batch_free(batch);

    // Initialize sampler
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("Failed to initialize sampler\n");
        return 1;
    }

    // Generation loop
    printf("\n=== Generating ===\n");
    std::string result;
    int max_tokens = params.n_predict;
    if (max_tokens <= 0) {
        max_tokens = 128;  // Default to 128 if not specified
    }

    auto * vocab = llama_model_get_vocab(model);

    int n_cur = input_tokens.size();  // Track current position in sequence

    for (int i = 0; i < max_tokens; i++) {
        // Get logits from the context
        auto * logits = llama_get_logits(ctx);
        if (!logits) {
            LOG_WRN("Warning: no logits available\n");
            break;
        }

        // Sample next token using common sampler
        llama_token next_token = common_sampler_sample(smpl, ctx, -1);

        // Check for end-of-sequence
        if (llama_vocab_is_eog(vocab, next_token)) {
            printf(" [end]\n");
            break;
        }

        // Convert token to text
        std::string token_text = common_token_to_piece(ctx, next_token);
        result += token_text;
        printf("%s", token_text.c_str());
        fflush(stdout);

        input_tokens.push_back(next_token);
        n_cur++;

        // Create batch for next token
        llama_batch batch = llama_batch_get_one(&next_token, 1);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("\nError: failed to decode token\n");
            break;
        }
    }

    printf("\n\n=== Generation completed ===\n");
    LOG_INF("Generated %zu characters\n", result.length());

    // Cleanup
    common_sampler_free(smpl);
    // ctx and model are automatically freed by smart pointers in llama_init
    llama_backend_free();

    return 0;
}
