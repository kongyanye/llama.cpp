#include "models.h"

llm_build_llama_shard::llm_build_llama_shard(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t n_embd_head_k = hparams.n_embd_head_k;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Input embedding
    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    // Position encoding
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Process layers - shards may contain a subset of the full model layers
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Attention normalization
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Self-attention using the helper function
        {
            // Get rope factors if supported
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // Query, Key, Value projections
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }

            // Reshape for attention
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(),    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(), n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(), n_tokens);

            // Apply RoPE (Rotary Position Embedding)
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Use the built-in attention function
            cur = build_attn(inp_attn,
                model.layers[il].wo, model.layers[il].bo,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // Handle output IDs for last layer
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // Feed-forward network
        if (model.layers[il].ffn_gate_inp == nullptr) {
            // Standard FFN
            cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE FFN
            cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                false, 0.0,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
            cb(cur, "ffn_moe_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    // Output normalization (may not be present in intermediate shards)
    // For shard models, we need to handle the case where output_norm/output might be NULL
    // This is normal for intermediate shards that only contain a subset of layers

    bool is_final_shard = (model.output != NULL || model.output_norm != NULL);

    if (is_final_shard) {
        // This shard has output components - it's likely the final shard
        if (model.output_norm != NULL) {
            cur = build_norm(cur,
                model.output_norm, NULL,
                LLM_NORM_RMS, -1);
            cb(cur, "result_norm", -1);
        }

        // Final output layer
        if (model.output != NULL) {
            cur = build_lora_mm(model.output, cur);
            cb(cur, "output", -1);
        }
    } else {
        // This is an intermediate shard - no output components
        // The final tensor will be used for hidden state extraction
        cb(cur, "hidden_state", -1);
    }

    // Build the computation graph
    ggml_build_forward_expand(gf, cur);
}