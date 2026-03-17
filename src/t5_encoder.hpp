#pragma once

// t5_encoder.hpp – minimal T5 encoder for text conditioning
//
// Implements the T5 encoder (encoder-only, no decoder) in C++ using GGML.
// Supports both T5-Base (hidden=768) and T5-XXL (hidden=4096) variants.
//
// GGUF tensor naming convention expected:
//   token_emb.weight
//   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
//   encoder.block.{i}.layer.0.layer_norm.weight
//   encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
//   encoder.block.{i}.layer.1.layer_norm.weight
//   encoder.final_layer_norm.weight

#include "ltx_common.hpp"

struct T5Config {
    int d_model       = 4096;   // hidden size
    int d_kv          = 64;     // key/value dim per head
    int num_heads     = 64;     // number of attention heads
    int d_ff          = 10240;  // feed-forward inner dim
    int num_layers    = 24;     // encoder layers
    int vocab_size    = 32128;  // vocabulary size
    int max_seq_len   = 512;    // maximum sequence length
    float eps         = 1e-6f;  // layer norm eps
};

// ── Simple BPE tokenizer (SentencePiece-compatible, loaded from GGUF) ────────

struct T5Tokenizer {
    std::vector<std::string> vocab;  // id → token string
    std::map<std::string, int> tok2id;
    int unk_id = 2, pad_id = 0, eos_id = 1;

    // Load vocabulary from GGUF KV array "tokenizer.ggml.tokens".
    bool load_from_gguf(struct gguf_context * gc) {
        int64_t kid = gguf_find_key(gc, "tokenizer.ggml.tokens");
        if (kid < 0) {
            LTX_ERR("T5 tokenizer: 'tokenizer.ggml.tokens' not found in GGUF");
            return false;
        }
        int64_t n = gguf_get_arr_n(gc, kid);
        vocab.resize(n);
        for (int64_t i = 0; i < n; ++i) {
            vocab[i] = gguf_get_arr_str(gc, kid, i);
            tok2id[vocab[i]] = static_cast<int>(i);
        }
        LTX_LOG("T5 tokenizer: loaded %lld tokens", (long long)n);
        return true;
    }

    // Naïve whitespace + subword tokenisation (SentencePiece ▁ prefix).
    // For production use, replace with a proper SentencePiece unigram tokenizer.
    std::vector<int> encode(const std::string & text, int max_len) const {
        std::vector<int> ids;

        // Split on whitespace and look up each word (incl. subword fall-back).
        std::string cur;
        auto flush = [&]() {
            if (cur.empty()) return;
            // prepend ▁ (U+2581 = \xe2\x96\x81)
            std::string tok = "\xe2\x96\x81" + cur;
            auto it = tok2id.find(tok);
            if (it != tok2id.end()) {
                ids.push_back(it->second);
            } else {
                // fall back character by character
                for (char c : cur) {
                    std::string ct(1, c);
                    auto it2 = tok2id.find(ct);
                    ids.push_back(it2 != tok2id.end() ? it2->second : unk_id);
                }
            }
            cur.clear();
        };

        for (char c : text) {
            if (c == ' ') { flush(); }
            else          { cur += c; }
        }
        flush();

        // Append EOS, pad / truncate to max_len.
        ids.push_back(eos_id);
        while ((int)ids.size() < max_len) ids.push_back(pad_id);
        if ((int)ids.size() > max_len)    ids.resize(max_len);

        return ids;
    }
};

// ── T5 encoder (GGML graph) ──────────────────────────────────────────────────

struct T5Encoder {
    T5Config cfg;
    T5Tokenizer tokenizer;

    // Pointers into the GGUF-loaded ggml_context – not owned.
    struct ggml_tensor * token_embed_weight = nullptr;

    struct LayerWeights {
        // Self-attention
        struct ggml_tensor * attn_q  = nullptr;
        struct ggml_tensor * attn_k  = nullptr;
        struct ggml_tensor * attn_v  = nullptr;
        struct ggml_tensor * attn_o  = nullptr;
        struct ggml_tensor * attn_ln = nullptr; // layer norm weight
        // FFN (SwiGLU-style: wi_0 gate, wi_1 value, wo output)
        struct ggml_tensor * ffn_wi0 = nullptr;
        struct ggml_tensor * ffn_wi1 = nullptr;
        struct ggml_tensor * ffn_wo  = nullptr;
        struct ggml_tensor * ffn_ln  = nullptr;
    };
    std::vector<LayerWeights> layers;
    struct ggml_tensor * final_ln = nullptr;

    // Load all weights from an open LtxGgufModel.
    bool load(LtxGgufModel & model) {
        // Read config from GGUF metadata when available.
        uint32_t nl = model.kv_u32("t5.block_count", 0);
        if (nl > 0) cfg.num_layers = (int)nl;
        uint32_t dm = model.kv_u32("t5.embedding_length", 0);
        if (dm > 0) cfg.d_model = (int)dm;
        uint32_t nh = model.kv_u32("t5.attention.head_count", 0);
        if (nh > 0) cfg.num_heads = (int)nh;
        uint32_t dff = model.kv_u32("t5.feed_forward_length", 0);
        if (dff > 0) cfg.d_ff = (int)dff;
        cfg.d_kv = cfg.d_model / cfg.num_heads;

        token_embed_weight = model.get_tensor("token_emb.weight");
        if (!token_embed_weight) {
            // Try alternative names used by llama.cpp T5 GGUF.
            token_embed_weight = model.get_tensor("token_embd.weight");
        }
        if (!token_embed_weight) {
            LTX_ERR("T5: cannot find token embedding weight");
            return false;
        }

        layers.resize(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) {
            auto & L = layers[i];
            char buf[256];
#define GET(field, fmt) \
    snprintf(buf, sizeof(buf), fmt, i); \
    L.field = model.get_tensor(buf);
            GET(attn_q,  "encoder.block.%d.layer.0.SelfAttention.q.weight");
            GET(attn_k,  "encoder.block.%d.layer.0.SelfAttention.k.weight");
            GET(attn_v,  "encoder.block.%d.layer.0.SelfAttention.v.weight");
            GET(attn_o,  "encoder.block.%d.layer.0.SelfAttention.o.weight");
            GET(attn_ln, "encoder.block.%d.layer.0.layer_norm.weight");
            GET(ffn_wi0, "encoder.block.%d.layer.1.DenseReluDense.wi_0.weight");
            GET(ffn_wi1, "encoder.block.%d.layer.1.DenseReluDense.wi_1.weight");
            GET(ffn_wo,  "encoder.block.%d.layer.1.DenseReluDense.wo.weight");
            GET(ffn_ln,  "encoder.block.%d.layer.1.layer_norm.weight");
#undef GET
        }

        final_ln = model.get_tensor("encoder.final_layer_norm.weight");

        // Tokenizer (optional – may be absent in text-encoder-only GGUFs).
        tokenizer.load_from_gguf(model.gguf_ctx);

        LTX_LOG("T5 encoder loaded: layers=%d d_model=%d d_ff=%d heads=%d",
                cfg.num_layers, cfg.d_model, cfg.d_ff, cfg.num_heads);
        return true;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────
    // Returns a float buffer [seq_len × d_model] allocated with new[].
    // Caller takes ownership.  hidden states are the T5 encoder outputs.
    std::vector<float> encode(const std::vector<int> & token_ids) const {
        int S = static_cast<int>(token_ids.size());
        int D = cfg.d_model;

        // Allocate a temporary ggml context for the computation graph.
        size_t ctx_bytes = 256 * 1024 * 1024; // 256 MB scratch
        struct ggml_init_params p{ ctx_bytes, nullptr, false };
        struct ggml_context * ctx = ggml_init(p);
        if (!ctx) LTX_ABORT("T5: ggml_init failed");

        // ── Token embeddings ─────────────────────────────────────────────────
        struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
        for (int i = 0; i < S; ++i) ((int32_t *)ids->data)[i] = token_ids[i];

        struct ggml_tensor * x; // [D, S]
        if (token_embed_weight) {
            x = ggml_get_rows(ctx, token_embed_weight, ids);
        } else {
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S);
            ggml_set_zero(x);
        }
        // x shape: [D, S]  (ggml: dim-0 is innermost / fastest)

        // ── Encoder layers ───────────────────────────────────────────────────
        for (int li = 0; li < cfg.num_layers; ++li) {
            const auto & L = layers[li];

            // ── Self-attention ────────────────────────────────────────────────
            struct ggml_tensor * nx = x;

            // RMS-norm (pre-norm)
            if (L.attn_ln) {
                nx = ggml_rms_norm(ctx, nx, cfg.eps);
                nx = ggml_mul(ctx, nx, ggml_repeat(ctx, L.attn_ln,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S)));
            }

            int H = cfg.num_heads, Dh = cfg.d_kv;
            if (L.attn_q && L.attn_k && L.attn_v && L.attn_o) {
                struct ggml_tensor * q = ggml_mul_mat(ctx, L.attn_q, nx); // [D, S]
                struct ggml_tensor * k = ggml_mul_mat(ctx, L.attn_k, nx);
                struct ggml_tensor * v = ggml_mul_mat(ctx, L.attn_v, nx);

                // Reshape to [Dh, H, S] then transpose for batched matmul.
                q = ggml_reshape_3d(ctx, q, Dh, H, S);
                k = ggml_reshape_3d(ctx, k, Dh, H, S);
                v = ggml_reshape_3d(ctx, v, Dh, H, S);

                // Scaled dot-product: attn = softmax(q @ k^T / sqrt(Dh)) @ v
                q = ggml_permute(ctx, q, 0, 2, 1, 3); // [Dh, S, H]
                k = ggml_permute(ctx, k, 0, 2, 1, 3);
                v = ggml_permute(ctx, v, 1, 2, 0, 3); // [S, Dh, H] -> transposed for output

                struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q); // [S, S, H]
                kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float)Dh));
                kq = ggml_soft_max(ctx, kq);

                struct ggml_tensor * attn_out = ggml_mul_mat(ctx, v, kq); // [Dh, S, H]
                attn_out = ggml_permute(ctx, attn_out, 0, 2, 1, 3);      // [Dh, H, S]
                attn_out = ggml_cont(ctx, attn_out);
                attn_out = ggml_reshape_2d(ctx, attn_out, D, S);          // [D, S]
                attn_out = ggml_mul_mat(ctx, L.attn_o, attn_out);

                x = ggml_add(ctx, x, attn_out); // residual
            }

            // ── FFN (SwiGLU) ──────────────────────────────────────────────────
            struct ggml_tensor * fx = x;
            if (L.ffn_ln) {
                fx = ggml_rms_norm(ctx, fx, cfg.eps);
                fx = ggml_mul(ctx, fx, ggml_repeat(ctx, L.ffn_ln,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S)));
            }
            if (L.ffn_wi0 && L.ffn_wi1 && L.ffn_wo) {
                struct ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_wi0, fx);
                struct ggml_tensor * val  = ggml_mul_mat(ctx, L.ffn_wi1, fx);
                gate = ggml_gelu(ctx, gate);
                struct ggml_tensor * ffn_out = ggml_mul(ctx, gate, val);
                ffn_out = ggml_mul_mat(ctx, L.ffn_wo, ffn_out);
                x = ggml_add(ctx, x, ffn_out);
            }
        }

        // Final layer norm.
        if (final_ln) {
            x = ggml_rms_norm(ctx, x, cfg.eps);
            x = ggml_mul(ctx, x, ggml_repeat(ctx, final_ln,
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S)));
        }

        // ── Execute graph ────────────────────────────────────────────────────
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, x);
        ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/4);

        // Copy result to output buffer.
        std::vector<float> out(S * D);
        memcpy(out.data(), x->data, S * D * sizeof(float));

        ggml_free(ctx);
        return out;
    }

    // Convenience: tokenize then encode.  Returns [seq_len × d_model].
    std::vector<float> encode_text(const std::string & text, int max_len = 256) const {
        auto ids = tokenizer.encode(text, max_len);
        return encode(ids);
    }
};
