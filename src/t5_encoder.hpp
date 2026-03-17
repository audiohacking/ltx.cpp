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

#include <algorithm>
#include <unordered_map>

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

// ── SentencePiece unigram tokenizer ──────────────────────────────────────────
//
// Implements the SentencePiece unigram algorithm used by T5:
//   - Text preprocessing: whitespace normalisation + ▁ (U+2581) insertion.
//   - Segmentation: Viterbi DP when unigram log-probability scores are present
//     in the GGUF (key "tokenizer.ggml.scores"); greedy longest-match otherwise.
//   - Fallback: characters with no vocabulary piece are emitted as unk_id,
//     advancing one full UTF-8 character to avoid splitting multi-byte sequences.
//
// Vocabulary and optional scores are read from GGUF metadata:
//   "tokenizer.ggml.tokens"  – string array: id → piece (UTF-8, ▁-prefixed)
//   "tokenizer.ggml.scores"  – float32 array: id → unigram log-probability

struct T5Tokenizer {
    std::vector<std::string>             vocab;          // id → piece
    std::vector<float>                   scores;         // id → log-prob (empty → greedy mode)
    std::unordered_map<std::string, int> tok2id;         // piece → id (O(1) lookup)
    int  unk_id        = 2;
    int  pad_id        = 0;
    int  eos_id        = 1;
    int  max_piece_len = 0;   // max byte-length of any vocabulary piece

    // Load vocabulary and (optional) scores from GGUF metadata.
    bool load_from_gguf(struct gguf_context * gc) {
        int64_t tokens_kid = gguf_find_key(gc, "tokenizer.ggml.tokens");
        if (tokens_kid < 0) {
            LTX_ERR("T5 tokenizer: 'tokenizer.ggml.tokens' not found in GGUF");
            return false;
        }
        size_t n = gguf_get_arr_n(gc, tokens_kid);
        vocab.resize(n);
        for (size_t i = 0; i < n; ++i) {
            vocab[i] = gguf_get_arr_str(gc, tokens_kid, i);
            tok2id[vocab[i]] = static_cast<int>(i);
            int len = static_cast<int>(vocab[i].size());
            if (len > max_piece_len) max_piece_len = len;
        }

        // Optional: unigram log-probability scores → enables Viterbi mode.
        int64_t scores_kid = gguf_find_key(gc, "tokenizer.ggml.scores");
        if (scores_kid >= 0 &&
                gguf_get_arr_type(gc, scores_kid) == GGUF_TYPE_FLOAT32) {
            size_t ns = gguf_get_arr_n(gc, scores_kid);
            const float * raw = reinterpret_cast<const float *>(
                gguf_get_arr_data(gc, scores_kid));
            if (raw) scores.assign(raw, raw + ns);
        }

        LTX_LOG("T5 tokenizer: loaded %zu tokens, max_piece=%d bytes, mode=%s",
                n, max_piece_len, scores.empty() ? "greedy" : "Viterbi");
        return true;
    }

    // SentencePiece text normalisation:
    //   1. Collapse runs of whitespace to a single space; strip leading/trailing.
    //   2. Prepend ▁ and replace each remaining space with ▁.
    static std::string preprocess(const std::string & text) {
        // Step 1: collapse and strip.
        std::string stripped;
        stripped.reserve(text.size());
        bool prev_ws = true;  // treat start as whitespace to drop leading ws
        for (unsigned char c : text) {
            bool is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
            if (is_ws) {
                if (!prev_ws) stripped += ' ';
            } else {
                stripped += static_cast<char>(c);
            }
            prev_ws = is_ws;
        }
        while (!stripped.empty() && stripped.back() == ' ') stripped.pop_back();

        // Step 2: insert ▁ (U+2581 = \xe2\x96\x81, 3 bytes).
        static const char SPIECE[4] = "\xe2\x96\x81";
        std::string out;
        out.reserve(stripped.size() * 2);
        out.append(SPIECE, 3);          // always prepend ▁
        for (char c : stripped) {
            if (c == ' ') out.append(SPIECE, 3);
            else          out += c;
        }
        return out;
    }

    // Return the byte-length of the UTF-8 character whose first byte is `b`.
    static int utf8_char_len(unsigned char b) {
        if (b < 0x80)            return 1;   // 0xxxxxxx – ASCII
        if ((b & 0xE0) == 0xC0)  return 2;   // 110xxxxx – 2-byte
        if ((b & 0xF0) == 0xE0)  return 3;   // 1110xxxx – 3-byte (e.g. ▁)
        if ((b & 0xF8) == 0xF0)  return 4;   // 11110xxx – 4-byte
        return 1;                             // invalid continuation byte: skip
    }

    // Viterbi optimal segmentation maximising the sum of unigram log-probs.
    std::vector<int> viterbi(const std::string & text) const {
        int n = static_cast<int>(text.size());
        if (n == 0) return {};

        constexpr float NEG_INF = -1e38f;
        // best[i]: best total score for text[0..i)
        std::vector<float>             best(n + 1, NEG_INF);
        // from[i]: {prev_position, token_id} that achieves best[i]
        std::vector<std::pair<int,int>> from(n + 1, {-1, -1});
        best[0] = 0.0f;

        for (int i = 0; i < n; ++i) {
            if (best[i] <= NEG_INF / 2.0f) continue;
            int  max_len   = std::min(max_piece_len, n - i);
            bool any_match = false;
            for (int len = 1; len <= max_len; ++len) {
                auto it = tok2id.find(text.substr(i, len));
                if (it == tok2id.end()) continue;
                int   tok      = it->second;
                float sc       = (tok < static_cast<int>(scores.size()))
                                 ? scores[tok] : 0.0f;
                float new_best = best[i] + sc;
                if (new_best > best[i + len]) {
                    best[i + len] = new_best;
                    from[i + len] = {i, tok};
                }
                any_match = true;
            }
            // No vocabulary piece covers position i: skip one UTF-8 char as unk.
            if (!any_match) {
                int skip = std::min(utf8_char_len(
                    static_cast<unsigned char>(text[i])), n - i);
                constexpr float UNK_PENALTY = -10.0f;
                if (best[i] + UNK_PENALTY > best[i + skip]) {
                    best[i + skip] = best[i] + UNK_PENALTY;
                    from[i + skip] = {i, unk_id};
                }
            }
        }

        // Backtrack from position n.
        std::vector<int> ids;
        for (int pos = n; pos > 0;) {
            auto [prev, tok] = from[pos];
            if (prev < 0) { ids.push_back(unk_id); break; }
            ids.push_back(tok);
            pos = prev;
        }
        std::reverse(ids.begin(), ids.end());
        return ids;
    }

    // Greedy longest-match segmentation (fallback when scores are absent).
    std::vector<int> greedy(const std::string & text) const {
        std::vector<int> ids;
        int n   = static_cast<int>(text.size());
        int pos = 0;
        while (pos < n) {
            int  max_len = std::min(max_piece_len, n - pos);
            bool found   = false;
            for (int len = max_len; len >= 1; --len) {
                auto it = tok2id.find(text.substr(pos, len));
                if (it != tok2id.end()) {
                    ids.push_back(it->second);
                    pos  += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                ids.push_back(unk_id);
                pos += std::min(utf8_char_len(
                    static_cast<unsigned char>(text[pos])), n - pos);
            }
        }
        return ids;
    }

    // Tokenise text; pad or truncate to max_len (EOS is appended before padding).
    std::vector<int> encode(const std::string & text, int max_len) const {
        std::string      processed = preprocess(text);
        std::vector<int> ids       = scores.empty() ? greedy(processed)
                                                    : viterbi(processed);
        ids.push_back(eos_id);
        while (static_cast<int>(ids.size()) < max_len) ids.push_back(pad_id);
        if  (static_cast<int>(ids.size()) > max_len)   ids.resize(max_len);
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
