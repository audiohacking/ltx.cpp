#pragma once

// ltx_dit.hpp – LTX-Video DiT (Diffusion Transformer) in C++ / GGML
//
// Architecture overview (LTXV):
//  - Video latents are patchified into tokens: patch_size = (1, 2, 2)
//  - Each token gets 3-D RoPE positional embeddings (t, h, w)
//  - N transformer blocks, each with:
//      • Self-attention with AdaLN modulation (scale/shift from timestep emb)
//      • Cross-attention to text encoder output
//      • FFN (SwiGLU)
//  - Output is unpatchified back to latent shape
//
// GGUF tensor name conventions (mirrors ComfyUI / diffusers naming):
//   dit.time_embedding.{linear_1,linear_2}.{weight,bias}
//   dit.patchify_proj.{weight,bias}
//   dit.adaln_single.{linear,emb.timestep_embedder.*}
//   dit.caption_projection.{weight,bias}
//   dit.transformer_blocks.{i}.{attn1,attn2,ff}.{...}
//   dit.proj_out.{weight,bias}

#include "ltx_common.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>


// ── LTX DiT config ───────────────────────────────────────────────────────────

struct DiTConfig {
    int hidden_size      = 2048;  // transformer hidden dim
    int num_layers       = 28;    // number of transformer blocks
    int num_heads        = 32;    // attention heads
    int head_dim         = 64;    // dim per head
    int cross_attn_dim   = 4096;  // text encoder output dim (T5-XXL)
    int patch_t          = 1;     // temporal patch size
    int patch_h          = 2;     // height patch size
    int patch_w          = 2;     // width patch size
    int latent_channels  = 128;   // VAE latent channels
    int freq_dim         = 256;   // sinusoidal embedding dim
    float norm_eps       = 1e-6f;
    // Derived
    int patch_dim() const { return patch_t * patch_h * patch_w * latent_channels; }
};

// ── Sinusoidal timestep embedding ────────────────────────────────────────────

static std::vector<float> sinusoidal_embedding(float t, int dim) {
    std::vector<float> emb(dim);
    int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq = std::exp(-std::log(10000.0f) * i / (half - 1));
        emb[i]        = std::cos(t * freq);
        emb[i + half] = std::sin(t * freq);
    }
    return emb;
}

// ── AdaLN-single: compute scale/shift from timestep conditioning ─────────────

struct AdaLNSingle {
    // Linear layers: [hidden_size → 6*hidden_size] (scale/shift for Q,K,V in attn; FFN)
    struct ggml_tensor * linear_w = nullptr;
    struct ggml_tensor * linear_b = nullptr;

    // Timestep MLP: emb → hidden_size
    struct ggml_tensor * emb_w1 = nullptr, * emb_b1 = nullptr; // linear_1
    struct ggml_tensor * emb_w2 = nullptr, * emb_b2 = nullptr; // linear_2
};

// ── Transformer block weights ─────────────────────────────────────────────────

struct DiTBlock {
    // Self-attention (with AdaLN)
    struct ggml_tensor * norm1_w = nullptr, * norm1_b = nullptr;
    struct ggml_tensor * attn1_q = nullptr, * attn1_q_b = nullptr;
    struct ggml_tensor * attn1_k = nullptr, * attn1_k_b = nullptr;
    struct ggml_tensor * attn1_v = nullptr, * attn1_v_b = nullptr;
    struct ggml_tensor * attn1_o = nullptr, * attn1_o_b = nullptr;
    // Optional: per-block AdaLN scale/shift projections
    struct ggml_tensor * adaln_w = nullptr, * adaln_b = nullptr;

    // Cross-attention
    struct ggml_tensor * norm2_w = nullptr, * norm2_b = nullptr;
    struct ggml_tensor * attn2_q = nullptr, * attn2_q_b = nullptr;
    struct ggml_tensor * attn2_k = nullptr, * attn2_k_b = nullptr;
    struct ggml_tensor * attn2_v = nullptr, * attn2_v_b = nullptr;
    struct ggml_tensor * attn2_o = nullptr, * attn2_o_b = nullptr;

    // FFN (SwiGLU: gate, up, down)
    struct ggml_tensor * norm3_w = nullptr, * norm3_b = nullptr;
    struct ggml_tensor * ff_gate = nullptr, * ff_gate_b = nullptr;
    struct ggml_tensor * ff_up   = nullptr, * ff_up_b   = nullptr;
    struct ggml_tensor * ff_down = nullptr, * ff_down_b = nullptr;
};

// ── LTX DiT model ─────────────────────────────────────────────────────────────

struct LtxDiT {
    DiTConfig cfg;
    AdaLNSingle adaln;

    struct ggml_tensor * patch_embed_w  = nullptr; // patchify projection weight
    struct ggml_tensor * patch_embed_b  = nullptr;
    struct ggml_tensor * cap_proj_w     = nullptr; // caption projection
    struct ggml_tensor * cap_proj_b     = nullptr;
    struct ggml_tensor * proj_out_w     = nullptr; // output unpatchify
    struct ggml_tensor * proj_out_b     = nullptr;
    struct ggml_tensor * final_norm_w   = nullptr; // final layer norm
    struct ggml_tensor * final_norm_b   = nullptr;

    std::vector<DiTBlock> blocks;

    // ── Load weights from GGUF model ─────────────────────────────────────────
    // Supports both LTX-Video (ltxv.*) and LTX 2.3 (ltx.*) metadata.
    bool load(LtxGgufModel & model) {
        // Read config from GGUF metadata. Try ltxv.* first, then ltx.* for LTX 2.3.
        uint32_t hs = model.kv_u32("ltxv.hidden_size", 0);
        if (hs == 0) hs = model.kv_u32("ltx.hidden_size", 0);
        if (hs > 0) cfg.hidden_size = (int)hs;
        uint32_t nl = model.kv_u32("ltxv.num_hidden_layers", 0);
        if (nl == 0) nl = model.kv_u32("ltx.num_hidden_layers", 0);
        if (nl > 0) cfg.num_layers = (int)nl;
        uint32_t nh = model.kv_u32("ltxv.num_attention_heads", 0);
        if (nh == 0) nh = model.kv_u32("ltx.num_attention_heads", 0);
        if (nh > 0) cfg.num_heads = (int)nh;
        uint32_t lc = model.kv_u32("ltxv.in_channels", 0);
        if (lc == 0) lc = model.kv_u32("ltx.in_channels", 0);
        if (lc > 0) cfg.latent_channels = (int)lc;
        uint32_t cross = model.kv_u32("ltxv.cross_attention_dim", 0);
        if (cross == 0) cross = model.kv_u32("ltx.cross_attention_dim", 0);
        if (cross > 0) cfg.cross_attn_dim = (int)cross;
        cfg.head_dim = cfg.hidden_size / cfg.num_heads;

        auto get = [&](const char * nm) { return model.get_tensor(nm); };

        // Patchify projection (Pd → D). Try ComfyUI/diffusers and stripped-prefix names.
        patch_embed_w = get("model.diffusion_model.patchify_proj.weight");
        patch_embed_b = get("model.diffusion_model.patchify_proj.bias");
        if (!patch_embed_w) {
            patch_embed_w = get("dit.patchify_proj.weight");
            patch_embed_b = get("dit.patchify_proj.bias");
        }
        if (!patch_embed_w) {
            patch_embed_w = get("patchify_proj.weight");
            patch_embed_b = get("patchify_proj.bias");
        }
        if (!patch_embed_w) {
            patch_embed_w = get("patch_embed.proj.weight");
            patch_embed_b = get("patch_embed.proj.bias");
        }
        if (!patch_embed_w) {
            patch_embed_w = get("x_embedder.proj.weight");
            patch_embed_b = get("x_embedder.proj.bias");
        }
        if (!patch_embed_w) {
            patch_embed_w = get("input_proj.weight");
            patch_embed_b = get("input_proj.bias");
        }
        // Fallback: GGUF may use different naming; find any tensor with shape (Pd, D) or (D, Pd).
        if (!patch_embed_w && model.gguf_ctx) {
            int64_t pd = (int64_t)cfg.patch_dim(), d = (int64_t)cfg.hidden_size;
            for (int64_t i = 0, n = gguf_get_n_tensors(model.gguf_ctx); i < n; ++i) {
                const char * name = gguf_get_tensor_name(model.gguf_ctx, i);
                struct ggml_tensor * t = model.get_tensor(name);
                if (t && ((t->ne[0] == pd && t->ne[1] == d) || (t->ne[0] == d && t->ne[1] == pd))) {
                    patch_embed_w = t;
                    std::string n(name);
                    size_t w = n.rfind(".weight");
                    if (w != std::string::npos)
                        patch_embed_b = model.get_tensor((n.replace(w, 7, ".bias")).c_str());
                    break;
                }
            }
        }
        // Accept (Pd, D) or (D, Pd). For 22B, patchify_proj is (128, 4096) in ggml → Pd=128, D=4096; infer if GGUF has no ltxv.hidden_size.
        {
            if (patch_embed_w) {
                int64_t pw0 = patch_embed_w->ne[0], pw1 = patch_embed_w->ne[1];
                int64_t d = (int64_t)cfg.hidden_size;
                int64_t pd_cfg = (int64_t)cfg.patch_dim();
                bool exact = (pw0 == pd_cfg && pw1 == d) || (pw0 == d && pw1 == pd_cfg);
                // One dimension is hidden_size (D), the other is patch_dim (Pd). GGUF may omit ltxv.hidden_size so infer from weight.
                int64_t inferred_d = (pw0 == d ? d : (pw1 == d ? d : 0));
                int64_t inferred_pd = (pw0 == d ? pw1 : (pw1 == d ? pw0 : 0));
                if (inferred_d == 0 && pw0 > 0 && pw1 > 0 && cfg.latent_channels > 0) {
                    // Neither dim matched cfg; infer D and Pd from weight (e.g. 22B: ne=128,4096 → Pd=128, D=4096).
                    if ((pw0 % (int64_t)cfg.latent_channels) == 0 && (pw1 % (int64_t)cfg.latent_channels) != 0) {
                        inferred_pd = pw0; inferred_d = pw1;
                    } else if ((pw1 % (int64_t)cfg.latent_channels) == 0 && (pw0 % (int64_t)cfg.latent_channels) != 0) {
                        inferred_pd = pw1; inferred_d = pw0;
                    } else if (pw0 == (int64_t)cfg.latent_channels || pw1 == (int64_t)cfg.latent_channels) {
                        inferred_pd = (pw0 == (int64_t)cfg.latent_channels ? pw0 : pw1);
                        inferred_d = (pw0 == (int64_t)cfg.latent_channels ? pw1 : pw0);
                    }
                }
                bool one_is_d = (inferred_d > 0 && (pw0 == inferred_d || pw1 == inferred_d));
                if (exact) {
                    // No change.
                } else if (one_is_d && inferred_pd > 0 && inferred_d > 0 && cfg.latent_channels > 0 &&
                           (inferred_pd % (int64_t)cfg.latent_channels) == 0) {
                    if (cfg.hidden_size != (int)inferred_d) {
                        cfg.hidden_size = (int)inferred_d;
                        if (cfg.num_heads > 0) cfg.head_dim = cfg.hidden_size / cfg.num_heads;
                        else { cfg.num_heads = (cfg.hidden_size <= 2048 ? 32 : 64); cfg.head_dim = cfg.hidden_size / cfg.num_heads; }
                    }
                    int64_t p = inferred_pd / (int64_t)cfg.latent_channels;
                    if (p == 1) {
                        cfg.patch_t = 1; cfg.patch_h = 1; cfg.patch_w = 1;
                    } else if (p == 4) {
                        cfg.patch_t = 1; cfg.patch_h = 2; cfg.patch_w = 2;
                    }
                } else if (!exact) {
                    patch_embed_w = nullptr;
                    patch_embed_b = nullptr;
                }
            }
        }
        if (!patch_embed_w && model.gguf_ctx) {
            LTX_ERR("DiT: patchify projection weight not found.");
            int64_t pd = (int64_t)cfg.patch_dim(), d = (int64_t)cfg.hidden_size;
            LTX_ERR("Candidates (shape Pd×D or D×Pd = %lld×%lld):", (long long)pd, (long long)d);
            for (int64_t i = 0, n = gguf_get_n_tensors(model.gguf_ctx); i < n; ++i) {
                const char * name = gguf_get_tensor_name(model.gguf_ctx, i);
                struct ggml_tensor * t = model.get_tensor(name);
                if (!t || !((t->ne[0] == pd && t->ne[1] == d) || (t->ne[0] == d && t->ne[1] == pd))) continue;
                (void)fprintf(stderr, "[ltx]   %s  ne=%lld,%lld\n", name, (long long)t->ne[0], (long long)t->ne[1]);
            }
            LTX_ERR("All tensors whose name contains 'proj' (look for patchify):");
            for (int64_t i = 0, n = gguf_get_n_tensors(model.gguf_ctx); i < n; ++i) {
                const char * name = gguf_get_tensor_name(model.gguf_ctx, i);
                if (!std::strstr(name, "proj")) continue;
                struct ggml_tensor * t = model.get_tensor(name);
                if (t)
                    (void)fprintf(stderr, "[ltx]   %s  ne=%lld,%lld,%lld,%lld\n", name,
                        (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
            }
            LTX_ABORT("DiT: no (512,2048) tensor found. Use a GGUF from this repo's convert.py, or add the key from the 'proj' list above.");
        }

        // Caption projection.
        cap_proj_w = get("model.diffusion_model.caption_projection.weight");
        cap_proj_b = get("model.diffusion_model.caption_projection.bias");
        if (!cap_proj_w) {
            cap_proj_w = get("dit.caption_projection.weight");
            cap_proj_b = get("dit.caption_projection.bias");
        }

        // AdaLN-single timestep embedder (ComfyUI GGUF may use no prefix).
        adaln.emb_w1 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight");
        adaln.emb_b1 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias");
        adaln.emb_w2 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.weight");
        adaln.emb_b2 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.bias");
        adaln.linear_w = get("model.diffusion_model.adaln_single.linear.weight");
        adaln.linear_b = get("model.diffusion_model.adaln_single.linear.bias");
        if (!adaln.emb_w1) {
            adaln.emb_w1 = get("adaln_single.emb.timestep_embedder.linear_1.weight");
            adaln.emb_b1 = get("adaln_single.emb.timestep_embedder.linear_1.bias");
            adaln.emb_w2 = get("adaln_single.emb.timestep_embedder.linear_2.weight");
            adaln.emb_b2 = get("adaln_single.emb.timestep_embedder.linear_2.bias");
            adaln.linear_w = get("adaln_single.linear.weight");
            adaln.linear_b = get("adaln_single.linear.bias");
        }

        // Output projection (22B GGUF uses proj_out.weight, norm_out.linear.weight without prefix).
        proj_out_w   = get("model.diffusion_model.proj_out.weight");
        proj_out_b   = get("model.diffusion_model.proj_out.bias");
        if (!proj_out_w) { proj_out_w = get("proj_out.weight"); proj_out_b = get("proj_out.bias"); }
        final_norm_w = get("model.diffusion_model.norm_out.linear.weight");
        final_norm_b = get("model.diffusion_model.norm_out.linear.bias");
        if (!final_norm_w) { final_norm_w = get("norm_out.linear.weight"); final_norm_b = get("norm_out.linear.bias"); }

        // Transformer blocks (22B GGUF uses "transformer_blocks." without model.diffusion_model. prefix).
        const char * block_prefix = get("transformer_blocks.0.norm1.weight")
            ? "transformer_blocks." : "model.diffusion_model.transformer_blocks.";
        if (std::strstr(block_prefix, "transformer_blocks.") && !std::strstr(block_prefix, "model.")) {
            int n = 0;
            char lbuf[128];
            for (; n < 256; ++n) {
                (void)snprintf(lbuf, sizeof(lbuf), "transformer_blocks.%d.norm1.weight", n);
                if (!get(lbuf)) break;
            }
            if (n > 0) cfg.num_layers = n;
        }
        blocks.resize(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) {
            auto & B = blocks[i];
            char buf[384];
            std::string pre = block_prefix;
#define GET(field, fmt, ...) \
    snprintf(buf, sizeof(buf), (pre + fmt).c_str(), i, ##__VA_ARGS__); \
    B.field = get(buf);

            GET(norm1_w,  "%d.norm1.weight");
            GET(norm1_b,  "%d.norm1.bias");
            GET(adaln_w,  "%d.scale_shift_table");     // combined AdaLN params
            GET(attn1_q,  "%d.attn1.to_q.weight");
            GET(attn1_q_b,"%d.attn1.to_q.bias");
            GET(attn1_k,  "%d.attn1.to_k.weight");
            GET(attn1_k_b,"%d.attn1.to_k.bias");
            GET(attn1_v,  "%d.attn1.to_v.weight");
            GET(attn1_v_b,"%d.attn1.to_v.bias");
            GET(attn1_o,  "%d.attn1.to_out.0.weight");
            GET(attn1_o_b,"%d.attn1.to_out.0.bias");
            GET(norm2_w,  "%d.norm2.weight");
            GET(norm2_b,  "%d.norm2.bias");
            GET(attn2_q,  "%d.attn2.to_q.weight");
            GET(attn2_q_b,"%d.attn2.to_q.bias");
            GET(attn2_k,  "%d.attn2.to_k.weight");
            GET(attn2_k_b,"%d.attn2.to_k.bias");
            GET(attn2_v,  "%d.attn2.to_v.weight");
            GET(attn2_v_b,"%d.attn2.to_v.bias");
            GET(attn2_o,  "%d.attn2.to_out.0.weight");
            GET(attn2_o_b,"%d.attn2.to_out.0.bias");
            GET(norm3_w,  "%d.ff.net.0.weight");
            GET(ff_gate,  "%d.ff.net.0.proj.weight");
            GET(ff_gate_b,"%d.ff.net.0.proj.bias");
            GET(ff_up,    "%d.ff.net.0.proj.weight");  // SwiGLU: proj packs gate+up
            GET(ff_down,  "%d.ff.net.2.weight");
            GET(ff_down_b,"%d.ff.net.2.bias");
#undef GET
        }

        LTX_LOG("DiT loaded: layers=%d hidden=%d heads=%d head_dim=%d",
                cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.head_dim);
        return true;
    }

    // ── Forward pass (CPU, float32) ───────────────────────────────────────────
    //
    // Inputs:
    //   latents:       [N_tok, patch_dim]  (patchified video latent)
    //   text_emb:      [S, cross_dim]      (T5 encoder output)
    //   timestep:      scalar in [0,1]     (noise level)
    //   n_tok:         number of latent patches
    //   seq_len:       text sequence length
    //   scratch_buf:   persistent buffer for ggml (reused each call; use dit_scratch_size_bytes()).
    //   scratch_size: size of scratch_buf in bytes.
    //   backend:       optional GGML backend (from ggml_backend_init_best()); if non-null, inference runs on backend (e.g. Metal/CUDA).
    //
    // Returns predicted noise/velocity: [N_tok × patch_dim]
    // Chunked (pre → each block → post), same scratch_buf reused; peak ~20 GB like ComfyUI.
    std::vector<float> forward(
            const float * latents,   int n_tok,
            const float * text_emb,  int seq_len,
            float timestep,
            void * scratch_buf, size_t scratch_size,
            ggml_backend_t backend = nullptr,
            bool show_blocks = false) const
    {
        (void)timestep;
        if (!scratch_buf || scratch_size == 0) { LTX_ERR("DiT forward: scratch_buf and scratch_size required"); return {}; }
        int D  = cfg.hidden_size;
        int Pd = cfg.patch_dim();
        int Cd = cfg.cross_attn_dim;
        int H  = cfg.num_heads;
        int Dh = cfg.head_dim;

        auto w_for_mul = [](struct ggml_context * ctx, struct ggml_tensor * W, struct ggml_tensor * src) -> struct ggml_tensor * {
            if (!W) return nullptr;
            if (W->ne[0] == src->ne[0]) return W;
            return ggml_cont(ctx, ggml_transpose(ctx, W));
        };

        struct ggml_init_params params;
        if (backend) {
            static const size_t LTX_DIT_MAX_GRAPH_NODES = 8192;
            static size_t no_alloc_size = 0;
            static std::vector<uint8_t> no_alloc_buf;
            if (no_alloc_buf.empty()) {
                no_alloc_size = ggml_tensor_overhead() * LTX_DIT_MAX_GRAPH_NODES + ggml_graph_overhead_custom(LTX_DIT_MAX_GRAPH_NODES, false);
                no_alloc_buf.resize(no_alloc_size);
            }
            params = { no_alloc_size, no_alloc_buf.data(), true };
        } else {
            params = { scratch_size, scratch_buf, false };
        }

        // ── Pre: patch_embed + cap_proj ───────────────────────────────────────
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) { LTX_ERR("DiT: ggml_init failed"); return {}; }
        struct ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Pd, n_tok);
        struct ggml_tensor * x = x_in;
        if (!backend) memcpy(x->data, latents, (size_t)n_tok * Pd * sizeof(float));
        if (patch_embed_w) {
            x = ggml_mul_mat(ctx, w_for_mul(ctx, patch_embed_w, x), x);
            x = ggml_cont(ctx, x);
            if (patch_embed_b) {
                struct ggml_tensor * b2 = ggml_repeat(ctx, patch_embed_b,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)x->ne[0], (int)x->ne[1]));
                x = ggml_add(ctx, x, b2);
            }
        }
        struct ggml_tensor * ctx_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cd, seq_len);
        struct ggml_tensor * ctx_emb = ctx_emb_in;
        if (!backend) memcpy(ctx_emb->data, text_emb, (size_t)seq_len * Cd * sizeof(float));
        if (cap_proj_w) {
            ctx_emb = ggml_mul_mat(ctx, w_for_mul(ctx, cap_proj_w, ctx_emb), ctx_emb);
            ctx_emb = ggml_cont(ctx, ctx_emb);
            if (cap_proj_b) {
                struct ggml_tensor * b2 = ggml_repeat(ctx, cap_proj_b,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)ctx_emb->ne[0], (int)ctx_emb->ne[1]));
                ctx_emb = ggml_add(ctx, ctx_emb, b2);
            }
        }
        struct ggml_cgraph * gf_pre = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_pre, x);
        ggml_build_forward_expand(gf_pre, ctx_emb);
        std::vector<float> x_host((size_t)D * n_tok);
        std::vector<float> ctx_emb_host((size_t)D * seq_len);
        if (backend) {
            ggml_backend_buffer_t pre_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!pre_buf) { ggml_free(ctx); return {}; }
            ggml_backend_tensor_set(x_in, latents, 0, (size_t)n_tok * Pd * sizeof(float));
            ggml_backend_tensor_set(ctx_emb_in, text_emb, 0, (size_t)seq_len * Cd * sizeof(float));
            if (ggml_backend_graph_compute(backend, gf_pre) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(pre_buf); ggml_free(ctx); return {}; }
            ggml_backend_tensor_get(x, x_host.data(), 0, (size_t)D * n_tok * sizeof(float));
            ggml_backend_tensor_get(ctx_emb, ctx_emb_host.data(), 0, (size_t)D * seq_len * sizeof(float));
            ggml_backend_buffer_free(pre_buf);
        } else {
            ggml_graph_compute_with_ctx(ctx, gf_pre, 4);
            memcpy(x_host.data(), x->data, (size_t)D * n_tok * sizeof(float));
            memcpy(ctx_emb_host.data(), ctx_emb->data, (size_t)D * seq_len * sizeof(float));
        }
        if (backend) ggml_free(ctx); else ggml_reset(ctx);

        // Query chunk size to avoid materializing n_tok×n_tok attention (ComfyUI uses flash/memory-efficient attn).
        // kq is (Dh,Nkv,chunk) = 128*36960*chunk*4 bytes; 256 -> ~4.8 GB, with soft_max copy fits in 21 GB.
        const int ATTN_Q_CHUNK = 256;
        auto proj_3d = [&](struct ggml_context * c, struct ggml_tensor * W, struct ggml_tensor * b,
                          struct ggml_tensor * src, int N) -> struct ggml_tensor * {
            if (!W) return ggml_new_tensor_3d(c, GGML_TYPE_F32, Dh, H, N);
            struct ggml_tensor * out = ggml_mul_mat(c, w_for_mul(c, W, src), src);
            out = ggml_cont(c, out);
            if (b) {
                struct ggml_tensor * b2 = ggml_repeat(c, b,
                    ggml_new_tensor_2d(c, GGML_TYPE_F32, (int)out->ne[0], (int)out->ne[1]));
                out = ggml_add(c, out, b2);
            }
            return ggml_reshape_3d(c, out, Dh, H, N);
        };
        auto attn = [&](struct ggml_context * c, struct ggml_tensor * q_src, struct ggml_tensor * kv_src,
                        int Nq, int Nkv,
                        struct ggml_tensor * Wq, struct ggml_tensor * Wk,
                        struct ggml_tensor * Wv, struct ggml_tensor * Wo,
                        struct ggml_tensor * bq, struct ggml_tensor * bk,
                        struct ggml_tensor * bv, struct ggml_tensor * bo,
                        float * out_host,
                        const float * q_src_host)  // when non-null: chunked path uses this instead of viewing q_src (q_src not yet computed)
            -> struct ggml_tensor *
        {
            if (out_host && q_src_host && Nq > ATTN_Q_CHUNK && Nkv == Nq) {
                // Chunked self-attention: K,V once, then Q in chunks (avoids n_tok×n_tok matrix).
                if (backend) { ctx = ggml_init(params); if (!ctx) return nullptr; }
                // For the backend path, kv_src may live in a previously freed Metal buffer
                // (e.g. the gnx scratch freed just before this call). Create a fresh input
                // tensor and upload from the host copy that the caller already computed.
                struct ggml_tensor * kv_in;
                if (backend) {
                    kv_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Nkv);
                } else {
                    kv_in = kv_src;
                }
                struct ggml_tensor * k = proj_3d(ctx, Wk, bk, kv_in, Nkv);
                struct ggml_tensor * v = proj_3d(ctx, Wv, bv, kv_in, Nkv);
                k = ggml_permute(ctx, k, 0, 2, 1, 3);
                v = ggml_permute(ctx, v, 1, 2, 0, 3);
                k = ggml_cont(ctx, k);
                v = ggml_cont(ctx, v);
                struct ggml_cgraph * gkv = ggml_new_graph(ctx);
                ggml_build_forward_expand(gkv, k);
                ggml_build_forward_expand(gkv, v);
                std::vector<float> k_host((size_t)Dh * Nkv * H);
                std::vector<float> v_host((size_t)Dh * Nkv * H);
                if (backend) {
                    ggml_backend_buffer_t kv_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (!kv_buf) { ggml_free(ctx); return nullptr; }
                    // kv_in is a fresh tensor; populate from the host copy of kv_src
                    ggml_backend_tensor_set(kv_in, q_src_host, 0, (size_t)Nkv * D * sizeof(float));
                    if (ggml_backend_graph_compute(backend, gkv) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(kv_buf); ggml_free(ctx); return nullptr; }
                    ggml_backend_tensor_get(k, k_host.data(), 0, k_host.size() * sizeof(float));
                    ggml_backend_tensor_get(v, v_host.data(), 0, v_host.size() * sizeof(float));
                    ggml_backend_buffer_free(kv_buf);
                } else {
                    ggml_graph_compute_with_ctx(ctx, gkv, 4);
                    memcpy(k_host.data(), k->data, k_host.size() * sizeof(float));
                    memcpy(v_host.data(), v->data, v_host.size() * sizeof(float));
                }
                if (backend) ggml_free(ctx); else ggml_reset(ctx);

                for (int start = 0; start < Nq; start += ATTN_Q_CHUNK) {
                    int len = (start + ATTN_Q_CHUNK <= Nq) ? ATTN_Q_CHUNK : (Nq - start);
                    if (backend) { ctx = ggml_init(params); if (!ctx) return nullptr; }
                    struct ggml_tensor * q_src_chunk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, len);
                    if (!backend) memcpy(q_src_chunk->data, q_src_host + (size_t)start * D, (size_t)len * D * sizeof(float));
                    struct ggml_tensor * q_chunk = proj_3d(ctx, Wq, bq, q_src_chunk, len);
                    q_chunk = ggml_permute(ctx, q_chunk, 0, 2, 1, 3);
                    q_chunk = ggml_cont(ctx, q_chunk);
                    struct ggml_tensor * k_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dh, Nkv, H);
                    struct ggml_tensor * v_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Nkv, Dh, H);
                    if (!backend) {
                        memcpy(k_t->data, k_host.data(), k_host.size() * sizeof(float));
                        for (int64_t i = 0; i < Nkv; ++i)
                            for (int64_t j = 0; j < Dh; ++j)
                                for (int h = 0; h < H; ++h)
                                    ((float *)v_t->data)[(i * Dh + j) * H + h] = ((const float *)v_host.data())[(h * Nkv + i) * Dh + j];
                    }
                    struct ggml_tensor * kq = ggml_mul_mat(ctx, k_t, q_chunk);
                    kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float)Dh));
                    kq = ggml_soft_max(ctx, kq);
                    struct ggml_tensor * out_c = ggml_mul_mat(ctx, v_t, kq);
                    out_c = ggml_permute(ctx, out_c, 0, 2, 1, 3);
                    out_c = ggml_cont(ctx, out_c);
                    out_c = ggml_reshape_2d(ctx, out_c, D, len);
                    if (Wo) {
                        out_c = ggml_mul_mat(ctx, w_for_mul(ctx, Wo, out_c), out_c);
                        out_c = ggml_cont(ctx, out_c);
                        if (bo) {
                            struct ggml_tensor * b2 = ggml_repeat(ctx, bo,
                                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)out_c->ne[0], (int)out_c->ne[1]));
                            out_c = ggml_add(ctx, out_c, b2);
                        }
                    }
                    struct ggml_cgraph * gch = ggml_new_graph(ctx);
                    ggml_build_forward_expand(gch, out_c);
                    if (backend) {
                        ggml_backend_buffer_t ch_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                        if (!ch_buf) { ggml_free(ctx); return nullptr; }
                        ggml_backend_tensor_set(q_src_chunk, q_src_host + (size_t)start * D, 0, (size_t)len * D * sizeof(float));
                        ggml_backend_tensor_set(k_t, k_host.data(), 0, k_host.size() * sizeof(float));
                        std::vector<float> v_t_layout((size_t)Nkv * Dh * H);
                        for (int64_t i = 0; i < Nkv; ++i)
                            for (int64_t j = 0; j < Dh; ++j)
                                for (int h = 0; h < H; ++h)
                                    v_t_layout[(size_t)((i * Dh + j) * H + h)] = v_host.data()[(size_t)((h * Nkv + i) * Dh + j)];
                        ggml_backend_tensor_set(v_t, v_t_layout.data(), 0, v_t_layout.size() * sizeof(float));
                        if (ggml_backend_graph_compute(backend, gch) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(ch_buf); ggml_free(ctx); return nullptr; }
                        ggml_backend_tensor_get(out_c, out_host + (size_t)start * D, 0, (size_t)len * D * sizeof(float));
                        ggml_backend_buffer_free(ch_buf);
                    } else {
                        ggml_graph_compute_with_ctx(ctx, gch, 4);
                        memcpy(out_host + (size_t)start * D, out_c->data, (size_t)len * D * sizeof(float));
                    }
                    if (backend) ggml_free(ctx); else ggml_reset(ctx);
                }
                return nullptr;
            }
            struct ggml_tensor * q = proj_3d(c, Wq, bq, q_src, Nq);
            struct ggml_tensor * k = proj_3d(c, Wk, bk, kv_src, Nkv);
            struct ggml_tensor * v = proj_3d(c, Wv, bv, kv_src, Nkv);
            q = ggml_permute(c, q, 0, 2, 1, 3);
            k = ggml_permute(c, k, 0, 2, 1, 3);
            v = ggml_permute(c, v, 1, 2, 0, 3);
            k = ggml_cont(c, k);
            v = ggml_cont(c, v);
            struct ggml_tensor * kq = ggml_mul_mat(c, k, q);
            kq = ggml_scale(c, kq, 1.0f / sqrtf((float)Dh));
            kq = ggml_soft_max(c, kq);
            struct ggml_tensor * out = ggml_mul_mat(c, v, kq);
            out = ggml_permute(c, out, 0, 2, 1, 3);
            out = ggml_cont(c, out);
            out = ggml_reshape_2d(c, out, D, Nq);
            if (Wo) {
                out = ggml_mul_mat(c, w_for_mul(c, Wo, out), out);
                out = ggml_cont(c, out);
                if (bo) {
                    struct ggml_tensor * b2 = ggml_repeat(c, bo,
                        ggml_new_tensor_2d(c, GGML_TYPE_F32, (int)out->ne[0], (int)out->ne[1]));
                    out = ggml_add(c, out, b2);
                }
            }
            return out;
        };

        // ── Blocks: one context per block, same scratch buffer ─────────────────
        for (int li = 0; li < cfg.num_layers; ++li) {
            const auto & B = blocks[li];
            if (show_blocks) {
                fprintf(stderr, "\r[ltx]   block %2d/%d  ", li + 1, cfg.num_layers);
                fflush(stderr);
            }
            if (backend) { ctx = ggml_init(params); if (!ctx) { LTX_ERR("DiT: block ggml_init failed"); return {}; } }
            struct ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok);
            if (!backend) memcpy(x_in->data, x_host.data(), (size_t)D * n_tok * sizeof(float));
            struct ggml_tensor * ctx_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, seq_len);
            if (!backend) memcpy(ctx_emb_in->data, ctx_emb_host.data(), (size_t)D * seq_len * sizeof(float));

            struct ggml_tensor * h = x_in;
            struct ggml_tensor * blk_x_in = x_in;
            struct ggml_tensor * blk_sa_extra = nullptr;
            std::vector<float> sa_out_host((size_t)D * n_tok);
            std::vector<float> nx_host;
            struct ggml_tensor * nx = h;
            if (B.norm1_w) {
                nx = ggml_rms_norm(ctx, nx, cfg.norm_eps);
                nx = ggml_mul(ctx, nx, B.norm1_w);
            }
            const float * q_src_host_ptr = nullptr;
            if (n_tok > ATTN_Q_CHUNK) {
                struct ggml_cgraph * gnx = ggml_new_graph(ctx);
                ggml_build_forward_expand(gnx, nx);
                nx_host.resize((size_t)D * n_tok);
                if (backend) {
                    ggml_backend_buffer_t nx_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (!nx_buf) { ggml_free(ctx); return {}; }
                    ggml_backend_tensor_set(x_in, x_host.data(), 0, (size_t)D * n_tok * sizeof(float));
                    ggml_backend_tensor_set(ctx_emb_in, ctx_emb_host.data(), 0, (size_t)D * seq_len * sizeof(float));
                    if (ggml_backend_graph_compute(backend, gnx) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(nx_buf); ggml_free(ctx); return {}; }
                    ggml_backend_tensor_get(nx, nx_host.data(), 0, (size_t)D * n_tok * sizeof(float));
                    ggml_backend_buffer_free(nx_buf);
                } else {
                    ggml_graph_compute_with_ctx(ctx, gnx, 4);
                    memcpy(nx_host.data(), nx->data, (size_t)D * n_tok * sizeof(float));
                    ggml_reset(ctx);
                }
                q_src_host_ptr = nx_host.data();
            }
            struct ggml_tensor * sa_out = attn(ctx, nx, nx, n_tok, n_tok,
                B.attn1_q, B.attn1_k, B.attn1_v, B.attn1_o,
                B.attn1_q_b, B.attn1_k_b, B.attn1_v_b, B.attn1_o_b,
                sa_out_host.data(), q_src_host_ptr);
            if (sa_out) {
                h = ggml_add(ctx, h, ggml_cont(ctx, sa_out));
            } else {
                // ctx was already freed inside the attn Q-chunk loop (backend) or needs reset (CPU)
                if (backend) { ctx = ggml_init(params); if (!ctx) { LTX_ERR("DiT: block ctx after chunked attn failed"); return {}; } } else ggml_reset(ctx);
                struct ggml_tensor * x_in2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok);
                if (!backend) memcpy(x_in2->data, x_host.data(), (size_t)D * n_tok * sizeof(float));
                struct ggml_tensor * sa_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok);
                if (!backend) memcpy(sa_t->data, sa_out_host.data(), (size_t)D * n_tok * sizeof(float));
                h = ggml_add(ctx, x_in2, sa_t);
                blk_x_in = x_in2;
                blk_sa_extra = sa_t;
                ctx_emb_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, seq_len);
                if (!backend) memcpy(ctx_emb_in->data, ctx_emb_host.data(), (size_t)D * seq_len * sizeof(float));
            }

            struct ggml_tensor * cx = h;
            if (B.norm2_w) {
                cx = ggml_rms_norm(ctx, cx, cfg.norm_eps);
                cx = ggml_mul(ctx, cx, B.norm2_w);
            }
            struct ggml_tensor * ca_out = attn(ctx, cx, ctx_emb_in, n_tok, seq_len,
                B.attn2_q, B.attn2_k, B.attn2_v, B.attn2_o,
                B.attn2_q_b, B.attn2_k_b, B.attn2_v_b, B.attn2_o_b,
                nullptr, nullptr);
            h = ggml_add(ctx, h, ggml_cont(ctx, ca_out));

            struct ggml_tensor * fx = h;
            if (B.norm3_w) {
                fx = ggml_rms_norm(ctx, fx, cfg.norm_eps);
                fx = ggml_mul(ctx, fx, B.norm3_w);
            }
            if (B.ff_gate && B.ff_down) {
                struct ggml_tensor * gate = ggml_mul_mat(ctx, w_for_mul(ctx, B.ff_gate, fx), fx);
                gate = ggml_cont(ctx, gate);
                if (B.ff_gate_b) {
                    struct ggml_tensor * b2 = ggml_repeat(ctx, B.ff_gate_b,
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)gate->ne[0], (int)gate->ne[1]));
                    gate = ggml_add(ctx, gate, b2);
                }
                int ff_dim = (int)gate->ne[0], half_ff = ff_dim / 2;
                struct ggml_tensor * g_half  = ggml_view_2d(ctx, gate, half_ff, n_tok, gate->nb[1], 0);
                struct ggml_tensor * up_half = ggml_view_2d(ctx, gate, half_ff, n_tok, gate->nb[1], half_ff * (int)ggml_element_size(gate));
                g_half = ggml_gelu(ctx, g_half);
                struct ggml_tensor * ffn_mid = ggml_mul(ctx, g_half, up_half);
                struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, w_for_mul(ctx, B.ff_down, ffn_mid), ffn_mid);
                ffn_out = ggml_cont(ctx, ffn_out);
                if (B.ff_down_b) {
                    struct ggml_tensor * b2 = ggml_repeat(ctx, B.ff_down_b,
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)ffn_out->ne[0], (int)ffn_out->ne[1]));
                    ffn_out = ggml_add(ctx, ffn_out, b2);
                }
                h = ggml_add(ctx, h, ggml_cont(ctx, ffn_out));
            }
            struct ggml_cgraph * gf_blk = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf_blk, h);
            if (backend) {
                ggml_backend_buffer_t blk_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (!blk_buf) { ggml_free(ctx); return {}; }
                ggml_backend_tensor_set(blk_x_in, x_host.data(), 0, (size_t)D * n_tok * sizeof(float));
                ggml_backend_tensor_set(ctx_emb_in, ctx_emb_host.data(), 0, (size_t)D * seq_len * sizeof(float));
                if (blk_sa_extra) ggml_backend_tensor_set(blk_sa_extra, sa_out_host.data(), 0, (size_t)D * n_tok * sizeof(float));
                if (ggml_backend_graph_compute(backend, gf_blk) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(blk_buf); ggml_free(ctx); return {}; }
                ggml_backend_tensor_get(h, x_host.data(), 0, (size_t)D * n_tok * sizeof(float));
                ggml_backend_buffer_free(blk_buf);
            } else {
                ggml_graph_compute_with_ctx(ctx, gf_blk, 4);
                memcpy(x_host.data(), h->data, (size_t)D * n_tok * sizeof(float));
            }
            if (backend) ggml_free(ctx); else ggml_reset(ctx);
        }

        // ── Post: final norm + proj_out ───────────────────────────────────────
        if (backend) { ctx = ggml_init(params); if (!ctx) { LTX_ERR("DiT: post ggml_init failed"); return {}; } }
        struct ggml_tensor * x_fin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok);
        struct ggml_tensor * x_fin = x_fin_in;
        if (!backend) memcpy(x_fin->data, x_host.data(), (size_t)D * n_tok * sizeof(float));
        if (final_norm_w) {
            x_fin = ggml_rms_norm(ctx, x_fin, cfg.norm_eps);
            x_fin = ggml_mul(ctx, x_fin, final_norm_w);
        }
        if (proj_out_w) {
            x_fin = ggml_mul_mat(ctx, w_for_mul(ctx, proj_out_w, x_fin), x_fin);
            x_fin = ggml_cont(ctx, x_fin);
            if (proj_out_b) {
                struct ggml_tensor * b2 = ggml_repeat(ctx, proj_out_b,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)x_fin->ne[0], (int)x_fin->ne[1]));
                x_fin = ggml_add(ctx, x_fin, b2);
            }
        }
        struct ggml_cgraph * gf_post = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf_post, x_fin);
        std::vector<float> out((size_t)n_tok * Pd);
        if (backend) {
            ggml_backend_buffer_t post_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!post_buf) { ggml_free(ctx); return {}; }
            ggml_backend_tensor_set(x_fin_in, x_host.data(), 0, (size_t)D * n_tok * sizeof(float));
            if (ggml_backend_graph_compute(backend, gf_post) != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(post_buf); ggml_free(ctx); return {}; }
            ggml_backend_tensor_get(x_fin, out.data(), 0, (size_t)n_tok * Pd * sizeof(float));
            ggml_backend_buffer_free(post_buf);
        } else {
            ggml_graph_compute_with_ctx(ctx, gf_post, 4);
            memcpy(out.data(), x_fin->data, (size_t)n_tok * Pd * sizeof(float));
        }
        ggml_free(ctx);
        ctx = nullptr;
        return out;
    }
};

// ── Patchify / Unpatchify ─────────────────────────────────────────────────────

// Patchify video latent [T_lat, H_lat, W_lat, C] → [N_tok, patch_dim]
// patch_size = (pt, ph, pw)
static std::vector<float> patchify(
        const float * lat,
        int T, int H, int W, int C,
        int pt, int ph, int pw)
{
    int Tp = T / pt, Hp = H / ph, Wp = W / pw;
    int N = Tp * Hp * Wp;
    int Pd = pt * ph * pw * C;
    std::vector<float> out(N * Pd);

    for (int tp = 0; tp < Tp; ++tp)
    for (int hp = 0; hp < Hp; ++hp)
    for (int wp = 0; wp < Wp; ++wp) {
        int tok = tp * Hp * Wp + hp * Wp + wp;
        float * dst = out.data() + tok * Pd;
        int d = 0;
        for (int dt = 0; dt < pt; ++dt)
        for (int dh = 0; dh < ph; ++dh)
        for (int dw = 0; dw < pw; ++dw) {
            int t = tp * pt + dt;
            int h = hp * ph + dh;
            int w = wp * pw + dw;
            const float * src = lat + ((t * H + h) * W + w) * C;
            for (int c = 0; c < C; ++c)
                dst[d++] = src[c];
        }
    }
    return out;
}

// Unpatchify [N_tok, patch_dim] → [T_lat, H_lat, W_lat, C]
static std::vector<float> unpatchify(
        const float * tok,
        int T, int H, int W, int C,
        int pt, int ph, int pw)
{
    int Tp = T / pt, Hp = H / ph, Wp = W / pw;
    int Pd = pt * ph * pw * C;
    std::vector<float> out(T * H * W * C, 0.0f);

    for (int tp = 0; tp < Tp; ++tp)
    for (int hp = 0; hp < Hp; ++hp)
    for (int wp = 0; wp < Wp; ++wp) {
        int tidx = tp * Hp * Wp + hp * Wp + wp;
        const float * src = tok + tidx * Pd;
        int d = 0;
        for (int dt = 0; dt < pt; ++dt)
        for (int dh = 0; dh < ph; ++dh)
        for (int dw = 0; dw < pw; ++dw) {
            int t = tp * pt + dt;
            int h = hp * ph + dh;
            int w = wp * pw + dw;
            float * dst = out.data() + ((t * H + h) * W + w) * C;
            for (int c = 0; c < C; ++c)
                dst[c] = src[d++];
        }
    }
    return out;
}
