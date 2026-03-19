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
#include "ltx_lora.hpp"
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
    // Self-attention weights (LTX 2.3 has no explicit pre-norm; uses AdaLN scale/shift only)
    struct ggml_tensor * attn1_q    = nullptr, * attn1_q_b   = nullptr;
    struct ggml_tensor * attn1_k    = nullptr, * attn1_k_b   = nullptr;
    struct ggml_tensor * attn1_v    = nullptr, * attn1_v_b   = nullptr;
    struct ggml_tensor * attn1_o    = nullptr, * attn1_o_b   = nullptr;
    struct ggml_tensor * attn1_qnorm = nullptr; // per-head RMS norm on Q
    struct ggml_tensor * attn1_knorm = nullptr; // per-head RMS norm on K

    // Per-block AdaLN learned offsets: scale_shift_table [D, 9]
    // (9 = 3 signals × 3 sublayers: SA, CA, FFN)
    struct ggml_tensor * adaln_w = nullptr;

    // Cross-attention
    struct ggml_tensor * attn2_q    = nullptr, * attn2_q_b   = nullptr;
    struct ggml_tensor * attn2_k    = nullptr, * attn2_k_b   = nullptr;
    struct ggml_tensor * attn2_v    = nullptr, * attn2_v_b   = nullptr;
    struct ggml_tensor * attn2_o    = nullptr, * attn2_o_b   = nullptr;
    struct ggml_tensor * attn2_qnorm = nullptr;
    struct ggml_tensor * attn2_knorm = nullptr;

    // FFN (GEGLU: fc1_gate packs gate+up, fc2 = down)
    struct ggml_tensor * ff_gate = nullptr, * ff_gate_b = nullptr;
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

        // Transformer blocks.
        // LTX 2.3 GGUF uses "transformer_blocks." prefix (no "model.diffusion_model." prefix).
        const char * block_prefix = get("transformer_blocks.0.attn1.to_q.weight")
            ? "transformer_blocks." : "model.diffusion_model.transformer_blocks.";
        if (std::strstr(block_prefix, "transformer_blocks.") && !std::strstr(block_prefix, "model.")) {
            int n = 0;
            char lbuf[128];
            for (; n < 256; ++n) {
                (void)snprintf(lbuf, sizeof(lbuf), "transformer_blocks.%d.attn1.to_q.weight", n);
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

            // Per-block AdaLN: scale_shift_table [D, 9] (9 = 3 sublayers × 3 signals)
            GET(adaln_w,    "%d.scale_shift_table");
            // Self-attention
            GET(attn1_q,    "%d.attn1.to_q.weight");
            GET(attn1_q_b,  "%d.attn1.to_q.bias");
            GET(attn1_k,    "%d.attn1.to_k.weight");
            GET(attn1_k_b,  "%d.attn1.to_k.bias");
            GET(attn1_v,    "%d.attn1.to_v.weight");
            GET(attn1_v_b,  "%d.attn1.to_v.bias");
            GET(attn1_o,    "%d.attn1.to_out.0.weight");
            GET(attn1_o_b,  "%d.attn1.to_out.0.bias");
            GET(attn1_qnorm,"%d.attn1.q_norm.weight");
            GET(attn1_knorm,"%d.attn1.k_norm.weight");
            // Cross-attention
            GET(attn2_q,    "%d.attn2.to_q.weight");
            GET(attn2_q_b,  "%d.attn2.to_q.bias");
            GET(attn2_k,    "%d.attn2.to_k.weight");
            GET(attn2_k_b,  "%d.attn2.to_k.bias");
            GET(attn2_v,    "%d.attn2.to_v.weight");
            GET(attn2_v_b,  "%d.attn2.to_v.bias");
            GET(attn2_o,    "%d.attn2.to_out.0.weight");
            GET(attn2_o_b,  "%d.attn2.to_out.0.bias");
            GET(attn2_qnorm,"%d.attn2.q_norm.weight");
            GET(attn2_knorm,"%d.attn2.k_norm.weight");
            // FFN (GEGLU): ff.net.0.proj packs gate+up [2*ff_dim, D]; ff.net.2 = down
            GET(ff_gate,    "%d.ff.net.0.proj.weight");
            GET(ff_gate_b,  "%d.ff.net.0.proj.bias");
            GET(ff_down,    "%d.ff.net.2.weight");
            GET(ff_down_b,  "%d.ff.net.2.bias");
#undef GET
        }

        LTX_LOG("DiT loaded: layers=%d hidden=%d heads=%d head_dim=%d",
                cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.head_dim);
        return true;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────
    //
    // Builds the entire DiT forward graph in one shot and dispatches via
    // ggml_backend_sched (Metal + CPU auto-scheduled).
    //
    // Inputs:
    //   latents:    [n_tok, patch_dim]  patchified video latent
    //   text_emb:   [seq_len, cross_dim] T5 encoder output
    //   timestep:   scalar in [0,1]     noise level
    //   frame_rate: fps (e.g. 24); embedded and added to timestep emb (LTX 2.3)
    //   sched:      backend scheduler (Metal+CPU or CPU-only)
    //
    // Returns predicted velocity: [n_tok × patch_dim]
    std::vector<float> forward(
            const float * latents,  int n_tok,
            const float * text_emb, int seq_len,
            float         timestep,
            float         frame_rate,
            ggml_backend_sched_t sched,
            const LtxLoRA * lora = nullptr) const
    {
        int D   = cfg.hidden_size;
        int Pd  = cfg.patch_dim();
        int Cd  = cfg.cross_attn_dim;
        int H   = cfg.num_heads;
        int Dh  = cfg.head_dim;

        // Allocate graph metadata context (no_alloc — sched owns actual buffers).
        // Per-block: ~120 nodes (SA + CA + FFN + AdaLN + q/k norms); 28 blocks → ~3360 nodes
        // With LoRA: ~6 extra nodes per adapted layer (A, B tensors + 4 ops), ~10 layers/block.
        const size_t LORA_EXTRA = lora ? (size_t)cfg.num_layers * 80 : 0;
        const size_t MAX_NODES = (size_t)cfg.num_layers * 128 + 512 + LORA_EXTRA;
        size_t ctx_size = ggml_tensor_overhead() * MAX_NODES * 2
                        + ggml_graph_overhead_custom(MAX_NODES, false);
        std::vector<uint8_t> ctx_buf(ctx_size);
        struct ggml_init_params ip = { ctx_size, ctx_buf.data(), /*no_alloc=*/true };
        struct ggml_context * ctx = ggml_init(ip);
        if (!ctx) { LTX_ERR("DiT: ggml_init failed"); return {}; }

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

        // ── Inputs ────────────────────────────────────────────────────────────
        struct ggml_tensor * x_in   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Pd, n_tok);
        struct ggml_tensor * ctx_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cd, seq_len);
        struct ggml_tensor * t_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        struct ggml_tensor * fps_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(x_in,   "latents");   ggml_set_input(x_in);
        ggml_set_name(ctx_in, "text_emb");  ggml_set_input(ctx_in);
        ggml_set_name(t_in,   "timestep");  ggml_set_input(t_in);
        ggml_set_name(fps_in, "frame_rate"); ggml_set_input(fps_in);

        // ── Timestep embedding → tproj [9*D] ──────────────────────────────────
        // t  → scale(×1000) → sinusoidal(freq_dim) → \
        //                                              add → linear_1+silu → linear_2+silu → adaln.linear
        // fps → sinusoidal(freq_dim)               → /
        struct ggml_tensor * tproj = nullptr;
        if (adaln.emb_w1 && adaln.emb_w2 && adaln.linear_w) {
            struct ggml_tensor * t_s   = ggml_scale(ctx, t_in, 1000.0f);
            struct ggml_tensor * t_emb = ggml_timestep_embedding(ctx, t_s, cfg.freq_dim, 10000);
            // Frame-rate embedding: same sinusoidal, added to timestep emb (LTX 2.3).
            struct ggml_tensor * fps_emb = ggml_timestep_embedding(ctx, fps_in, cfg.freq_dim, 10000);
            t_emb = ggml_add(ctx, t_emb, fps_emb);
            struct ggml_tensor * h     = ggml_mul_mat(ctx, adaln.emb_w1, t_emb);
            if (adaln.emb_b1) h = ggml_add(ctx, h, adaln.emb_b1);
            h = ggml_silu(ctx, h);
            h = ggml_mul_mat(ctx, adaln.emb_w2, h);
            if (adaln.emb_b2) h = ggml_add(ctx, h, adaln.emb_b2);
            h = ggml_silu(ctx, h);
            tproj = ggml_mul_mat(ctx, adaln.linear_w, h);
            if (adaln.linear_b) tproj = ggml_add(ctx, tproj, adaln.linear_b);
        }

        // ── Patch embed + caption projection ──────────────────────────────────
        struct ggml_tensor * x = x_in;
        if (patch_embed_w) {
            x = ggml_mul_mat(ctx, patch_embed_w, x);
            if (patch_embed_b) x = ggml_add(ctx, x, patch_embed_b);
        }
        struct ggml_tensor * enc = ctx_in;
        if (cap_proj_w) {
            enc = ggml_mul_mat(ctx, cap_proj_w, enc);
            if (cap_proj_b) enc = ggml_add(ctx, enc, cap_proj_b);
        }

        // Helper: linear proj → reshape [Dh, H, S] → permute → [Dh, S, H] (flash_attn layout)
        const float attn_scale = 1.0f / sqrtf((float)Dh);

        // Apply LoRA delta to an already-computed output: out += scale * B @ (A @ src).
        // Uses pre-migrated GPU tensors (lp->gpu_A / gpu_B) — zero per-step data copy.
        // src shape: [in_feat, S];  out shape: [out_feat, S].
        auto lora_add = [&](struct ggml_tensor * out, struct ggml_tensor * src,
                            const std::string & lora_key) -> struct ggml_tensor * {
            if (!lora || lora_key.empty()) return out;
            const LoRAPair * lp = lora->find(lora_key);
            if (!lp || !lp->gpu_A || !lp->gpu_B) return out;
            // gpu_A/gpu_B are already on the backend (Metal/CPU); reference them directly.
            struct ggml_tensor * ax    = ggml_mul_mat(ctx, lp->gpu_A, src);  // [rank, S]
            struct ggml_tensor * delta = ggml_mul_mat(ctx, lp->gpu_B, ax);   // [out_feat, S]
            delta = ggml_scale(ctx, delta, lora->scale);
            return ggml_add(ctx, out, delta);
        };

        // Project src → Q/K/V, optionally apply RMS norm (q/k_norm weight [D]) before head-split.
        // norm_w shape: [D] — applied per-token over the full hidden dim before reshape into heads.
        // lora_key: canonical LoRA key for this projection (empty = no LoRA).
        auto qkv_proj = [&](struct ggml_tensor * W, struct ggml_tensor * b,
                            struct ggml_tensor * norm_w,
                            struct ggml_tensor * src, int S,
                            const std::string & lora_key = "") -> struct ggml_tensor * {
            struct ggml_tensor * o = ggml_mul_mat(ctx, W, src); // [D, S]
            if (b) o = ggml_add(ctx, o, b);
            o = lora_add(o, src, lora_key);
            if (norm_w) {
                o = ggml_rms_norm(ctx, o, cfg.norm_eps); // normalize per token over D
                o = ggml_mul(ctx, o, norm_w);            // scale by learned weight [D]
            }
            o = ggml_reshape_3d(ctx, o, Dh, H, S);
            o = ggml_permute(ctx, o, 0, 2, 1, 3); // → [Dh, S, H, 1]
            return ggml_cont(ctx, o);
        };

        // ── Transformer blocks ─────────────────────────────────────────────────
        // LTX 2.3 architecture (no pre-norm layers; only AdaLN + q/k norms):
        //   scale_shift_table [D, 9]:  9 = 3 sublayers × (shift, scale, gate)
        //     [0..2]: SA  (shift_sa,  scale_sa,  gate_sa)
        //     [3..5]: CA  (shift_ca,  scale_ca,  gate_ca)
        //     [6..8]: FFN (shift_ffn, scale_ffn, gate_ffn)
        for (int li = 0; li < cfg.num_layers; ++li) {
            const auto & B = blocks[li];
            // LoRA canonical key prefix for this block.
            std::string lk = lora ? ("transformer_blocks." + std::to_string(li) + ".") : std::string{};

            // AdaLN: scale_shift_table [D, 9] + shared tproj [9*D] → 9 modulation vectors
            struct ggml_tensor * shift_sa  = nullptr, * scale_sa  = nullptr, * gate_sa  = nullptr;
            struct ggml_tensor * shift_ca  = nullptr, * scale_ca  = nullptr, * gate_ca  = nullptr;
            struct ggml_tensor * shift_ffn = nullptr, * scale_ffn = nullptr, * gate_ffn = nullptr;
            if (tproj && B.adaln_w) {
                struct ggml_tensor * ss = B.adaln_w;
                if (ss->type != GGML_TYPE_F32) ss = ggml_cast(ctx, ss, GGML_TYPE_F32);
                struct ggml_tensor * m = ggml_add(ctx, ggml_reshape_1d(ctx, ss, 9 * D), tproj);
                size_t Db = (size_t)D * sizeof(float);
                shift_sa  = ggml_view_1d(ctx, m, D, 0 * Db);
                scale_sa  = ggml_view_1d(ctx, m, D, 1 * Db);
                gate_sa   = ggml_view_1d(ctx, m, D, 2 * Db);
                shift_ca  = ggml_view_1d(ctx, m, D, 3 * Db);
                scale_ca  = ggml_view_1d(ctx, m, D, 4 * Db);
                gate_ca   = ggml_view_1d(ctx, m, D, 5 * Db);
                shift_ffn = ggml_view_1d(ctx, m, D, 6 * Db);
                scale_ffn = ggml_view_1d(ctx, m, D, 7 * Db);
                gate_ffn  = ggml_view_1d(ctx, m, D, 8 * Db);
            }

            // ── Self-attention ──────────────────────────────────────────────────
            // AdaLN modulation: x_mod = (1 + scale) * x + shift = x + x*scale + shift
            struct ggml_tensor * nx = x;
            if (scale_sa) {
                nx = ggml_add(ctx, ggml_add(ctx, nx, ggml_mul(ctx, nx, scale_sa)), shift_sa);
            }
            if (B.attn1_q) {
                struct ggml_tensor * q = qkv_proj(B.attn1_q, B.attn1_q_b, B.attn1_qnorm, nx, n_tok, lk + "attn1.to_q");
                struct ggml_tensor * k = qkv_proj(B.attn1_k, B.attn1_k_b, B.attn1_knorm, nx, n_tok, lk + "attn1.to_k");
                struct ggml_tensor * v = qkv_proj(B.attn1_v, B.attn1_v_b, nullptr,        nx, n_tok, lk + "attn1.to_v");
                struct ggml_tensor * sa_in = nx; // keep input for LoRA output proj
                struct ggml_tensor * sa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, attn_scale, 0.0f, 0.0f);
                sa = ggml_cont(ctx, ggml_reshape_2d(ctx, sa, D, n_tok));
                if (B.attn1_o) {
                    struct ggml_tensor * sa_pre_o = sa;
                    sa = ggml_mul_mat(ctx, B.attn1_o, sa);
                    if (B.attn1_o_b) sa = ggml_add(ctx, sa, B.attn1_o_b);
                    sa = lora_add(sa, sa_pre_o, lk + "attn1.to_out.0");
                }
                (void)sa_in;
                if (gate_sa) sa = ggml_mul(ctx, sa, gate_sa);
                x = ggml_add(ctx, x, sa);
            }

            // ── Cross-attention ─────────────────────────────────────────────────
            struct ggml_tensor * cx = x;
            if (scale_ca) {
                cx = ggml_add(ctx, ggml_add(ctx, cx, ggml_mul(ctx, cx, scale_ca)), shift_ca);
            }
            if (B.attn2_q) {
                struct ggml_tensor * cq = qkv_proj(B.attn2_q, B.attn2_q_b, B.attn2_qnorm, cx,  n_tok,   lk + "attn2.to_q");
                struct ggml_tensor * ck = qkv_proj(B.attn2_k, B.attn2_k_b, B.attn2_knorm, enc, seq_len, lk + "attn2.to_k");
                struct ggml_tensor * cv = qkv_proj(B.attn2_v, B.attn2_v_b, nullptr,        enc, seq_len, lk + "attn2.to_v");
                struct ggml_tensor * ca = ggml_flash_attn_ext(ctx, cq, ck, cv, nullptr, attn_scale, 0.0f, 0.0f);
                ca = ggml_cont(ctx, ggml_reshape_2d(ctx, ca, D, n_tok));
                if (B.attn2_o) {
                    struct ggml_tensor * ca_pre_o = ca;
                    ca = ggml_mul_mat(ctx, B.attn2_o, ca);
                    if (B.attn2_o_b) ca = ggml_add(ctx, ca, B.attn2_o_b);
                    ca = lora_add(ca, ca_pre_o, lk + "attn2.to_out.0");
                }
                if (gate_ca) ca = ggml_mul(ctx, ca, gate_ca);
                x = ggml_add(ctx, x, ca);
            }

            // ── FFN (GEGLU) ──────────────────────────────────────────────────────
            if (B.ff_gate && B.ff_down) {
                struct ggml_tensor * fx = x;
                if (scale_ffn) {
                    fx = ggml_add(ctx, ggml_add(ctx, fx, ggml_mul(ctx, fx, scale_ffn)), shift_ffn);
                }
                struct ggml_tensor * gu = ggml_mul_mat(ctx, B.ff_gate, fx);
                if (B.ff_gate_b) gu = ggml_add(ctx, gu, B.ff_gate_b);
                gu = lora_add(gu, fx, lk + "ff.net.0.proj");
                // LTX 2.3 uses GELU activation (not split GEGLU): ff_dim = ne[1] of ff_gate
                struct ggml_tensor * ffn_mid = ggml_gelu(ctx, gu);
                struct ggml_tensor * ffn_out = ggml_mul_mat(ctx, B.ff_down, ffn_mid);
                if (B.ff_down_b) ffn_out = ggml_add(ctx, ffn_out, B.ff_down_b);
                ffn_out = lora_add(ffn_out, ffn_mid, lk + "ff.net.2");
                if (gate_ffn) ffn_out = ggml_mul(ctx, ffn_out, gate_ffn);
                x = ggml_add(ctx, x, ffn_out);
            }
        }

        // ── Final norm + proj_out ─────────────────────────────────────────────
        if (final_norm_w) { x = ggml_rms_norm(ctx, x, cfg.norm_eps); x = ggml_mul(ctx, x, final_norm_w); }
        if (proj_out_w) {
            x = ggml_mul_mat(ctx, proj_out_w, x);
            if (proj_out_b) x = ggml_add(ctx, x, proj_out_b);
        }
        ggml_set_name(x, "velocity");
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);

        // ── Schedule → allocate → set inputs → compute → retrieve ────────────
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LTX_ERR("DiT: sched alloc failed"); ggml_free(ctx); return {};
        }
        ggml_backend_tensor_set(x_in,   latents,    0, (size_t)n_tok   * Pd * sizeof(float));
        ggml_backend_tensor_set(ctx_in, text_emb,   0, (size_t)seq_len * Cd * sizeof(float));
        ggml_backend_tensor_set(t_in,   &timestep,  0, sizeof(float));
        ggml_backend_tensor_set(fps_in, &frame_rate, 0, sizeof(float));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            LTX_ERR("DiT: compute failed"); ggml_free(ctx); return {};
        }

        std::vector<float> out((size_t)n_tok * Pd);
        ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float));
        ggml_free(ctx);
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

// ── Audio patchify / unpatchify (for AV pipeline) ─────────────────────────────
// Audio latent layout: [T, C, F] with C=8, F=16 (Lightricks AudioLatentShape).
// Patchify: (T, C, F) → (T, C*F) so n_audio_tok = T, Pd_audio = 128.
// Matches Python: "b c t f -> b t (c f)".

static std::vector<float> patchify_audio(
        const float * lat,
        int T, int C, int F)
{
    int Pd = C * F;
    std::vector<float> out((size_t)T * Pd);
    for (int t = 0; t < T; ++t) {
        float * dst = out.data() + (size_t)t * Pd;
        int d = 0;
        for (int c = 0; c < C; ++c)
            for (int f = 0; f < F; ++f)
                dst[d++] = lat[((size_t)t * C + c) * F + f];
    }
    return out;
}

static std::vector<float> unpatchify_audio(
        const float * tok,
        int T, int C, int F)
{
    int Pd = C * F;
    std::vector<float> out((size_t)T * C * F, 0.0f);
    for (int t = 0; t < T; ++t) {
        const float * src = tok + (size_t)t * Pd;
        int d = 0;
        for (int c = 0; c < C; ++c)
            for (int f = 0; f < F; ++f)
                out[((size_t)t * C + c) * F + f] = src[d++];
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
