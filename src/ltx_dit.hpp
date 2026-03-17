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
    bool load(LtxGgufModel & model) {
        // Read config from GGUF metadata.
        uint32_t hs = model.kv_u32("ltxv.hidden_size", 0);
        if (hs > 0) cfg.hidden_size = (int)hs;
        uint32_t nl = model.kv_u32("ltxv.num_hidden_layers", 0);
        if (nl > 0) cfg.num_layers = (int)nl;
        uint32_t nh = model.kv_u32("ltxv.num_attention_heads", 0);
        if (nh > 0) cfg.num_heads = (int)nh;
        uint32_t lc = model.kv_u32("ltxv.in_channels", 0);
        if (lc > 0) cfg.latent_channels = (int)lc;
        cfg.head_dim = cfg.hidden_size / cfg.num_heads;

        auto get = [&](const char * nm) { return model.get_tensor(nm); };

        // Patchify projection.
        patch_embed_w = get("model.diffusion_model.patchify_proj.weight");
        patch_embed_b = get("model.diffusion_model.patchify_proj.bias");
        if (!patch_embed_w) {
            patch_embed_w = get("dit.patchify_proj.weight");
            patch_embed_b = get("dit.patchify_proj.bias");
        }

        // Caption projection.
        cap_proj_w = get("model.diffusion_model.caption_projection.weight");
        cap_proj_b = get("model.diffusion_model.caption_projection.bias");
        if (!cap_proj_w) {
            cap_proj_w = get("dit.caption_projection.weight");
            cap_proj_b = get("dit.caption_projection.bias");
        }

        // AdaLN-single timestep embedder.
        adaln.emb_w1 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight");
        adaln.emb_b1 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.bias");
        adaln.emb_w2 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.weight");
        adaln.emb_b2 = get("model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.bias");
        adaln.linear_w = get("model.diffusion_model.adaln_single.linear.weight");
        adaln.linear_b = get("model.diffusion_model.adaln_single.linear.bias");

        // Output projection.
        proj_out_w   = get("model.diffusion_model.proj_out.weight");
        proj_out_b   = get("model.diffusion_model.proj_out.bias");
        final_norm_w = get("model.diffusion_model.norm_out.linear.weight");
        final_norm_b = get("model.diffusion_model.norm_out.linear.bias");

        // Transformer blocks.
        blocks.resize(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) {
            auto & B = blocks[i];
            char buf[384];
            std::string pre = "model.diffusion_model.transformer_blocks.";
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
    //   latents:    [N_tok, patch_dim]  (patchified video latent)
    //   text_emb:   [S, cross_dim]      (T5 encoder output)
    //   timestep:   scalar in [0,1]     (noise level)
    //   n_tok:      number of latent patches
    //   seq_len:    text sequence length
    //
    // Returns predicted noise/velocity: [N_tok × patch_dim]
    std::vector<float> forward(
            const float * latents,   int n_tok,
            const float * text_emb,  int seq_len,
            float timestep) const
    {
        int D  = cfg.hidden_size;
        int Pd = cfg.patch_dim();
        int Cd = cfg.cross_attn_dim;

        // Allocate scratch ggml context.
        size_t mem = (size_t)1024 * 1024 * 1024; // 1 GB scratch
        struct ggml_init_params p{mem, nullptr, false};
        struct ggml_context * ctx = ggml_init(p);
        if (!ctx) LTX_ABORT("DiT: ggml_init failed");

        // ── Timestep embedding ────────────────────────────────────────────────
        auto ts_sincos = sinusoidal_embedding(timestep, cfg.freq_dim);
        struct ggml_tensor * ts_emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.freq_dim);
        memcpy(ts_emb->data, ts_sincos.data(), cfg.freq_dim * sizeof(float));

        // Linear MLP for timestep: freq_dim → hidden_size → hidden_size.
        struct ggml_tensor * t_hid = ts_emb;
        if (adaln.emb_w1) {
            t_hid = ggml_mul_mat(ctx, adaln.emb_w1, t_hid);
            if (adaln.emb_b1) t_hid = ggml_add(ctx, t_hid, adaln.emb_b1);
            t_hid = ggml_silu(ctx, t_hid);
        }
        if (adaln.emb_w2) {
            t_hid = ggml_mul_mat(ctx, adaln.emb_w2, t_hid);
            if (adaln.emb_b2) t_hid = ggml_add(ctx, t_hid, adaln.emb_b2);
        }
        // t_hid: [hidden_size]

        // AdaLN-single: project timestep embedding to 6*hidden_size.
        struct ggml_tensor * ada_params = t_hid;
        if (adaln.linear_w) {
            ada_params = ggml_mul_mat(ctx, adaln.linear_w, t_hid);
            if (adaln.linear_b) ada_params = ggml_add(ctx, ada_params, adaln.linear_b);
        }
        // ada_params: [6*hidden_size] → chunk into shift/scale for attn and ffn.

        // ── Patchify: project latent patches to hidden_size ───────────────────
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Pd, n_tok);
        memcpy(x->data, latents, n_tok * Pd * sizeof(float));
        if (patch_embed_w) {
            x = ggml_mul_mat(ctx, patch_embed_w, x); // [D, n_tok]
            if (patch_embed_b) {
                struct ggml_tensor * b2d = ggml_repeat(ctx, patch_embed_b,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok));
                x = ggml_add(ctx, x, b2d);
            }
        }

        // ── Text conditioning: project T5 embeddings → hidden_size ───────────
        struct ggml_tensor * ctx_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Cd, seq_len);
        memcpy(ctx_emb->data, text_emb, seq_len * Cd * sizeof(float));
        if (cap_proj_w) {
            ctx_emb = ggml_mul_mat(ctx, cap_proj_w, ctx_emb); // [D, seq_len]
            if (cap_proj_b) {
                struct ggml_tensor * b2d = ggml_repeat(ctx, cap_proj_b,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, seq_len));
                ctx_emb = ggml_add(ctx, ctx_emb, b2d);
            }
        }

        // ── Transformer blocks ────────────────────────────────────────────────
        int H  = cfg.num_heads;
        int Dh = cfg.head_dim;

        for (int li = 0; li < cfg.num_layers; ++li) {
            const auto & B = blocks[li];

            // Helper lambda: self-attention or cross-attention.
            auto attn = [&](struct ggml_tensor * q_src, struct ggml_tensor * kv_src,
                            int Nq, int Nkv,
                            struct ggml_tensor * Wq, struct ggml_tensor * Wk,
                            struct ggml_tensor * Wv, struct ggml_tensor * Wo,
                            struct ggml_tensor * bq, struct ggml_tensor * bk,
                            struct ggml_tensor * bv, struct ggml_tensor * bo)
                -> struct ggml_tensor *
            {
                auto proj = [&](struct ggml_tensor * W, struct ggml_tensor * b,
                                struct ggml_tensor * src, int N)
                    -> struct ggml_tensor * {
                    if (!W) return ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dh, H, N);
                    struct ggml_tensor * out = ggml_mul_mat(ctx, W, src); // [D, N]
                    if (b) {
                        struct ggml_tensor * b3d = ggml_repeat(ctx, b,
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N));
                        out = ggml_add(ctx, out, b3d);
                    }
                    out = ggml_reshape_3d(ctx, out, Dh, H, N);
                    return out;
                };

                struct ggml_tensor * q = proj(Wq, bq, q_src, Nq);   // [Dh, H, Nq]
                struct ggml_tensor * k = proj(Wk, bk, kv_src, Nkv); // [Dh, H, Nkv]
                struct ggml_tensor * v = proj(Wv, bv, kv_src, Nkv);

                q = ggml_permute(ctx, q, 0, 2, 1, 3); // [Dh, Nq,  H]
                k = ggml_permute(ctx, k, 0, 2, 1, 3); // [Dh, Nkv, H]
                v = ggml_permute(ctx, v, 1, 2, 0, 3); // [Nkv, Dh, H]

                struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);   // [Nkv, Nq, H]
                kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float)Dh));
                kq = ggml_soft_max(ctx, kq);
                struct ggml_tensor * out = ggml_mul_mat(ctx, v, kq); // [Dh, Nq, H]
                out = ggml_permute(ctx, out, 0, 2, 1, 3);            // [Dh, H, Nq]
                out = ggml_cont(ctx, out);
                out = ggml_reshape_2d(ctx, out, D, Nq);              // [D, Nq]
                if (Wo) {
                    out = ggml_mul_mat(ctx, Wo, out);
                    if (bo) {
                        struct ggml_tensor * b2d = ggml_repeat(ctx, bo,
                            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Nq));
                        out = ggml_add(ctx, out, b2d);
                    }
                }
                return out;
            };

            // Pre-norm for self-attention.
            struct ggml_tensor * nx = x;
            if (B.norm1_w) {
                nx = ggml_rms_norm(ctx, nx, cfg.norm_eps);
                struct ggml_tensor * scale = ggml_repeat(ctx, B.norm1_w,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok));
                nx = ggml_mul(ctx, nx, scale);
            }

            // Self-attention.
            struct ggml_tensor * sa_out = attn(
                nx, nx, n_tok, n_tok,
                B.attn1_q, B.attn1_k, B.attn1_v, B.attn1_o,
                B.attn1_q_b, B.attn1_k_b, B.attn1_v_b, B.attn1_o_b);
            x = ggml_add(ctx, x, sa_out);

            // Cross-attention.
            struct ggml_tensor * cx = x;
            if (B.norm2_w) {
                cx = ggml_rms_norm(ctx, cx, cfg.norm_eps);
                struct ggml_tensor * scale = ggml_repeat(ctx, B.norm2_w,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok));
                cx = ggml_mul(ctx, cx, scale);
            }
            struct ggml_tensor * ca_out = attn(
                cx, ctx_emb, n_tok, seq_len,
                B.attn2_q, B.attn2_k, B.attn2_v, B.attn2_o,
                B.attn2_q_b, B.attn2_k_b, B.attn2_v_b, B.attn2_o_b);
            x = ggml_add(ctx, x, ca_out);

            // FFN (SwiGLU).
            struct ggml_tensor * fx = x;
            if (B.norm3_w) {
                fx = ggml_rms_norm(ctx, fx, cfg.norm_eps);
                struct ggml_tensor * scale = ggml_repeat(ctx, B.norm3_w,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok));
                fx = ggml_mul(ctx, fx, scale);
            }
            if (B.ff_gate && B.ff_down) {
                struct ggml_tensor * gate = ggml_mul_mat(ctx, B.ff_gate, fx);
                if (B.ff_gate_b) gate = ggml_add(ctx, gate,
                    ggml_repeat(ctx, B.ff_gate_b,
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                            B.ff_gate->ne[1], n_tok)));

                // SwiGLU splits gate in half: first half is gate, second is up.
                int ff_dim = (int)gate->ne[0];
                int half_ff = ff_dim / 2;
                struct ggml_tensor * g_half  = ggml_view_2d(ctx, gate,
                    half_ff, n_tok, gate->nb[1], 0);
                struct ggml_tensor * up_half = ggml_view_2d(ctx, gate,
                    half_ff, n_tok, gate->nb[1],
                    half_ff * ggml_element_size(gate));
                g_half = ggml_gelu(ctx, g_half);
                struct ggml_tensor * ffn_out = ggml_mul(ctx, g_half, up_half);
                ffn_out = ggml_mul_mat(ctx, B.ff_down, ffn_out);
                if (B.ff_down_b) ffn_out = ggml_add(ctx, ffn_out,
                    ggml_repeat(ctx, B.ff_down_b,
                        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok)));
                x = ggml_add(ctx, x, ffn_out);
            }
        }

        // Final norm + output projection.
        if (final_norm_w) {
            x = ggml_rms_norm(ctx, x, cfg.norm_eps);
            struct ggml_tensor * scale = ggml_repeat(ctx, final_norm_w,
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_tok));
            x = ggml_mul(ctx, x, scale);
        }
        if (proj_out_w) {
            x = ggml_mul_mat(ctx, proj_out_w, x);
            if (proj_out_b) x = ggml_add(ctx, x,
                ggml_repeat(ctx, proj_out_b,
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Pd, n_tok)));
        }

        // ── Execute graph ────────────────────────────────────────────────────
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, x);
        ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/4);

        std::vector<float> out(n_tok * Pd);
        memcpy(out.data(), x->data, n_tok * Pd * sizeof(float));

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
