#pragma once

// ltx_lora.hpp – LoRA weight loading and GPU-resident in-graph application for LTX-Video DiT
//
// Loads rank-384 LoRA from ltx-2.3-22b-distilled-lora-384.safetensors.
// Only the video-path weights are used (attn1, attn2, ff for transformer_blocks.N.*).
// Audio-path modules (audio_attn*, audio_to_video_attn, video_to_audio_attn) are
// skipped because our GGUF model is video-only.
//
// LoRA naming convention in the safetensors file:
//   diffusion_model.transformer_blocks.{N}.attn1.to_q.lora_A.weight  [rank, in_features]
//   diffusion_model.transformer_blocks.{N}.attn1.to_q.lora_B.weight  [out_features, rank]
//
// Merge formula (applied per linear layer in the GGML graph):
//   y = W @ x  →  y = W @ x + scale * lora_B @ (lora_A @ x)
//
// GPU residency:
//   Call gpu_upload(backend, buf_out) once after load() to migrate all A/B matrices
//   to the Metal/CUDA backend.  After upload, lora_A/gpu_A point to GPU-resident tensors
//   that can be referenced directly in no_alloc GGML graphs — zero per-step data copy.
//   The CPU-side float vectors (A, B) are freed after upload to reclaim ~7 GB RAM.

#include "ltx_common.hpp"
#include "safetensors_loader.hpp"

#include <cstring>
#include <map>
#include <string>
#include <vector>

struct LoRAPair {
    std::vector<float> A;              // [rank, in_features]  — freed after gpu_upload
    std::vector<float> B;             // [out_features, rank] — freed after gpu_upload
    int rank = 0, in_feat = 0, out_feat = 0;
    struct ggml_tensor * gpu_A = nullptr;  // Metal/CPU-resident after gpu_upload
    struct ggml_tensor * gpu_B = nullptr;
};

struct LtxLoRA {
    std::map<std::string, LoRAPair> pairs; // key = canonical layer name
    float scale = 1.0f;                    // lora_scale / rank (baked at load time)

    // Persistent ggml context holding tensor metadata for GPU-resident weights.
    struct ggml_context * weight_ctx = nullptr;

    ~LtxLoRA() {
        if (weight_ctx) { ggml_free(weight_ctx); weight_ctx = nullptr; }
    }

    // ── Load from safetensors ──────────────────────────────────────────────────
    // Only loads video-path weights that match our DiT.
    // lora_scale: strength multiplier (1.0 = full strength, as in ComfyUI).
    bool load(const std::string & path, float lora_scale = 1.0f) {
        SafetensorsLoader st;
        if (!st.load(path)) {
            LTX_ERR("LoRA: failed to load %s", path.c_str());
            return false;
        }
        // scale = lora_scale / rank (rank=384 for this file; divide to normalise)
        const float rank_f = 384.0f;
        scale = lora_scale / rank_f;

        std::map<std::string, std::vector<float>> A_map, B_map;
        std::map<std::string, std::vector<int64_t>> A_shape, B_shape;

        const std::string prefix = "diffusion_model.";
        for (const auto & name : st.tensor_names()) {
            // Skip audio-path modules (not present in video-only GGUF).
            if (name.find("audio_") != std::string::npos) continue;
            if (name.find("av_ca_") != std::string::npos) continue;
            if (name.find("video_to_audio") != std::string::npos) continue;
            if (name.find("prompt_adaln") != std::string::npos) continue;

            bool is_A = (name.rfind(".lora_A.weight") != std::string::npos);
            bool is_B = (name.rfind(".lora_B.weight") != std::string::npos);
            if (!is_A && !is_B) continue;

            // Strip "diffusion_model." prefix and ".lora_A/B.weight" suffix.
            std::string key = name;
            if (key.substr(0, prefix.size()) == prefix)
                key = key.substr(prefix.size());
            if (is_A) key = key.substr(0, key.size() - std::string(".lora_A.weight").size());
            else      key = key.substr(0, key.size() - std::string(".lora_B.weight").size());

            auto shape = st.tensor_shape(name);
            auto data  = st.tensor_f32(name);
            if (data.empty()) continue;

            if (is_A) { A_map[key] = std::move(data); A_shape[key] = shape; }
            else      { B_map[key] = std::move(data); B_shape[key] = shape; }
        }

        for (auto & [key, Adata] : A_map) {
            auto Bit = B_map.find(key);
            if (Bit == B_map.end()) continue;
            auto & as = A_shape[key];  // [rank, in_feat]
            auto & bs = B_shape[key];  // [out_feat, rank]
            if (as.size() < 2 || bs.size() < 2) continue;
            LoRAPair p;
            p.rank     = (int)as[0];
            p.in_feat  = (int)as[1];
            p.out_feat = (int)bs[0];
            p.A = std::move(Adata);
            p.B = std::move(Bit->second);
            pairs[key] = std::move(p);
        }

        LTX_LOG("LoRA loaded: %zu video-path pairs, scale=%.4f (lora_scale=%.1f, rank=%d)",
                pairs.size(), (double)scale, (double)lora_scale, (int)rank_f);
        return !pairs.empty();
    }

    // ── Upload all A/B matrices to a backend (Metal/CUDA/CPU) ─────────────────
    // Creates a single ggml_context for all LoRA tensors, copies float data in,
    // then migrates to the backend.  buf_out receives the allocated backend buffers
    // (caller must keep them alive and free them when done).
    // After this call, gpu_A/gpu_B point to GPU-resident tensors ready for graph use.
    // The CPU float vectors (A, B) are freed to reclaim ~7 GB RAM.
    bool gpu_upload(ggml_backend_t backend, std::vector<ggml_backend_buffer_t> & buf_out) {
        if (pairs.empty() || !backend) return false;

        // Allocate a single ggml_context large enough for all tensor metadata + float data.
        size_t n_tensors = pairs.size() * 2;
        size_t ctx_meta  = ggml_tensor_overhead() * n_tensors + 4096;
        size_t data_bytes = 0;
        for (auto & [k, p] : pairs)
            data_bytes += (p.A.size() + p.B.size()) * sizeof(float);

        struct ggml_init_params ip = { ctx_meta + data_bytes, nullptr, /*no_alloc=*/false };
        weight_ctx = ggml_init(ip);
        if (!weight_ctx) {
            LTX_ERR("LoRA: failed to allocate weight context (%zu MB)",
                    (ctx_meta + data_bytes) / (1024 * 1024));
            return false;
        }

        // Create F32 tensors and copy data in.
        // GGML storage: ne[0] = innermost (contiguous) dimension.
        // Python A:[rank, in_feat] → GGML ne[0]=in_feat, ne[1]=rank.
        // Python B:[out_feat, rank] → GGML ne[0]=rank, ne[1]=out_feat.
        for (auto & [key, p] : pairs) {
            p.gpu_A = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, p.in_feat, p.rank);
            p.gpu_B = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, p.rank,    p.out_feat);
            if (!p.gpu_A || !p.gpu_B) {
                LTX_ERR("LoRA: tensor alloc failed for key %s", key.c_str());
                continue;
            }
            memcpy(p.gpu_A->data, p.A.data(), p.A.size() * sizeof(float));
            memcpy(p.gpu_B->data, p.B.data(), p.B.size() * sizeof(float));
        }

        // Migrate to backend (Metal/CUDA).
        int n_bufs = ltx_backend_migrate_ctx(weight_ctx, backend, buf_out);

        // Free CPU float storage — data is now on GPU.
        for (auto & [key, p] : pairs) {
            p.A.clear(); p.A.shrink_to_fit();
            p.B.clear(); p.B.shrink_to_fit();
        }

        size_t total_mb = 0;
        for (auto b : buf_out) total_mb += ggml_backend_buffer_get_size(b) / (1024 * 1024);
        LTX_LOG("LoRA weights on GPU: %d buffers, %zu MB  (CPU vectors freed)",
                n_bufs, total_mb);
        return n_bufs > 0;
    }

    const LoRAPair * find(const std::string & key) const {
        auto it = pairs.find(key);
        return it != pairs.end() ? &it->second : nullptr;
    }
};
