#pragma once

// ltx_common.hpp – shared utilities for ltx.cpp
//
// Provides GGUF loading helpers, tensor access wrappers, and
// lightweight logging macros used across all ltx.cpp modules.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ── Logging ──────────────────────────────────────────────────────────────────

#define LTX_LOG(fmt, ...)  fprintf(stderr, "[ltx] " fmt "\n", ##__VA_ARGS__)
#define LTX_ERR(fmt, ...)  fprintf(stderr, "[ltx ERROR] " fmt "\n", ##__VA_ARGS__)
#define LTX_ABORT(fmt, ...) do { LTX_ERR(fmt, ##__VA_ARGS__); std::abort(); } while(0)

// ── GGUF model context ───────────────────────────────────────────────────────

struct LtxGgufModel {
    struct gguf_context * gguf_ctx  = nullptr;
    struct ggml_context * ggml_ctx  = nullptr;
    std::string           path;

    // Open a GGUF file: populate gguf_ctx for metadata and ggml_ctx for tensors.
    bool open(const std::string & fpath) {
        path = fpath;
        struct gguf_init_params p;
        p.no_alloc = false;
        p.ctx      = &ggml_ctx;
        gguf_ctx = gguf_init_from_file(fpath.c_str(), p);
        if (!gguf_ctx) {
            LTX_ERR("failed to open GGUF file: %s", fpath.c_str());
            return false;
        }
        return true;
    }

    ~LtxGgufModel() {
        if (gguf_ctx) gguf_free(gguf_ctx);
        if (ggml_ctx) ggml_free(ggml_ctx);
    }

    // Find a tensor by name; returns nullptr if not found.
    struct ggml_tensor * get_tensor(const char * name) const {
        return ggml_get_tensor(ggml_ctx, name);
    }

    // Open a safetensors file (e.g. VAE): populate ggml_ctx only; gguf_ctx stays nullptr.
    // Tensor names are stored with "vae." prefix when not already present.
    bool open_safetensors(const std::string & fpath);

    // Read a string KV value (returns "" if missing). Safe when gguf_ctx is nullptr.
    std::string kv_str(const char * key) const {
        if (!gguf_ctx) return "";
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return "";
        return gguf_get_val_str(gguf_ctx, idx);
    }

    int64_t kv_i64(const char * key, int64_t def = 0) const {
        if (!gguf_ctx) return def;
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return def;
        return gguf_get_val_i64(gguf_ctx, idx);
    }

    uint32_t kv_u32(const char * key, uint32_t def = 0) const {
        if (!gguf_ctx) return def;
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return def;
        return gguf_get_val_u32(gguf_ctx, idx);
    }

    float kv_f32(const char * key, float def = 0.0f) const {
        if (!gguf_ctx) return def;
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return def;
        return gguf_get_val_f32(gguf_ctx, idx);
    }
};

// Default: unlimited (migrate all tensors). Override with LTX_MIGRATE_MAX_TENSOR_MB env var
// (e.g. export LTX_MIGRATE_MAX_TENSOR_MB=4096 on GPUs with limited VRAM).
static constexpr size_t LTX_MIGRATE_MAX_TENSOR_BYTES_DEFAULT = (size_t)-1; // unlimited

// ── Backend: migrate context to backend (backend-agnostic) ───────────────────
// Moves all tensors in ctx onto the given backend so inference can run on GPU
// without backend-specific code. Uses one buffer per tensor to avoid single-buffer
// size limits. Tensors that exceed max_tensor_bytes are silently skipped (left on
// CPU); the scheduler handles mixed-backend graphs transparently.
// Caller must free the returned buffers when done (stored in buf_out).
// Returns number of buffers created, or 0 if no tensors were migrated.
// Env LTX_MIGRATE_MAX_TENSOR_MB: max size per tensor in MB; 0 = no limit (try full migration).
static inline int ltx_backend_migrate_ctx(ggml_context * ctx, ggml_backend_t backend,
        std::vector<ggml_backend_buffer_t> & buf_out) {
    buf_out.clear();
    if (!ctx || !backend) return 0;
    size_t max_tensor_bytes = LTX_MIGRATE_MAX_TENSOR_BYTES_DEFAULT;
    if (const char * env = std::getenv("LTX_MIGRATE_MAX_TENSOR_MB")) {
        long mb = std::atol(env);
        if (mb >= 0) max_tensor_bytes = (mb == 0) ? (size_t)-1 : (size_t)mb * 1024u * 1024u;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    // First pass: collect (tensor, alloc_size) for tensors that fit.
    std::vector<std::pair<ggml_tensor *, size_t>> to_migrate;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        size_t sz = ggml_backend_buft_get_alloc_size(buft, t);
        if (sz == 0 || sz > max_tensor_bytes) continue; // skip zero-size or oversized
        to_migrate.push_back({t, sz});
    }
    if (to_migrate.empty()) return 0;
    // Second pass: allocate backend buffers and copy.
    buf_out.reserve(to_migrate.size());
    for (auto & [t, sz] : to_migrate) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(backend, sz);
        if (!buf) {
            for (ggml_backend_buffer_t b : buf_out) ggml_backend_buffer_free(b);
            buf_out.clear();
            return 0;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buf_out.push_back(buf);
    }
    for (size_t i = 0; i < to_migrate.size(); ++i) {
        ggml_tensor * t   = to_migrate[i].first;
        void * old_data   = t->data;
        size_t nbytes     = ggml_nbytes(t);
        t->data   = nullptr;
        t->buffer = nullptr;
        ggml_tallocr talloc = ggml_tallocr_new(buf_out[i]);
        ggml_tallocr_alloc(&talloc, t);
        ggml_backend_tensor_set(t, old_data, 0, nbytes);
    }
    return (int)buf_out.size();
}

// ── Tensor helpers ───────────────────────────────────────────────────────────

// Convenient float-buffer access to a 1-D or flat-viewed tensor.
static inline float * f32_data(struct ggml_tensor * t) {
    return reinterpret_cast<float *>(t->data);
}

// Number of elements in a tensor.
static inline int64_t ggml_nelements_safe(const struct ggml_tensor * t) {
    return ggml_nelements(t);
}

// ── RNG ──────────────────────────────────────────────────────────────────────

struct LtxRng {
    std::mt19937 gen;
    std::normal_distribution<float> nd{0.0f, 1.0f};

    explicit LtxRng(uint64_t seed = 42) : gen(seed) {}

    float next() { return nd(gen); }

    // Fill a flat float buffer with N(0,1) samples.
    void fill(float * buf, size_t n) {
        for (size_t i = 0; i < n; ++i) buf[i] = next();
    }
};

// ── Sigmoid / softmax helpers (CPU only, small tensors) ─────────────────────

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ── Video frame buffer ───────────────────────────────────────────────────────

// Stores decoded video as uint8 RGB frames [frames][height][width][3].
struct VideoBuffer {
    int frames, height, width;
    std::vector<uint8_t> data; // frames * height * width * 3

    VideoBuffer(int F, int H, int W)
        : frames(F), height(H), width(W), data(F * H * W * 3, 0) {}

    uint8_t * frame(int f) { return data.data() + f * height * width * 3; }
    const uint8_t * frame(int f) const { return data.data() + f * height * width * 3; }

    // Clamp float [-1,1] → uint8 [0,255].
    static uint8_t clamp_u8(float v) {
        int i = static_cast<int>((v + 1.0f) * 127.5f + 0.5f);
        return static_cast<uint8_t>(i < 0 ? 0 : (i > 255 ? 255 : i));
    }
};

// ── Simple PPM writer ────────────────────────────────────────────────────────

static void write_ppm(const std::string & path, const uint8_t * rgb, int W, int H) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { LTX_ERR("cannot write %s", path.c_str()); return; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(rgb, 1, W * H * 3, f);
    fclose(f);
}

// Write video frames as individual PPM files (frame_0000.ppm, …).
static void write_video_frames(const VideoBuffer & vbuf, const std::string & out_prefix) {
    for (int f = 0; f < vbuf.frames; ++f) {
        char fname[512];
        snprintf(fname, sizeof(fname), "%s_%04d.ppm", out_prefix.c_str(), f);
        write_ppm(fname, vbuf.frame(f), vbuf.width, vbuf.height);
    }
    LTX_LOG("wrote %d PPM frames with prefix '%s'", vbuf.frames, out_prefix.c_str());
}

// Image loading: implemented in ltx-generate.cpp (single TU with STB_IMAGE_IMPLEMENTATION).
VideoBuffer load_image(const std::string & path);
std::vector<uint8_t> resize_bilinear(
    const uint8_t * src, int W_src, int H_src, int W_dst, int H_dst);
