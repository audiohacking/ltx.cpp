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

    // Read a string KV value (returns "" if missing).
    std::string kv_str(const char * key) const {
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return "";
        return gguf_get_val_str(gguf_ctx, idx);
    }

    int64_t kv_i64(const char * key, int64_t def = 0) const {
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return def;
        return gguf_get_val_i64(gguf_ctx, idx);
    }

    uint32_t kv_u32(const char * key, uint32_t def = 0) const {
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return def;
        return gguf_get_val_u32(gguf_ctx, idx);
    }

    float kv_f32(const char * key, float def = 0.0f) const {
        int64_t idx = gguf_find_key(gguf_ctx, key);
        if (idx < 0) return def;
        return gguf_get_val_f32(gguf_ctx, idx);
    }
};

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

// ── PPM image reader ─────────────────────────────────────────────────────────
//
// Supports binary PPM (P6) and ASCII PPM (P5/P6).
// Returns raw uint8 RGB pixels in a VideoBuffer with frames=1.
// On failure returns an empty VideoBuffer (frames=0).

static VideoBuffer load_ppm(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LTX_ERR("cannot open image: %s", path.c_str());
        return VideoBuffer(0, 0, 0);
    }

    auto skip_ws_comments = [&]() {
        int c;
        while ((c = fgetc(f)) != EOF) {
            if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n') {} }
            else if (c != ' ' && c != '\t' && c != '\n' && c != '\r') { ungetc(c, f); break; }
        }
    };

    char magic[3] = {};
    if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || (magic[1] != '6' && magic[1] != '3')) {
        LTX_ERR("unsupported image format (only P6 binary PPM supported): %s", path.c_str());
        fclose(f);
        return VideoBuffer(0, 0, 0);
    }
    bool binary = (magic[1] == '6');

    skip_ws_comments();
    int W = 0, H = 0, maxval = 0;
    if (fscanf(f, "%d", &W) != 1 || W <= 0) { fclose(f); return VideoBuffer(0, 0, 0); }
    skip_ws_comments();
    if (fscanf(f, "%d", &H) != 1 || H <= 0) { fclose(f); return VideoBuffer(0, 0, 0); }
    skip_ws_comments();
    if (fscanf(f, "%d", &maxval) != 1 || maxval <= 0) { fclose(f); return VideoBuffer(0, 0, 0); }
    // Consume exactly one whitespace character after maxval.
    (void)fgetc(f);

    VideoBuffer buf(1, H, W);
    uint8_t * dst = buf.frame(0);

    if (binary) {
        size_t npix = (size_t)W * H * 3;
        if (maxval <= 255) {
            if (fread(dst, 1, npix, f) != npix) {
                LTX_ERR("truncated PPM: %s", path.c_str());
                fclose(f);
                return VideoBuffer(0, 0, 0);
            }
        } else {
            // 16-bit PPM: read big-endian uint16 and scale to uint8.
            for (size_t i = 0; i < npix; ++i) {
                int hi = fgetc(f), lo = fgetc(f);
                if (hi == EOF || lo == EOF) break;
                dst[i] = (uint8_t)(((hi << 8) | lo) * 255 / maxval);
            }
        }
    } else {
        // ASCII PPM (P3).
        for (int i = 0; i < W * H * 3; ++i) {
            int v = 0;
            if (fscanf(f, "%d", &v) != 1) break;
            dst[i] = (uint8_t)(v * 255 / maxval);
        }
    }

    fclose(f);
    LTX_LOG("loaded PPM: %s (%dx%d)", path.c_str(), W, H);
    return buf;
}

// Bilinear resize of a uint8 RGB image [H_src × W_src × 3] → [H_dst × W_dst × 3].
static std::vector<uint8_t> resize_bilinear(
        const uint8_t * src, int W_src, int H_src,
        int W_dst, int H_dst)
{
    std::vector<uint8_t> out(W_dst * H_dst * 3);
    float sx = (float)W_src / W_dst;
    float sy = (float)H_src / H_dst;

    for (int yd = 0; yd < H_dst; ++yd)
    for (int xd = 0; xd < W_dst; ++xd) {
        float xf = (xd + 0.5f) * sx - 0.5f;
        float yf = (yd + 0.5f) * sy - 0.5f;
        int x0 = std::max(0, (int)xf),         x1 = std::min(W_src - 1, x0 + 1);
        int y0 = std::max(0, (int)yf),         y1 = std::min(H_src - 1, y0 + 1);
        float qx = xf - x0, qy = yf - y0;

        for (int c = 0; c < 3; ++c) {
            float v00 = src[(y0 * W_src + x0) * 3 + c];
            float v10 = src[(y0 * W_src + x1) * 3 + c];
            float v01 = src[(y1 * W_src + x0) * 3 + c];
            float v11 = src[(y1 * W_src + x1) * 3 + c];
            float v = (1 - qy) * ((1 - qx) * v00 + qx * v10)
                    +      qy  * ((1 - qx) * v01 + qx * v11);
            out[(yd * W_dst + xd) * 3 + c] = (uint8_t)(v + 0.5f);
        }
    }
    return out;
}
