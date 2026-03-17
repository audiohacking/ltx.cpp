// ltx-quantize.cpp – GGUF requantizer for LTX-Video models
//
// Reads a BF16/F32 GGUF and writes a new GGUF with all eligible tensors
// quantized to the requested type (e.g. Q4_K_M, Q8_0, Q5_K_M).
//
// Usage:
//   ltx-quantize <input.gguf> <output.gguf> <quant_type>
//
//   quant_type: Q4_K_M | Q5_K_M | Q6_K | Q8_0 | BF16 | F32

#include "ltx_common.hpp"
#include "gguf.h"
#include <cstring>
#include <map>
#include <string>

static const std::map<std::string, ggml_type> QUANT_MAP = {
    {"Q4_K_M", GGML_TYPE_Q4_K},
    {"Q5_K_M", GGML_TYPE_Q5_K},
    {"Q6_K",   GGML_TYPE_Q6_K},
    {"Q8_0",   GGML_TYPE_Q8_0},
    {"BF16",   GGML_TYPE_BF16},
    {"F32",    GGML_TYPE_F32},
    {"F16",    GGML_TYPE_F16},
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <input.gguf> <output.gguf> <quant_type>\n"
        "  quant_type: Q4_K_M | Q5_K_M | Q6_K | Q8_0 | BF16 | F32\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc != 4) { print_usage(argv[0]); return 1; }

    std::string in_path   = argv[1];
    std::string out_path  = argv[2];
    std::string quant_str = argv[3];

    auto it = QUANT_MAP.find(quant_str);
    if (it == QUANT_MAP.end()) {
        fprintf(stderr, "Unknown quant type: %s\n", quant_str.c_str());
        print_usage(argv[0]);
        return 1;
    }
    ggml_type target_type = it->second;

    LTX_LOG("quantize: %s -> %s  [%s]", in_path.c_str(), out_path.c_str(), quant_str.c_str());

    // Load source model.
    LtxGgufModel src;
    if (!src.open(in_path)) return 1;

    // Iterate tensors and quantize.
    int n_tensors = gguf_get_n_tensors(src.gguf_ctx);
    LTX_LOG("source has %d tensors", n_tensors);

    // Build output GGUF.
    struct gguf_context * out_ctx = gguf_init_empty();

    // Copy all KV metadata (scalars and arrays alike).
    gguf_set_kv(out_ctx, src.gguf_ctx);

    // Add quantized tensors.
    for (int ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(src.gguf_ctx, ti);
        struct ggml_tensor * t = ggml_get_tensor(src.ggml_ctx, name);
        if (!t) { LTX_ERR("tensor not found: %s", name); continue; }

        // Only quantize 2-D+ float tensors; keep 1-D (biases, norms) as F32.
        ggml_type out_type = target_type;
        bool is_1d = (t->ne[1] <= 1 && t->ne[2] <= 1 && t->ne[3] <= 1);
        if (is_1d) out_type = GGML_TYPE_F32;
        // Embeddings also stay F32.
        if (strstr(name, "embed") && strstr(name, "weight")) out_type = GGML_TYPE_F32;

        if (t->type == out_type) {
            gguf_add_tensor(out_ctx, t);
            LTX_LOG("  keep   [%s]  %-40s", ggml_type_name(t->type), name);
            continue;
        }

        // Convert to F32 first if needed, then quantize.
        size_t n_elems = (size_t)ggml_nelements(t);

        std::vector<float> f32buf(n_elems);
        if (t->type == GGML_TYPE_F32) {
            memcpy(f32buf.data(), t->data, n_elems * sizeof(float));
        } else if (t->type == GGML_TYPE_BF16) {
            const uint16_t * src_bf = reinterpret_cast<const uint16_t *>(t->data);
            for (size_t i = 0; i < n_elems; ++i) {
                uint32_t u = (uint32_t)src_bf[i] << 16;
                memcpy(&f32buf[i], &u, 4);
            }
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(t->data),
                                  f32buf.data(), (int64_t)n_elems);
        } else {
            // Dequantize using type traits.
            const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
            if (traits && traits->to_float) {
                traits->to_float(t->data, f32buf.data(), (int64_t)n_elems);
            } else {
                LTX_ERR("no dequantization support for type %s", ggml_type_name(t->type));
                continue;
            }
        }

        // Quantize.
        size_t qsize = ggml_row_size(out_type, (int64_t)(n_elems / t->ne[0]));
        qsize *= t->ne[0]; // total quantized bytes
        std::vector<uint8_t> qbuf(qsize);

        ggml_quantize_chunk(out_type, f32buf.data(), qbuf.data(),
                            0, t->ne[1] > 0 ? t->ne[1] : 1, t->ne[0], nullptr);

        // Build a temporary tensor with the quantized data and add it.
        // We create a ggml_context just for this tensor header.
        size_t tmp_mem = sizeof(struct ggml_tensor) + 128;
        struct ggml_init_params tp{tmp_mem, nullptr, true /*no_alloc*/};
        struct ggml_context * tctx = ggml_init(tp);
        struct ggml_tensor * qt = ggml_new_tensor(tctx, out_type, GGML_MAX_DIMS, t->ne);
        ggml_set_name(qt, name);
        qt->data = qbuf.data();
        gguf_add_tensor(out_ctx, qt);
        ggml_free(tctx);

        LTX_LOG("  quant  [%s→%s]  %-40s  (%.1f MB → %.1f MB)",
                ggml_type_name(t->type), ggml_type_name(out_type), name,
                (double)(n_elems * sizeof(float)) / 1e6,
                (double)qsize / 1e6);
    }

    // Write output file.
    gguf_write_to_file(out_ctx, out_path.c_str(), false);
    gguf_free(out_ctx);

    LTX_LOG("quantize done: %s", out_path.c_str());
    return 0;
}
