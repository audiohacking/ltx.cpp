// safetensors_loader.cpp – load safetensors (e.g. VAE) into ggml_context for ltx.cpp
// Reference: https://github.com/huggingface/safetensors
// Format: 8-byte header size (LE uint64), then JSON header, then tensor data.

#include "ltx_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Parse a single tensor entry in the JSON: "key": { "dtype": "F32", "shape": [...], "data_offsets": [a,b] }
// Returns tensor name, dtype ("F32"/"BF16"), shape, and data_offsets. Returns false on parse error.
bool parse_tensor_entry(const char * start, const char * end,
        std::string & name, std::string & dtype, std::vector<int64_t> & shape,
        uint64_t & off_begin, uint64_t & off_end)
{
    name.clear();
    dtype.clear();
    shape.clear();
    off_begin = off_end = 0;
    const char * p = start;
    if (p >= end || *p != '"') return false;
    ++p;
    const char * name_start = p;
    while (p < end && *p != '"') ++p;
    if (p >= end || (p - name_start) <= 0 || (p - name_start) > 512) return false;
    name.assign(name_start, (size_t)(p - name_start));
    ++p;
    while (p < end && *p != '{') ++p;
    if (p >= end) return false;
    const char * obj = p;

    // Find "data_offsets":[ start, end ]
    const char * do_str = std::strstr(obj, "\"data_offsets\"");
    if (!do_str || do_str >= end) return false;
    do_str = std::strchr(do_str, '[');
    if (!do_str || do_str >= end) return false;
    ++do_str;
    unsigned long a = 0, b = 0;
    if (sscanf(do_str, "%lu,%lu", &a, &b) != 2) return false;
    off_begin = a;
    off_end = b;

    // Find "dtype":"F32" or "BF16" (value after colon)
    const char * dt_str = std::strstr(obj, "\"dtype\"");
    if (!dt_str || dt_str >= end) return false;
    dt_str = std::strchr(dt_str, ':');
    if (!dt_str || dt_str >= end) return false;
    ++dt_str;
    while (dt_str < end && (*dt_str == ' ' || *dt_str == '\t')) ++dt_str;
    if (dt_str >= end || *dt_str != '"') return false;
    ++dt_str;  // first char of value
    const char * dt_end = std::strchr(dt_str, '"');
    if (!dt_end || dt_end >= end || (dt_end - dt_str) <= 0 || (dt_end - dt_str) > 16) return false;
    dtype.assign(dt_str, (size_t)(dt_end - dt_str));

    // Find "shape":[ ... ]
    const char * sh_str = std::strstr(obj, "\"shape\"");
    if (!sh_str || sh_str >= end) return false;
    sh_str = std::strchr(sh_str, '[');
    if (!sh_str || sh_str >= end) return false;
    ++sh_str;
    while (sh_str < end && *sh_str != ']') {
        while (sh_str < end && (*sh_str == ',' || *sh_str == ' ')) ++sh_str;
        if (sh_str >= end || *sh_str == ']') break;
        unsigned long dim = 0;
        if (sscanf(sh_str, "%lu", &dim) != 1) break;
        shape.push_back((int64_t)dim);
        while (sh_str < end && *sh_str != ',' && *sh_str != ']') ++sh_str;
    }
    return true;
}

// Find next tensor key in JSON. Skip __metadata__ (and skip its whole value). Returns start of next entry or end.
const char * next_tensor_key(const char * & p, const char * end)
{
    while (p < end) {
        if (*p == '"') {
            const char * key_start = p + 1;
            const char * key_end = key_start;
            while (key_end < end && *key_end != '"' && (key_end - key_start) < 512) ++key_end;
            if (key_end >= end || *key_end != '"') { ++p; continue; }
            size_t key_len = (size_t)(key_end - key_start);
            if (key_len == 0) { p = key_end + 1; continue; }
            if (key_len == 13 && std::memcmp(key_start, "__metadata__", 13) == 0) {
                p = key_end + 1;
                while (p < end && *p != ':') ++p;
                if (p >= end) return end;
                ++p;
                while (p < end && (*p == ' ' || *p == '\t')) ++p;
                if (p >= end || *p != '{') return end;
                int depth = 1;
                ++p;
                while (p < end && depth > 0) {
                    if (*p == '{') ++depth;
                    else if (*p == '}') --depth;
                    ++p;
                }
                continue;
            }
            p = key_end;
            return key_start - 1;  // point to opening "
        }
        ++p;
    }
    return end;
}

} // namespace

bool LtxGgufModel::open_safetensors(const std::string & fpath)
{
    path = fpath;
    std::ifstream f(fpath, std::ios::binary | std::ios::ate);
    if (!f) {
        LTX_ERR("failed to open safetensors file: %s", fpath.c_str());
        return false;
    }
    uint64_t file_size = (uint64_t)f.tellg();
    f.seekg(0);
    if (file_size < 8) {
        LTX_ERR("safetensors file too small: %s", fpath.c_str());
        return false;
    }
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char *>(&header_size), 8);
    if (header_size > file_size - 8 || header_size > 1024 * 1024 * 10) {
        LTX_ERR("invalid safetensors header size: %llu", (unsigned long long)header_size);
        return false;
    }
    std::string header(header_size, '\0');
    f.read(&header[0], header_size);
    uint64_t data_base = 8 + header_size;
    if (data_base > file_size) {
        LTX_ERR("safetensors data base past file end");
        return false;
    }

    // Count tensors and compute total size for ggml context
    struct TensorInfo {
        std::string name;
        std::string dtype;
        std::vector<int64_t> shape;
        uint64_t off_begin, off_end;
    };
    std::vector<TensorInfo> tensors;
    const char * h = header.c_str();
    const char * h_end = h + header.size();
    const char * p = h;
    p = next_tensor_key(p, h_end);
    while (p < h_end) {
        const char * entry_start = p;
        const char * obj_start = std::strchr(p, '{');
        if (!obj_start || obj_start >= h_end) break;
        int depth = 0;
        const char * entry_end = obj_start;
        for (const char * q = obj_start; q < h_end; ++q) {
            if (*q == '{') ++depth;
            else if (*q == '}') {
                --depth;
                if (depth == 0) { entry_end = q; break; }
            }
        }
        if (entry_end <= obj_start || (entry_end + 1) <= entry_start) break;
        TensorInfo ti;
        if (parse_tensor_entry(entry_start, entry_end + 1, ti.name, ti.dtype, ti.shape, ti.off_begin, ti.off_end)) {
            if (ti.off_end > ti.off_begin && data_base + ti.off_end <= file_size)
                tensors.push_back(std::move(ti));
        }
        p = entry_end + 1;
        while (p < h_end && (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r')) ++p;
        if (p < h_end && *p == '"') {
            const char * next = next_tensor_key(p, h_end);
            p = next;
        } else
            break;
    }

    size_t total_elems = 0;
    for (const auto & t : tensors) {
        size_t n = 1;
        for (int64_t d : t.shape) n *= (size_t)d;
        total_elems += n;
    }
    size_t ctx_size = total_elems * sizeof(float) + 1024 * 1024;  // extra for ggml overhead
    struct ggml_init_params init_p{ ctx_size, nullptr, false };
    ggml_ctx = ggml_init(init_p);
    if (!ggml_ctx) {
        LTX_ERR("safetensors: ggml_init failed");
        return false;
    }

    std::vector<float> f32_buf;
    for (const auto & t : tensors) {
        size_t nelem = 1;
        for (int64_t d : t.shape) nelem *= (size_t)d;
        uint64_t nbytes = (t.off_end - t.off_begin);
        bool is_bf16 = (t.dtype == "BF16" || t.dtype == "bf16");
        bool is_f32 = (t.dtype == "F32" || t.dtype == "f32" || t.dtype == "FP32");
        if (!is_f32 && !is_bf16) {
            LTX_LOG("safetensors: skip tensor %s (dtype %s)", t.name.c_str(), t.dtype.c_str());
            continue;
        }
        int nd = (int)t.shape.size();
        if (nd <= 0 || nd > 4) continue;
        int64_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < nd; ++i) ne[i] = t.shape[i];

        struct ggml_tensor * ten = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, nd, ne);
        if (!ten) continue;

        f.seekg((std::streampos)(data_base + t.off_begin));
        if (is_f32) {
            if (nbytes != nelem * sizeof(float)) continue;
            f.read(reinterpret_cast<char *>(ten->data), nbytes);
        } else {
            if (nbytes != nelem * 2) continue;  // BF16 = 2 bytes per element
            f32_buf.resize(nelem);
            std::vector<uint16_t> bf16(nelem);
            f.read(reinterpret_cast<char *>(bf16.data()), nbytes);
            for (size_t i = 0; i < nelem; ++i) {
                uint32_t u = (uint32_t)bf16[i] << 16;
                f32_buf[i] = *reinterpret_cast<float *>(&u);
            }
            memcpy(ten->data, f32_buf.data(), nelem * sizeof(float));
        }
        std::string gname = t.name;
        if (gname.compare(0, 4, "vae.") != 0)
            gname = "vae." + t.name;
        ggml_set_name(ten, gname.c_str());
    }

    LTX_LOG("safetensors loaded: %s (%zu tensors)", fpath.c_str(), tensors.size());
    return true;
}
