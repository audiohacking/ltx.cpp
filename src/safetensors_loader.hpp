#pragma once

// safetensors_loader.hpp – lightweight header-only safetensors reader (F32/BF16 only).
// Used by ltx_lora.hpp and other modules that need direct tensor access without ggml.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct SafetensorsLoader {
    struct TensorInfo {
        std::string           name;
        std::vector<int64_t>  shape;
        uint64_t              off_begin = 0, off_end = 0;
        bool                  is_bf16   = false;
    };

    std::vector<TensorInfo>       tensors_;
    std::map<std::string, size_t> name_idx_;
    std::vector<uint8_t>          file_data_;   // entire file mmap'd into RAM
    uint64_t                      data_base_ = 0;

    bool load(const std::string & path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        uint64_t file_size = (uint64_t)f.tellg();
        f.seekg(0);
        if (file_size < 8) return false;

        file_data_.resize(file_size);
        f.read(reinterpret_cast<char *>(file_data_.data()), (std::streamsize)file_size);
        if (!f) return false;

        uint64_t header_size = 0;
        std::memcpy(&header_size, file_data_.data(), 8);
        if (header_size > file_size - 8 || header_size > 64u * 1024 * 1024) return false;
        data_base_ = 8 + header_size;

        const char * h     = reinterpret_cast<const char *>(file_data_.data()) + 8;
        const char * h_end = h + header_size;
        const char * p     = h;

        while (p < h_end) {
            // Find next quoted key.
            while (p < h_end && *p != '"') ++p;
            if (p >= h_end) break;
            ++p;
            const char * key_s = p;
            while (p < h_end && *p != '"') ++p;
            if (p >= h_end) break;
            std::string key(key_s, (size_t)(p - key_s));
            ++p;

            // Skip past ':'
            while (p < h_end && *p != ':') ++p;
            if (p >= h_end) break;
            ++p;
            while (p < h_end && (*p == ' ' || *p == '\t' || *p == '\n')) ++p;
            if (p >= h_end || *p != '{') { ++p; continue; }

            // Find matching '}'
            const char * obj = p;
            int depth = 0;
            while (p < h_end) {
                if (*p == '{') ++depth;
                else if (*p == '}') { --depth; if (depth == 0) { ++p; break; } }
                ++p;
            }
            const char * obj_end = p;

            if (key == "__metadata__") continue;

            TensorInfo ti;
            ti.name = key;

            // dtype
            const char * dt = std::strstr(obj, "\"dtype\"");
            if (!dt || dt >= obj_end) continue;
            dt = std::strchr(dt, ':');
            if (!dt || dt >= obj_end) continue;
            ++dt;
            while (dt < obj_end && (*dt == ' ' || *dt == '"')) ++dt;
            if (std::strncmp(dt, "BF16", 4) == 0 || std::strncmp(dt, "bf16", 4) == 0)
                ti.is_bf16 = true;
            else if (std::strncmp(dt, "F32", 3) != 0 && std::strncmp(dt, "f32", 3) != 0 &&
                     std::strncmp(dt, "FP32", 4) != 0)
                continue;  // unsupported dtype

            // shape
            const char * sh = std::strstr(obj, "\"shape\"");
            if (!sh || sh >= obj_end) continue;
            sh = std::strchr(sh, '[');
            if (!sh || sh >= obj_end) continue;
            ++sh;
            while (sh < obj_end && *sh != ']') {
                while (sh < obj_end && (*sh == ',' || *sh == ' ')) ++sh;
                if (sh >= obj_end || *sh == ']') break;
                unsigned long d = 0;
                if (sscanf(sh, "%lu", &d) != 1) break;
                ti.shape.push_back((int64_t)d);
                while (sh < obj_end && *sh != ',' && *sh != ']') ++sh;
            }

            // data_offsets
            const char * do_ = std::strstr(obj, "\"data_offsets\"");
            if (!do_ || do_ >= obj_end) continue;
            do_ = std::strchr(do_, '[');
            if (!do_ || do_ >= obj_end) continue;
            ++do_;
            unsigned long a = 0, b = 0;
            if (sscanf(do_, "%lu,%lu", &a, &b) != 2) continue;
            ti.off_begin = a; ti.off_end = b;

            if (ti.off_end <= ti.off_begin || data_base_ + ti.off_end > file_size) continue;

            size_t idx = tensors_.size();
            tensors_.push_back(std::move(ti));
            name_idx_[tensors_.back().name] = idx;
        }
        return !tensors_.empty();
    }

    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (const auto & t : tensors_) names.push_back(t.name);
        return names;
    }

    std::vector<int64_t> tensor_shape(const std::string & name) const {
        auto it = name_idx_.find(name);
        if (it == name_idx_.end()) return {};
        return tensors_[it->second].shape;
    }

    std::vector<float> tensor_f32(const std::string & name) const {
        auto it = name_idx_.find(name);
        if (it == name_idx_.end()) return {};
        const TensorInfo & ti = tensors_[it->second];

        size_t nelem = 1;
        for (int64_t d : ti.shape) nelem *= (size_t)d;
        std::vector<float> out(nelem);

        const uint8_t * src = file_data_.data() + data_base_ + ti.off_begin;
        uint64_t nbytes = ti.off_end - ti.off_begin;

        if (ti.is_bf16) {
            if (nbytes != nelem * 2) return {};
            const uint16_t * bf16 = reinterpret_cast<const uint16_t *>(src);
            for (size_t i = 0; i < nelem; ++i) {
                uint32_t u = (uint32_t)bf16[i] << 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else {
            if (nbytes != nelem * sizeof(float)) return {};
            std::memcpy(out.data(), src, nbytes);
        }
        return out;
    }
};
