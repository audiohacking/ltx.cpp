# ltx.cpp — Developer Guide

This document collects everything a new contributor needs to understand the
codebase, set up their development environment, extend the implementation, and
navigate the known limitations.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Repository layout](#2-repository-layout)
3. [Getting started](#3-getting-started)
   - [Prerequisites](#prerequisites)
   - [Clone & initialise submodules](#clone--initialise-submodules)
   - [Build configurations](#build-configurations)
   - [Obtaining model files](#obtaining-model-files)
4. [End-to-end data flow](#4-end-to-end-data-flow)
5. [Source file reference](#5-source-file-reference)
   - [ltx\_common.hpp](#ltx_commonhpp)
   - [scheduler.hpp](#schedulerhpp)
   - [t5\_encoder.hpp](#t5_encoderhpp)
   - [video\_vae.hpp](#video_vaehpp)
   - [ltx\_dit.hpp](#ltx_dithpp)
   - [ltx-generate.cpp](#ltx-generatecpp)
   - [ltx-quantize.cpp](#ltx-quantizecpp)
   - [convert.py](#convertpy)
6. [GGUF model format conventions](#6-gguf-model-format-conventions)
   - [DiT GGUF](#dit-gguf)
   - [VAE GGUF](#vae-gguf)
   - [T5 GGUF](#t5-gguf)
7. [Image-to-video (I2V) design](#7-image-to-video-i2v-design)
   - [VaeEncoder](#vaeencoder)
   - [Frame-conditioning schedule](#frame-conditioning-schedule)
   - [Hard-pinning at t=0](#hard-pinning-at-t0)
8. [Key algorithms and design decisions](#8-key-algorithms-and-design-decisions)
   - [Rectified Flow (RF) scheduling](#rectified-flow-rf-scheduling)
   - [Classifier-free guidance](#classifier-free-guidance)
   - [Patchify / unpatchify](#patchify--unpatchify)
   - [Latent dimension formulas](#latent-dimension-formulas)
   - [Tokenizer](#tokenizer)
9. [Adding a new backend (GPU/Metal/Vulkan)](#9-adding-a-new-backend-gpumetalvulkan)
10. [Known limitations and open tasks](#10-known-limitations-and-open-tasks)
11. [Coding conventions](#11-coding-conventions)
12. [Testing](#12-testing)
13. [Contributing](#13-contributing)

---

## 1. Project overview

**ltx.cpp** is a self-contained C++17 inference engine for
[LTX-Video](https://github.com/Lightricks/LTX-Video) (Lightricks), built on
top of [GGML](https://github.com/ggml-org/ggml).

Goals:
- **No Python at runtime** — all inference is done from a single compiled binary.
- **Cross-platform** — CPU (any OS), CUDA, ROCm/HIP, Metal (macOS), Vulkan.
- **Memory-efficient** — weights stored and computed in quantised GGUF format
  (Q4\_K\_M through BF16).
- **Three generation modes**: text-to-video (T2V), image-to-video (I2V), and
  keyframe interpolation.

The project is intentionally *not* a 1:1 port of the original diffusers/PyTorch
code; instead it provides a minimal, readable C++ implementation that is easy
to extend.

---

## 2. Repository layout

```
ltx.cpp/
├── CMakeLists.txt        Build system (C++17 + GGML)
├── README.md             End-user documentation
├── DEV.md                ← this file
│
├── src/
│   ├── ltx_common.hpp    Shared utilities: GGUF loading, logging, VideoBuffer,
│   │                     image loading (stb_image), bilinear resize
│   ├── scheduler.hpp     Rectified-Flow Euler scheduler + CFG
│   ├── t5_encoder.hpp    T5-XXL text encoder (GGML graph)
│   ├── video_vae.hpp     CausalVideoVAE decoder + VaeEncoder (I2V)
│   ├── ltx_dit.hpp       LTX-Video DiT forward pass (GGML graph)
│   ├── ltx-generate.cpp  Main binary: argument parsing + inference orchestration
│   ├── ltx-quantize.cpp  Re-quantize GGUF files (BF16 → Q4_K_M / Q8_0 / …)
│   └── stb_image.h       Vendored stb_image v2.28 (public domain)
│
├── convert.py            Python: safetensors → GGUF conversion
├── checkpoints.sh        Download raw HF safetensors checkpoints
├── models.sh             Download pre-quantised GGUF models from Unsloth/HF
├── quantize.sh           Shell wrapper: run ltx-quantize on all BF16 GGUFs
│
└── ggml/                 Git submodule — GGML tensor library
```

**Key design rule**: every module is a single header-only file (`*.hpp`).
There is no separate `src/` library — headers are included directly by
`ltx-generate.cpp`.  This keeps the build trivial and avoids link-time
complexity.

---

## 3. Getting started

### Prerequisites

| Tool | Purpose | Minimum version |
|------|---------|-----------------|
| `cmake` | Build system | 3.16 |
| C++ compiler | Build | C++17 (GCC 9+, Clang 10+, MSVC 19.29+) |
| `git` | Submodule checkout | any |
| `python3` + `pip` | Model conversion (optional at inference time) | 3.9+ |
| `ffmpeg` | PPM → MP4 conversion (optional) | any |
| CUDA toolkit | GPU inference via CUDA (optional) | 11.8+ |

### Clone & initialise submodules

```bash
git clone https://github.com/audiohacking/ltx.cpp
cd ltx.cpp
git submodule update --init    # pulls the ggml submodule (~10 MB)
```

### Build configurations

All options are passed as `-D` flags to CMake:

```bash
mkdir build && cd build

# ── CPU only (default) ───────────────────────────────────────────────────────
cmake ..

# ── NVIDIA GPU (CUDA) ────────────────────────────────────────────────────────
cmake .. -DLTX_CUDA=ON

# ── AMD GPU (ROCm/HIP) ───────────────────────────────────────────────────────
cmake .. -DLTX_HIP=ON

# ── Apple Silicon / macOS (Metal) ────────────────────────────────────────────
cmake .. -DLTX_METAL=ON

# ── Vulkan ───────────────────────────────────────────────────────────────────
cmake .. -DLTX_VULKAN=ON

# ── Build ────────────────────────────────────────────────────────────────────
cmake --build . --config Release -j$(nproc)
```

The CMake options (`LTX_CUDA`, `LTX_HIP`, `LTX_METAL`, `LTX_VULKAN`) forward
to the corresponding `GGML_*` options in the ggml submodule — no extra wiring
is needed.

Output binaries appear in `build/`:
- `ltx-generate` — inference
- `ltx-quantize` — quantization utility

### Obtaining model files

**Option A – pre-quantised GGUF (recommended for first run)**

```bash
./models.sh             # downloads Q8_0 (~7 GB) into ./models/
./models.sh --quant Q4_K_M   # smaller, faster
```

**Option B – convert from safetensors**

```bash
pip install gguf safetensors transformers
./checkpoints.sh        # downloads raw HF checkpoints

python3 convert.py --model dit \
    --input  checkpoints/ltxv-2b-0.9.6-dev.safetensors \
    --output models/ltxv-2b-BF16.gguf

python3 convert.py --model vae \
    --input  checkpoints/ltxv-vae.safetensors \
    --output models/ltxv-vae-BF16.gguf

python3 convert.py --model t5 \
    --input  checkpoints/t5-xxl/ \
    --output models/t5-xxl-BF16.gguf

./quantize.sh Q8_0      # re-quantise BF16 → Q8_0
```

---

## 4. End-to-end data flow

### Text-to-video (T2V)

```
CLI args
  │
  ├─ --prompt  →  T5Encoder::encode_text()
  │                  tokenise → GGML graph → float[seq_len × 4096]
  │
  ├─ --dit / --vae / --t5  →  LtxGgufModel::open()
  │                              gguf_init_from_file() loads tensors into ggml_context
  │
  │  latent dims:  T_lat = (frames-1)/4 + 1
  │               H_lat = height / 8
  │               W_lat = width  / 8
  │
  ├─ LtxRng::fill()  →  random noise latent  [T_lat × H_lat × W_lat × 128]
  │
  └─ denoising loop  (steps times):
       │
       ├─ patchify()        [T,H,W,C] → [N_tok, patch_dim]    (patch_size=1×2×2)
       │
       ├─ LtxDiT::forward() [N_tok, Pd] + text_emb + timestep → velocity [N_tok, Pd]
       │    └─ GGML graph: patchify proj → N×(self-attn + cross-attn + SwiGLU FFN) → proj_out
       │
       ├─ (if CFG) second forward() with uncond_emb → apply_cfg()
       │
       ├─ unpatchify()      velocity [N_tok, Pd] → [T,H,W,C]
       │
       ├─ RFScheduler::euler_step()   x_t += dt * v
       │
       └─ (if I2V) frame conditioning blend  (see §7)
            │
            └─ after final step: hard-pin reference frame latents
  │
  └─ VaeDecoder::decode()  [T_lat, H_lat, W_lat, 128] → [T_vid, H_vid, W_vid, 3]
       │
       └─ write_video_frames()  →  output/frame_NNNN.ppm
```

### Image-to-video (I2V) additions

```
--start-frame / --end-frame  (PNG/JPG/BMP/TGA/PPM)
  │
  ├─ load_image()  →  VideoBuffer (stb_image, 8-bit RGB)
  │
  └─ VaeEncoder::encode_frame()
       ├─ resize_bilinear()  pixel [H×W×3] → latent spatial [H_lat×W_lat×3]
       ├─ normalise to [-1,1]
       └─ project 3-ch → 128-ch latent
            ├─ if conv_in_w present in GGUF: learned 1×1 conv projection
            └─ else: pseudo-encoding (channel tiling × 3.0 scale)

  → start_lat / end_lat  [H_lat × W_lat × 128]

  These latents are blended into the live denoising latent after each Euler step
  (see §7 for the full schedule).
```

---

## 5. Source file reference

### `ltx_common.hpp`

Shared include pulled by every other module.

| Symbol | Description |
|--------|-------------|
| `LTX_LOG / LTX_ERR / LTX_ABORT` | `fprintf(stderr,…)` logging macros |
| `LtxGgufModel` | Wrapper around `gguf_context` + `ggml_context`. Opened with `open(path)`, tensor lookup with `get_tensor(name)`, metadata with `kv_str/kv_i64/kv_u32/kv_f32` |
| `f32_data(t)` | Cast `ggml_tensor::data` to `float*` |
| `LtxRng` | `std::mt19937` + `std::normal_distribution<float>` seeded from `--seed` |
| `sigmoid / gelu` | Inline CPU helpers for small activations |
| `VideoBuffer` | `uint8_t` frame store `[F×H×W×3]`; `clamp_u8(float)` maps `[-1,1]→[0,255]` |
| `write_ppm / write_video_frames` | Binary P6 PPM output |
| `load_image(path)` | stb_image-backed loader; returns `VideoBuffer(0,0,0)` on failure |
| `resize_bilinear(src, …)` | In-place bilinear resize of uint8 RGB data |

**stb_image integration**: `STB_IMAGE_IMPLEMENTATION` is defined once inside
`ltx_common.hpp`.  Only the decoders actually used are compiled in
(`STBI_ONLY_PNG`, `STBI_ONLY_JPEG`, `STBI_ONLY_BMP`, `STBI_ONLY_TGA`,
`STBI_ONLY_PNM`).  Because `ltx_common.hpp` is included by exactly one
translation unit (`ltx-generate.cpp`), there is no ODR violation.

---

### `scheduler.hpp`

Implements the Rectified Flow Euler sampler.

```
RFScheduler(steps, shift, cfg)
  .timesteps()      → vector<float> of length steps+1, from 1.0 → 0.0
  ::euler_step()    → x_t += (t_next - t_cur) * v
  ::apply_cfg()     → v_out = v_uncond + scale * (v_cond - v_uncond)
```

The **flow-shift** rescales the linear schedule so that more steps are spent
near `t=0` (fine detail), which is important for the distilled LTX-Video model:

```
alpha = (steps - i) / steps        # linear 1→0
t     = alpha * shift / (1 + (shift-1) * alpha)
```

With `shift=3.0` (default), the schedule is compressed toward `t=0`.

---

### `t5_encoder.hpp`

Minimal T5-XXL encoder (encoder stack only — no decoder).

| Symbol | Description |
|--------|-------------|
| `T5Config` | `d_model=4096`, `num_heads=64`, `d_ff=10240`, `num_layers=24`, `vocab_size=32128` — read from GGUF KV at runtime |
| `T5Tokenizer` | Naïve whitespace + SentencePiece `▁`-prefix tokenizer loaded from `tokenizer.ggml.tokens` array in the GGUF; unk fallback is per-character |
| `T5Encoder::load()` | Reads weights named `encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight` etc. |
| `T5Encoder::encode(ids)` | Builds a GGML graph: embedding lookup → N × (RMSNorm + self-attn + RMSNorm + SwiGLU FFN) → final RMSNorm. Returns `float[S × d_model]` |
| `T5Encoder::encode_text(str)` | Tokenises then calls `encode()` |

**Known limitation**: the tokenizer is naive (whitespace-split + character
fallback).  Rare or multi-byte tokens may be mishandled.  A proper
SentencePiece unigram model should replace it for production use.

---

### `video_vae.hpp`

#### `VaeDecoder`

Weights layout expected in the GGUF (prefix `vae.decoder.*`):
- `conv_in.weight / .bias` — post-quant conv (latent_channels → mid_channels)
- `mid_block.resnets.{0,1}.*` — two residual blocks
- `mid_block.attentions.0.*` — self-attention (simplified)
- `up_blocks.{0..3}.resnets.{0,1}.*` — four upsample stages
- `up_blocks.{b}.upsamplers.0.conv.*` — spatial upsamplers
- `conv_norm_out.*` / `conv_out.*` — final group-norm + output conv

`decode(latents, T_lat, H_lat, W_lat)` runs a simplified per-frame 2-D decode
with nearest-neighbour temporal upsampling.  Full causal 3-D conv decode is a
planned improvement (see §10).

#### `VaeEncoder`

Added for I2V conditioning.  Only `vae.encoder.conv_in.weight/bias` are
currently loaded.  When present, a 1×1 learned projection is used; otherwise a
pseudo-encoding tiles normalised RGB across the 128 latent channels.

---

### `ltx_dit.hpp`

The main diffusion transformer.

**Config** (read from GGUF KV):
| Key | Default |
|-----|---------|
| `ltxv.hidden_size` | 2048 |
| `ltxv.num_hidden_layers` | 28 |
| `ltxv.num_attention_heads` | 32 |
| `ltxv.in_channels` | 128 |

**Tensor naming** (primary, from Lightricks diffusers export):
```
model.diffusion_model.patchify_proj.{weight,bias}
model.diffusion_model.adaln_single.emb.timestep_embedder.linear_{1,2}.{weight,bias}
model.diffusion_model.adaln_single.linear.{weight,bias}
model.diffusion_model.caption_projection.{weight,bias}
model.diffusion_model.transformer_blocks.{i}.attn1.to_{q,k,v,out.0}.{weight,bias}
model.diffusion_model.transformer_blocks.{i}.attn2.to_{q,k,v,out.0}.{weight,bias}
model.diffusion_model.transformer_blocks.{i}.ff.net.{0.proj,2}.{weight,bias}
model.diffusion_model.proj_out.{weight,bias}
model.diffusion_model.norm_out.linear.{weight,bias}
```
Fallback names with prefix `dit.*` are also tried.

**Forward pass** (per call to `LtxDiT::forward()`):
1. Sinusoidal timestep embedding → MLP → `hidden_size` vector
2. AdaLN-single linear → `6 × hidden_size` (scale/shift params; currently
   stored but not yet fully applied per-block — see §10)
3. Patchify projection: `[N_tok, patch_dim]` → `[N_tok, hidden_size]`
4. Caption projection: `[S, 4096]` → `[S, hidden_size]`
5. N × transformer blocks:
   - Pre-norm (RMSNorm) + self-attention (multi-head, scaled dot-product)
   - Pre-norm + cross-attention (latent queries, text keys/values)
   - Pre-norm + SwiGLU FFN (gate×up → down)
6. Final RMSNorm + output projection → `[N_tok, patch_dim]`
7. GGML graph execution (`ggml_graph_compute_with_ctx`)

**Note on scratch memory**: each forward call allocates 1 GB of scratch via
`ggml_init`.  This is safe for a single call but not ideal for batching.  A
planned improvement is to pre-allocate a persistent scratch context.

---

### `ltx-generate.cpp`

Orchestrates the full inference pipeline.

**`Args` struct** — all CLI parameters with defaults:

| Field | Flag | Default |
|-------|------|---------|
| `dit_path` | `--dit` | required |
| `vae_path` | `--vae` | required |
| `t5_path` | `--t5` | required |
| `prompt` | `--prompt` / `-p` | `"A beautiful scenic landscape…"` |
| `negative_prompt` | `--neg` / `-n` | `""` |
| `frames` | `--frames` | 25 |
| `height` | `--height` | 480 |
| `width` | `--width` | 704 |
| `steps` | `--steps` | 40 |
| `cfg_scale` | `--cfg` | 3.0 |
| `shift` | `--shift` | 3.0 |
| `seed` | `--seed` | 42 |
| `out_prefix` | `--out` | `"output/frame"` |
| `start_frame_path` | `--start-frame` | `""` (disabled) |
| `end_frame_path` | `--end-frame` | `""` (disabled) |
| `frame_strength` | `--frame-strength` | 1.0 |
| `threads` | `--threads` | 4 |
| `verbose` | `-v` | false |

**Output**: frames are written as `{out_prefix}_{NNNN}.ppm`.  The output
directory is created automatically (including intermediate directories).

---

### `ltx-quantize.cpp`

Standalone quantizer that reads a BF16/F32 GGUF and writes a new GGUF with
all 2-D+ weight tensors quantised to the requested type.

Rules:
- **1-D tensors** (biases, norms) → kept as F32
- **Embedding weights** → kept as F32
- Everything else → quantised to `target_type`

All GGUF KV metadata is copied verbatim.  String arrays (e.g. the tokenizer
vocabulary) are not currently copied — this is a known limitation (see §10).

Supported quant types: `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, `F32`, `F16`.

---

### `convert.py`

Python script that reads HuggingFace safetensors checkpoints and writes GGUF
files that ltx.cpp can load.

| Converter | `--model` | Input | Output arch |
|-----------|-----------|-------|-------------|
| `convert_dit()` | `dit` | single `.safetensors` | `ltxv` |
| `convert_vae()` | `vae` | single `.safetensors` | `ltxv-vae` |
| `convert_t5()` | `t5` | directory of shards | `t5` |

The DiT converter passes tensor names through unchanged from the safetensors
file.  The VAE converter prefixes all names with `"vae."` if not already
present.  The T5 converter remaps `encoder.embed_tokens.weight` →
`token_emb.weight` and skips decoder tensors.

For T5, the HF tokenizer vocabulary can be embedded into the GGUF via
`--tokenizer <path>`, which runs `transformers.T5Tokenizer` and writes
`tokenizer.ggml.tokens` as a string array.

---

## 6. GGUF model format conventions

### DiT GGUF

Architecture string: `"ltxv"`

| KV key | Type | Description |
|--------|------|-------------|
| `general.architecture` | string | `"ltxv"` |
| `ltxv.hidden_size` | uint32 | transformer hidden dim |
| `ltxv.num_hidden_layers` | uint32 | number of transformer blocks |
| `ltxv.num_attention_heads` | uint32 | attention heads |
| `ltxv.in_channels` | uint32 | VAE latent channels (128) |
| `ltxv.cross_attention_dim` | uint32 | text encoder dim (4096) |
| `ltxv.patch_size` | uint32 | spatial patch size (2) |

### VAE GGUF

Architecture string: `"ltxv-vae"`

| KV key | Type | Description |
|--------|------|-------------|
| `general.architecture` | string | `"ltxv-vae"` |
| `vae.latent_channels` | uint32 | 128 |
| `vae.spatial_scale` | uint32 | 8 (8× spatial downsampling) |
| `vae.temporal_scale` | uint32 | 4 (4× temporal downsampling) |

### T5 GGUF

Architecture string: `"t5"`

| KV key | Type | Description |
|--------|------|-------------|
| `general.architecture` | string | `"t5"` |
| `t5.block_count` | uint32 | encoder layers (24 for XXL) |
| `t5.embedding_length` | uint32 | d\_model (4096) |
| `t5.feed_forward_length` | uint32 | d\_ff (10240) |
| `t5.attention.head_count` | uint32 | num\_heads (64) |
| `t5.vocab_size` | uint32 | 32128 |
| `tokenizer.ggml.tokens` | string[] | SentencePiece vocabulary |

---

## 7. Image-to-video (I2V) design

The I2V implementation does not modify the DiT architecture.  Instead it
works by conditioning the *latent* directly at the boundary frames before and
after each denoising step.

### VaeEncoder

`VaeEncoder::encode_frame(img_u8, H_pix, W_pix, H_lat, W_lat)`:

1. **Bilinear resize** the image to `[H_lat, W_lat, 3]` using
   `resize_bilinear()` (in `ltx_common.hpp`).
2. **Normalise** pixels `uint8 [0,255]` → `float [-1,1]`:
   `norm = pixel / 127.5 - 1.0`
3. **Project** 3 channels → `C=128` latent channels:
   - **With encoder weights** (`vae.encoder.conv_in.weight` in the GGUF):  
     Apply the learned 1×1 convolution as a `[C, 3]` matrix multiply.
   - **Without encoder weights** (pseudo-encoding):  
     Assign each latent channel to one of the three colour channels
     (R/G/B, `C/3` channels each), scaled by 3.0 to match typical latent
     statistics.

### Frame-conditioning schedule

After every Euler denoising step the first and/or last temporal latent frames
are blended toward the encoded reference:

```
blend = clamp(frame_strength * (1 - t_next), 0, 1)
lat[T=0]    = lat[T=0]    * (1 - blend) + start_lat * blend
lat[T=T-1]  = lat[T=T-1]  * (1 - blend) + end_lat   * blend
```

- At the start of denoising (`t=1`), `blend=0` — the reference is not imposed
  yet so the DiT can form global structure freely.
- As denoising progresses toward `t=0`, `blend` increases linearly to
  `frame_strength`, pulling the frame latents toward the reference.

### Hard-pinning at t=0

When `frame_strength >= 1.0` (default), after all denoising steps finish the
reference latent is copied verbatim into the output latent buffer:

```cpp
memcpy(latents.data(), start_lat.data(), frame_lat_size * sizeof(float));
```

This guarantees the decoded output frame exactly matches the reference image
appearance, regardless of any residual denoising drift.

---

## 8. Key algorithms and design decisions

### Rectified Flow (RF) scheduling

LTX-Video was trained with Rectified Flow.  The forward process is:

```
x_t = (1 - t) * x_0 + t * noise    t ∈ [0, 1]
```

The model predicts the velocity `v = dx/dt = noise - x_0`.  The Euler ODE
solver steps backward from `t=1` to `t=0`:

```
x_{t-dt} = x_t + dt * v_θ(x_t, t)     (dt < 0)
```

### Classifier-free guidance

With `--cfg > 1.0`, the DiT is called twice per step:
- Once with the text embedding (`v_cond`)
- Once with the empty-string embedding (`v_uncond`)

The guided velocity is:

```
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

The unconditional embedding is computed by encoding the `--neg` prompt
(default: empty string).

### Patchify / unpatchify

The DiT operates on *tokens*, not on the raw latent volume.  The latent
`[T_lat, H_lat, W_lat, C]` is chunked into non-overlapping patches of size
`(pt=1, ph=2, pw=2)` along the temporal, height, and width dimensions:

```
patch_dim = pt * ph * pw * C = 1 * 2 * 2 * 128 = 512
N_tok     = (T_lat/pt) * (H_lat/ph) * (W_lat/pw)
```

`patchify()` and `unpatchify()` are helper functions called from
`ltx-generate.cpp`.  Both are pure memory rearrangements with no arithmetic.

### Latent dimension formulas

| Video dimension | Latent dimension | Formula |
|-----------------|-----------------|---------|
| `frames` | `T_lat` | `(frames − 1) / 4 + 1` |
| `height` | `H_lat` | `height / 8` |
| `width` | `W_lat` | `width / 8` |
| `T_vid` (decoded) | — | `(T_lat − 1) * 4 + 1` |

The temporal scale is 4× and the spatial scale is 8×.  These values are read
from the VAE GGUF (`vae.temporal_scale`, `vae.spatial_scale`).

### Tokenizer

The T5 tokenizer implements the **SentencePiece unigram** algorithm in pure
C++ with no external library dependency.  The vocabulary and optional
log-probability scores are loaded from the GGUF metadata at model-load time:

| GGUF key | Type | Description |
|----------|------|-------------|
| `tokenizer.ggml.tokens` | string[] | id → piece (UTF-8, ▁-prefixed) |
| `tokenizer.ggml.scores` | float32[] | id → unigram log-probability (optional) |

**Preprocessing** (`T5Tokenizer::preprocess`):
1. Collapse runs of whitespace to a single space; strip leading/trailing.
2. Prepend `▁` (U+2581) to the beginning; replace each remaining space with `▁`.

**Segmentation** — two modes depending on whether scores are in the GGUF:

| Mode | Condition | Algorithm |
|------|-----------|-----------|
| Viterbi | `tokenizer.ggml.scores` present | DP over byte positions; maximises sum of log-probs; O(n × max_piece_len) |
| Greedy | scores absent | Longest-match scan from left; O(n × max_piece_len) |

In both modes an **unk fallback** advances one full UTF-8 character (not one
byte) when no vocabulary piece covers the current position, preventing split
multi-byte sequences from producing garbage tokens.

Scores are written by `convert.py --tokenizer` (via
`tok.sp_model.GetScore(i)`) and preserved through quantization by
`ltx-quantize` (via `gguf_set_kv`).

---

## 9. Adding a new backend (GPU/Metal/Vulkan)

GGML abstracts hardware via *backends*.  Adding GPU support requires only a
CMake flag:

```bash
cmake .. -DLTX_CUDA=ON      # NVIDIA
cmake .. -DLTX_HIP=ON       # AMD ROCm
cmake .. -DLTX_METAL=ON     # Apple Metal
cmake .. -DLTX_VULKAN=ON    # Vulkan
```

From the C++ side there is nothing more to do — GGML automatically selects the
best available backend at runtime.  If you want to *explicitly* target a
backend, use `ggml_backend_*` APIs from `ggml-backend.h`.

The main performance bottleneck is the DiT `forward()` call, which rebuilds a
`ggml_cgraph` on every step.  A future improvement is to build the graph once
and re-use it across steps by parameterising the timestep embedding.

---

## 10. Known limitations and open tasks

These are the main areas where the implementation is deliberately simplified
and where contributions are most welcome.

| # | Area | Current state | What needs doing |
|---|------|---------------|-----------------|
| 1 | **VAE decoder** | Per-frame 2-D decode + nearest-neighbour temporal upsampling | Implement full causal 3-D conv decode using `ggml_conv_1d / 2d`; wire temporal upsampling via transposed conv |
| 2 | **VAE encoder** | Only the first `conv_in` layer is used; pseudo-encoding fallback | Implement full encoder stack for accurate I2V latent inversion |
| 3 | **AdaLN-single** | Timestep embedding is computed but per-block scale/shift is not fully applied | Apply `ada_params` chunks as scale/shift in each block's norms |
| 4 | **3-D RoPE** | Positional embeddings are not yet applied | Add rotary embeddings along (t, h, w) axes to Q and K tensors |
| 5 | **T5 tokenizer** | ~~Whitespace-split + per-char fallback~~ **Fixed**: full SentencePiece unigram Viterbi DP (when scores in GGUF) or greedy longest-match | — |
| 6 | **`ltx-quantize` metadata** | ~~String arrays (tokenizer vocab) are skipped during quantization~~ **Fixed**: `gguf_set_kv` copies all KV pairs including arrays | — |
| 7 | **Persistent scratch** | DiT allocates 1 GB of ggml scratch per forward call | Pre-allocate a single scratch context and reset between calls |
| 8 | **Batch size > 1** | Only batch=1 is implemented | Add batch dimension to enable parallel generation |
| 9 | **CFG single-pass** | CFG requires two full forward passes | Implement single-pass CFG by duplicating the batch |
| 10 | **Threading** | `--threads` is parsed but not passed to `ggml_graph_compute_with_ctx` | Wire the thread count through to `ggml_graph_compute_with_ctx(ctx, gf, n_threads)` |
| 11 | **Output formats** | Only binary PPM (P6) output | Add JPEG/PNG output via stb_image_write or a similar library |
| 12 | **Windows `_mkdir`** | Only one level of directory is created on Windows | Implement recursive mkdir for Windows |

---

## 11. Coding conventions

- **Language**: C++17 throughout; no exceptions (use return codes).
- **Headers only**: all modules live in `src/*.hpp`.  Only the two `main()`
  translation units are `.cpp` files.
- **No STL containers with `new`/`delete`**: use `std::vector<float>` for all
  large buffers; GGML tensors are owned by the `ggml_context` they were created
  in.
- **Logging**: use `LTX_LOG(fmt, …)` for info and `LTX_ERR(fmt, …)` for
  errors.  Both write to `stderr`.  Progress during the denoising loop uses
  `\r` overwrite for a clean single-line display.
- **Error handling**: functions return `bool` or an empty/zero-frames
  `VideoBuffer` on failure.  `LTX_ABORT` for truly unrecoverable conditions.
- **Naming**: `snake_case` for variables and functions; `PascalCase` for
  structs; `UPPER_CASE` for macros.
- **Comments**: section headers use the `// ── … ───` style; inline
  comments explain *why*, not *what*.
- **Third-party code**: vendored in `src/` (currently only `stb_image.h`).
  Keep separate from project code; suppress vendor warnings at the CMake level,
  not inside the header.
- **No `#pragma once` in `.cpp` files**: only in `*.hpp`.

---

## 12. Testing

There is no formal test suite yet.  Validation is currently done by:

1. **Build smoke test** — `cmake --build . -j$(nproc)` must produce zero errors
   and zero warnings (except those from vendored third-party headers).
2. **Argument parsing** — run `./build/ltx-generate --help` and verify the
   usage text is correct.
3. **Image loading** — write a short C++ snippet that calls `load_image()` with
   a PNG, a JPEG, a PPM, and a missing file, and assert the results.
4. **End-to-end generation** — run `ltx-generate` with real model files and
   check that the output PPM frames are non-zero and have the expected
   dimensions.

**Planned**: a `tests/` directory with:
- Unit tests for `RFScheduler::timesteps()` (known values)
- Unit tests for `patchify` / `unpatchify` round-trip
- Unit tests for `resize_bilinear`
- An integration test that runs `ltx-generate` with tiny synthetic GGUF stubs

---

## 13. Contributing

1. **Fork** the repository and create a branch from `main`.
2. **Read §10** to find where help is most needed.
3. **Keep PRs focused** — one feature or fix per PR.
4. **Match the style** described in §11.
5. **Document** any new CLI flag in both `print_usage()` (in
   `ltx-generate.cpp`) and `README.md`.
6. **Update this file** (`DEV.md`) if you add a new module, change the GGUF
   schema, or significantly alter the data flow.
7. **No model weights** should ever be committed to the repo.

For questions, open a GitHub Discussion or issue in the
[audiohacking/ltx.cpp](https://github.com/audiohacking/ltx.cpp) repository.
