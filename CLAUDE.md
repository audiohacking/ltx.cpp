# ltx.cpp

C++ inference engine for LTX-Video (LTX 2.3) — text-to-video and image-to-video generation using GGML backends (Metal, CUDA, CPU).

## Build

```bash
# Debug build (used during development)
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --target ltx-generate -j

# Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ltx-generate -j
```

Backend flags (default: Metal on Apple, CPU elsewhere):
```bash
cmake -B build -DLTX_CUDA=ON    # CUDA
cmake -B build -DLTX_VULKAN=ON  # Vulkan
cmake -B build -DLTX_HIP=ON     # ROCm/AMD
```

## Models

Download with `./models.sh` (requires `huggingface-cli`):
```bash
./models.sh              # DiT Q4_K_M + T5 Q8_0 + VAE + extras
./models.sh --minimal    # DiT + T5 + VAE only
./models.sh --quant Q8_0 # different DiT quant
```

Models land flat under `models/`. Key files:
- `models/ltx-2.3-22b-dev-Q4_K_M.gguf` — DiT weights
- `models/ltx-2.3-22b-dev_video_vae.safetensors` — VAE
- `models/t5-v1_1-xxl-encoder-Q8_0.gguf` — T5 text encoder

## Run

```bash
build/ltx-generate \
  --dit  models/ltx-2.3-22b-dev-Q4_K_M.gguf \
  --vae  models/ltx-2.3-22b-dev_video_vae.safetensors \
  --t5   models/t5-v1_1-xxl-encoder-Q8_0.gguf \
  --prompt "A cat on a bench" \
  --frames 25 --height 480 --width 704 \
  --steps 20 --out output/frame
```

Useful flags:
- `-v` — verbose per-step logging
- `--perf` — print CPU%/RSS/free-RAM/GPU-MB to stderr every 10 s
- `--start-frame img.png` — image-to-video (I2V)
- `--end-frame img.png` — keyframe interpolation
- `--seed N`, `--cfg F`, `--shift F`, `--threads N`

## Test

Quick smoke test (GPU migration, 2 steps, tiny resolution):
```bash
BIN=build_debug/ltx-generate bash scripts/test-gpu-migration.sh
```

## Source layout

| File | Purpose |
|------|---------|
| `src/ltx-generate.cpp` | Main binary: arg parsing, model loading, denoising loop |
| `src/ltx_dit.hpp` | DiT transformer (forward pass, block loop, Metal/CPU paths) |
| `src/video_vae.hpp` | VAE encoder/decoder (safetensors) |
| `src/t5_encoder.hpp` | T5-XXL text encoder (GGUF) |
| `src/scheduler.hpp` | RF flow scheduler (timesteps, Euler step, CFG) |
| `src/ltx_perf.hpp` | Background perf monitor thread (CPU/RAM stats) |
| `src/ltx_common.hpp` | Shared macros (`LTX_LOG`, `LTX_ERR`), GGML helpers |
| `src/safetensors_loader.cpp` | safetensors file loader |

## Architecture notes

- **Backend**: `ggml_backend_init_best()` auto-selects Metal/CUDA/etc; falls back to CPU. DiT weights are migrated to the backend via `ltx_backend_migrate_ctx`.
- **DiT forward**: single full graph per step, dispatched via `ggml_backend_sched` (Metal + CPU). Inputs: latents, text_emb, timestep; output: velocity.
- **Scheduler reserve**: Before the denoise loop, `reserve_sched(sched, n_tok, seq_len)` pre-allocates backend buffers from a measure graph so each step reuses them (avoids realloc overhead).
- **CFG**: two forward passes per step (cond + uncond) when `cfg_scale > 1.0`.
- **LTX_MIGRATE_MAX_TENSOR_MB**: env var to override per-tensor GPU migration cap (default 6 GB). Set to `0` to attempt full migration.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `LTX_MIGRATE_MAX_TENSOR_MB` | `6144` | Max single-tensor size for GPU migration |
