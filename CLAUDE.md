# ltx.cpp

C++ inference engine for LTX-Video (LTX 2.3) — text-to-video, image-to-video, and **audio-video (AV)** generation using GGML backends (Metal, CUDA, CPU).

**Branch `audio-video`**: same DiT sees concatenated video+audio latent; one denoise loop; output is video frames + WAV. See `docs/AV_PIPELINE.md` and README “Audio-video (AV)” section.

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

Download with `./models.sh` (requires `curl` or `wget`):
```bash
./models.sh              # Dev DiT (default) + T5 + VAE + extras
./models.sh --distilled  # Distilled DiT (few-step 4–8, CFG=1) from same repo
./models.sh --minimal    # DiT + T5 + VAE only
./models.sh --quant Q8_0 # different DiT quant
```

Models land flat under `models/`. Key files:
- `models/ltx-2.3-22b-dev-Q4_K_M.gguf` — Dev DiT (default)
- `models/ltx-2.3-22b-distilled-Q4_K_M.gguf` — Distilled DiT (with `--distilled`)
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

**Audio-video (AV):** add `--av` and optionally `--out-wav path.wav` to get video frames + WAV from the same run. Mux with ffmpeg: `ffmpeg -framerate 24 -i out_%04d.ppm -i out.wav -c:v libx264 -c:a aac -shortest out.mp4`.

Useful flags:
- `-v` — verbose per-step logging
- `--perf` — print CPU%/RSS/free-RAM/GPU-MB to stderr every 10 s
- `--av` — enable audio+video path (concat latent → DiT → split → decode both)
- `--audio-vae path` — optional; for full audio VAE decoder when implemented
- `--out-wav path` — WAV output when `--av` (default: `<out prefix>.wav`)
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
| `src/ltx-generate.cpp` | Main binary: arg parsing, model loading, denoising loop; AV path (concat/split, WAV output) |
| `src/ltx_dit.hpp` | DiT transformer (forward pass, block loop); `patchify_audio` / `unpatchify_audio` for AV |
| `src/video_vae.hpp` | VAE encoder/decoder (safetensors) |
| `src/t5_encoder.hpp` | T5-XXL text encoder (GGUF) |
| `src/scheduler.hpp` | RF flow scheduler (timesteps, Euler step, CFG) |
| `src/ltx_perf.hpp` | Background perf monitor thread (CPU/RAM stats) |
| `src/ltx_common.hpp` | Shared macros (`LTX_LOG`, `LTX_ERR`), GGML helpers |
| `src/safetensors_loader.cpp` | safetensors file loader |
| `docs/AV_PIPELINE.md` | AV pipeline design (token concat, shapes, CLI) |

## Architecture notes

- **Backend**: `ggml_backend_init_best()` auto-selects Metal/CUDA/etc; falls back to CPU. DiT weights are migrated to the backend via `ltx_backend_migrate_ctx`.
- **DiT forward**: chunked execution — one transformer block at a time, same scratch buffer reused. Metal path uses `no_alloc` + backend buffers (scratch is unused). CPU path bump-allocates from scratch (~64 × n_tok × hidden_size × f32).
- **Scratch sizing**: computed from actual `n_tok` and `hidden_size` after model load; 1 byte on GPU path, capped at 32 GB on CPU.
- **CFG**: two forward passes per step (cond + uncond) when `cfg_scale > 1.0`.
- **LTX_MIGRATE_MAX_TENSOR_MB**: env var to override per-tensor GPU migration cap (default 6 GB). Set to `0` to attempt full migration.

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `LTX_MIGRATE_MAX_TENSOR_MB` | `6144` | Max single-tensor size for GPU migration |

## Branch: audio-video

- **AV path**: with `--av`, video and audio latents are patchified, concatenated (video then audio tokens), passed through one DiT forward, then split; Euler step on both; video decoded with existing VAE, audio turned into WAV via a latent→waveform fallback.
- **Full audio VAE** (safetensors decoder) is not yet implemented; audio quality uses the fallback. See `docs/AV_PIPELINE.md` and `DEV.md` §5.
