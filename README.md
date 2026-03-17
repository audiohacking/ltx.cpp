# ltx.cpp

Portable C++17 inference of **LTX-Video** (Lightricks) using
[GGML](https://github.com/ggml-org/ggml) / GGUF.  
Text-to-video generation runs on CPU, CUDA, ROCm, Metal, and Vulkan — no
Python required at inference time.

Inspired by [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp).

---

## Features

- **Text-to-video** inference with the LTX-Video 2.3 DiT
- **Image-to-video (I2V)** — animate a reference image (`--start-frame`)
- **Keyframe interpolation** — provide both start and end frames to interpolate between them (`--start-frame` + `--end-frame`)
- Quantised GGUF weights (Q4\_K\_M → Q8\_0 → BF16)
- Classifier-free guidance + flow-shift Euler sampler
- PPM frame output (pipe to ffmpeg for MP4)
- Single `ltx-generate` binary — no Python at runtime

---

## Build

```bash
git submodule update --init          # pull ggml

mkdir build && cd build

# CPU only
cmake ..

# With NVIDIA GPU (CUDA)
cmake .. -DLTX_CUDA=ON

# With AMD GPU (ROCm)
cmake .. -DLTX_HIP=ON

# macOS (Metal)
cmake .. -DLTX_METAL=ON

# Vulkan
cmake .. -DLTX_VULKAN=ON

cmake --build . --config Release -j$(nproc)
```

Builds two binaries:

| Binary          | Purpose                            |
|-----------------|------------------------------------|
| `ltx-generate`  | Text-to-video inference            |
| `ltx-quantize`  | Re-quantize GGUF files             |

---

## Models

### Option A – Download pre-quantised GGUFs (recommended)

```bash
pip install huggingface_hub          # for hf_hub_download

./models.sh                          # Q8_0 (~7 GB total)
./models.sh --quant Q4_K_M           # smaller, faster
./models.sh --all                    # every quant
```

Downloads three GGUF files into `models/`:

| File                              | Contents              | Size (Q8\_0) |
|-----------------------------------|-----------------------|--------------|
| `ltxv-2b-*-Q8_0.gguf`            | Video DiT (2B params) | ~2.1 GB      |
| `ltxv-vae-Q8_0.gguf`             | CausalVideoVAE        | ~400 MB      |
| `t5-xxl-Q8_0.gguf`               | T5-XXL text encoder   | ~4.6 GB      |

### Option B – Convert from safetensors

```bash
pip install gguf safetensors transformers

./checkpoints.sh                     # download raw HF checkpoints

python3 convert.py --model dit \
    --input  checkpoints/ltxv-2b-0.9.6-dev.safetensors \
    --output models/ltxv-2b-BF16.gguf

python3 convert.py --model vae \
    --input  checkpoints/ltxv-vae.safetensors \
    --output models/ltxv-vae-BF16.gguf

python3 convert.py --model t5 \
    --input  checkpoints/t5-xxl/ \
    --output models/t5-xxl-BF16.gguf

./quantize.sh Q8_0                   # BF16 → Q8_0
```

---

## Quick Start

### Text-to-video

```bash
mkdir -p output

./build/ltx-generate \
    --dit    models/ltxv-2b-0.9.6-dev-Q8_0.gguf \
    --vae    models/ltxv-vae-Q8_0.gguf \
    --t5     models/t5-xxl-Q8_0.gguf \
    --prompt "A peaceful waterfall in a lush forest, cinematic, 4K" \
    --frames 25 \
    --height 480 --width 704 \
    --steps  40  --cfg 3.0  --shift 3.0 \
    --seed   42  --out output/frame
```

### Image-to-video (I2V) — animate a reference image

Provide a **PNG, JPG, BMP, TGA, or PPM** image as `--start-frame`.  The video will
start from (and be strongly conditioned on) that image and animate from there
based on the prompt.  No conversion step is needed — standard image formats are
supported natively.

```bash
./build/ltx-generate \
    --dit    models/ltxv-2b-0.9.6-dev-Q8_0.gguf \
    --vae    models/ltxv-vae-Q8_0.gguf \
    --t5     models/t5-xxl-Q8_0.gguf \
    --prompt "Camera slowly pans right, birds fly overhead" \
    --start-frame photo.jpg \
    --frames 25 --height 480 --width 704 \
    --steps 40 --cfg 3.0 --out output/frame
```

### Keyframe interpolation — animate between two images

Provide both `--start-frame` and `--end-frame` to generate a video that
transitions smoothly from the first image to the last.

```bash
./build/ltx-generate \
    --dit    models/ltxv-2b-0.9.6-dev-Q8_0.gguf \
    --vae    models/ltxv-vae-Q8_0.gguf \
    --t5     models/t5-xxl-Q8_0.gguf \
    --prompt "A serene forest scene, gentle breeze, cinematic" \
    --start-frame beginning.png \
    --end-frame   ending.png \
    --frames 33 --height 480 --width 704 \
    --steps 40 --cfg 3.0 --out output/frame
```

Use `--frame-strength` (0..1) to control how strongly the reference frame(s)
constrain the generation.  Default is `1.0` (fully pinned).  Lower values
give the model more creative freedom around the reference.

**Supported input image formats:** PNG, JPEG/JPG, BMP, TGA, PPM/PGM
(powered by stb_image — no additional libraries required)

Convert the PPM output frames to MP4:

```bash
ffmpeg -framerate 24 -i output/frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

## Command-Line Reference

```
ltx-generate [options]

Required:
  --dit    <path>   DiT model GGUF file
  --vae    <path>   VAE decoder GGUF file
  --t5     <path>   T5 text encoder GGUF file

Generation:
  --prompt  <text>  Positive text prompt
  --neg     <text>  Negative prompt (default: empty)
  --frames  <N>     Number of output video frames   (default: 25)
  --height  <H>     Frame height in pixels           (default: 480)
  --width   <W>     Frame width in pixels            (default: 704)
  --steps   <N>     Denoising steps                  (default: 40)
  --cfg     <f>     Classifier-free guidance scale   (default: 3.0)
  --shift   <f>     Flow-shift parameter             (default: 3.0)
  --seed    <N>     RNG seed                         (default: 42)
  --out     <pfx>   Output frame file prefix         (default: output/frame)

Image-to-video (I2V) conditioning:
  --start-frame  <path>  PNG/JPG/BMP/TGA/PPM image: animate from this reference frame
  --end-frame    <path>  PNG/JPG/BMP/TGA/PPM image: end at this frame (keyframe interp)
  --frame-strength <f>   Conditioning strength [0..1]  (default: 1.0)
                          1.0 = fully pin frame, 0.5 = soft guidance

Performance:
  --threads <N>     CPU worker threads               (default: 4)
  -v                Verbose logging per step
```

---

## Architecture

### Text-to-video

```
Text prompt
    │
    ▼
T5-XXL encoder          (GGUF: t5-xxl-*.gguf)
    │  [seq_len × 4096 embeddings]
    │
    ▼
LTX-Video DiT           (GGUF: ltxv-2b-*.gguf)
  ┌─────────────────────────────────────────┐
  │  Random noise latent                    │
  │  [T_lat × H_lat × W_lat × 128]         │
  │       │                                 │
  │  ┌────┴──────────────────────────┐      │
  │  │  N × Transformer block        │      │
  │  │    self-attn  (3D RoPE)       │      │
  │  │    cross-attn (text cond.)    │      │
  │  │    FFN (SwiGLU)               │      │
  │  │    AdaLN (timestep cond.)     │      │
  │  └────────────────────────────┬──┘      │
  │  Euler ODE (flow matching)    │         │
  └───────────────────────────────┘         │
    │  [T_lat × H_lat × W_lat × 128]        │
    ▼
CausalVideoVAE decoder  (GGUF: ltxv-vae-*.gguf)
    │  [T_vid × H_vid × W_vid × 3] pixels
    ▼
PPM frames  →  ffmpeg  →  MP4
```

### Image-to-video (I2V) / Keyframe interpolation

```
Reference image(s) (PPM)
    │
    ▼
VaeEncoder.encode_frame()     pixel [H×W×3] → latent [H_lat×W_lat×128]
    │  start_lat / end_lat
    │
    ├──────────────────────────────────────────────────┐
    ▼                                                  ▼
Random noise latent                        Frame conditioning
[T_lat × H_lat × W_lat × 128]             per denoising step:
    │                                        lat[T=0]  ← blend(start_lat, t)
    │  Denoising loop (same as T2V)          lat[T=-1] ← blend(end_lat,   t)
    │        +
    │  frame-pinning after each Euler step
    ▼
VAE decode + PPM output
```

The conditioning blend weight increases as the timestep approaches 0
(clean signal), so early steps use mostly noise for global structure while
later steps are progressively more pinned to the reference image(s).

| Dimension       | Formula                            |
|-----------------|------------------------------------|
| T\_lat          | (frames − 1) ÷ 4 + 1              |
| H\_lat          | height ÷ 8                         |
| W\_lat          | width ÷ 8                          |
| T\_vid          | (T\_lat − 1) × 4 + 1              |

---

## References

- [LTX-Video (Lightricks)](https://github.com/Lightricks/LTX-Video)
- [Unsloth LTX-2.3 GGUF models](https://huggingface.co/unsloth/LTX-2.3-GGUF)
- [GGML](https://github.com/ggml-org/ggml)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
