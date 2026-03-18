# LTX 2.3 ComfyUI workflow reference (for ltx.cpp)

Reference only — no Python. This describes the working ComfyUI setup so ltx.cpp can stay aligned with the same model and behavior.

## LTX-2.3-GGUF (models we use)

We use the [unsloth/LTX-2.3-GGUF](https://huggingface.co/unsloth/LTX-2.3-GGUF) quantized DiT and VAEs. Unsloth Dynamic 2.0 upcasts important layers; tooling from ComfyUI-GGUF (city96).

- **Dev**: full 22B DiT, needs ≥20 steps, better quality. Use for main generation.
- **Distilled**: few-step (4–8), CFG=1; useful as draft or refine (e.g. LoRA on top of dev).

Base model: [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3). LTX-2.3 is a DiT-based **audio-video** foundation model (synchronized video + audio in one model).

## Model files (what the workflow uses)

| Role | File | Format | Source |
|------|------|--------|--------|
| DiT (UNet) | `ltx-2.3-22b-dev-Q4_K_M.gguf` | GGUF | unsloth/LTX-2.3-GGUF |
| Video VAE | `ltx-2.3-22b-dev_video_vae.safetensors` | safetensors | unsloth/LTX-2.3-GGUF vae/ |
| Audio VAE | `ltx-2.3-22b-dev_audio_vae.safetensors` | safetensors | unsloth/LTX-2.3-GGUF vae/ |
| Text encoder | `gemma-3-12b-it-qat-UD-Q4_K_XL.gguf` | GGUF | unsloth/gemma-3-12b-it-qat-GGUF |
| Embedding connectors | `ltx-2.3-22b-dev_embeddings_connectors.safetensors` | safetensors | unsloth/LTX-2.3-GGUF text_encoders/ |
| Optional LoRA | `ltx-2.3-22b-distilled-lora-384.safetensors` | safetensors | Lightricks/LTX-2.3 |
| Optional upscaler | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | safetensors | Lightricks/LTX-2.3 |

ComfyUI loads the DiT via **UnetLoaderGGUF** and text via **DualCLIPLoaderGGUF** with type `"ltxv"` (Gemma GGUF + connectors).

## Latent and resolution rules

- **Video latent**: from `EmptyLTXVLatentVideo` — width, height, length (frames), batch.
- **Frames**: workflow uses **97** (satisfies LTX rule: frames = 8×n + 1).
- **Resolution**: width/height must be **divisible by 32** (e.g. 768×512).
- **Audio latent**: same “length” as video (97 frames), frame_rate 24–25; concatenated with video latent before the model.
- **AV latent**: model sees **concatenated video + audio latent**; after sampling, **LTXVSeparateAVLatent** splits back to video and audio for decoding.

**The LTX 2.3 GGUF is full audio-video**: it expects combined video+audio latent and outputs combined velocity. ltx.cpp currently only feeds **video** latent and uses only the **video** part of the output; the audio path (audio latent, split, audio VAE decode, WAV output) is not yet implemented.

## Scheduler and sampling

- **LTXVScheduler** (first stage): steps=**20**, parameters 2.05, 0.95, true, 0.1 → outputs **SIGMAS**.
- **Sampler**: **euler_ancestral**.
- **CFG**: **4** in first stage; refinement stage uses CFG **1** with LoRA.
- **Refinement** (optional): second pass with **ManualSigmas** `0.909375, 0.725, 0.421875, 0.0` (4 steps), same sampler, CFG 1, LoRA applied.

So a minimal video-only pipeline matches: ~20 steps, euler_ancestral, CFG ~4, and sigmas from an LTX-style scheduler.

## Conditioning

- **LTXVConditioning**: takes positive/negative conditioning and **frame_rate** (e.g. 25).
- Negative prompt in workflow: `"blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles"`.
- Text comes from **Gemma** + **embeddings_connectors**, not T5-XXL. ltx.cpp currently uses T5-XXL; for full parity with this workflow we’d need Gemma + connectors (or confirm DiT accepts T5 embeddings when dimensions match).

## Decode and output

- **Video**: **VAEDecodeTiled** on video latent → images (tile settings 512, 64, 4096, 8).
- **Audio**: **LTXVAudioVAEDecode** on audio latent → audio.
- **CreateVideo**: images + audio + fps (25) → final video.

For ltx.cpp (current): we only decode the **video** part of the DiT output; tiled decode is an implementation detail for memory. Full pipeline would also decode the audio part with **LTXVAudioVAEDecode** and mux.

## Summary for ltx.cpp

1. **DiT**: GGUF `ltx-2.3-22b-dev-*.gguf` — already supported.
2. **VAE**: Video VAE is `ltx-2.3-22b-dev_video_vae.safetensors` on HF; ltx.cpp needs GGUF or safetensors support for this file.
3. **Text**: ComfyUI uses Gemma 3 12B GGUF + embeddings_connectors; ltx.cpp uses T5-XXL — same cross-attention dimension (4096) may allow reuse.
4. **Frames**: Use 8n+1 (e.g. 25, 33, 97).
5. **Resolution**: Width and height divisible by 32.
6. **Sampling**: euler_ancestral, ~20 steps, CFG ~4, LTX-style sigmas (flow-shift / scheduler params 2.05, 0.95, etc.).
7. **Audio not yet in ltx.cpp**: the GGUF is full AV; we currently feed only video latent and use only the video part of the output. To add audio: implement combined AV latent (concat video+audio noise), run DiT (already supports it), split output with LTXVSeparateAVLatent, decode audio latent with **LTXVAudioVAEDecode** (`ltx-2.3-22b-dev_audio_vae.safetensors`) and vocoder, then mux with video (e.g. ffmpeg).

This file is reference only to keep ltx.cpp consistent with the working ComfyUI LTX 2.3 GGUF workflow.
