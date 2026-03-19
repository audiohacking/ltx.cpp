# Audio-Video pipeline design (ltx.cpp)

This document describes how the **combined audio+video** path works so that ltx.cpp can produce both video frames and a WAV from the same GGUF DiT and VAEs used in ComfyUI.

## Goal

- **Input**: Text prompt, frame count, resolution (same as today).
- **Output**: Video frames (PPM/PNG) **and** a WAV file (optionally muxed with ffmpeg).
- **Models**: Same stack as [LTX_COMFY_REFERENCE.md](LTX_COMFY_REFERENCE.md): unsloth/LTX-2.3-GGUF DiT, video VAE, **audio VAE** (`ltx-2.3-22b-dev_audio_vae.safetensors`).

## AV latent layout (token concatenation)

The LTX 2.3 DiT is an **audio-video** model: it expects a **single sequence of tokens** formed by concatenating **video tokens** and **audio tokens** in that order. Each token has the same dimension **Pd = 128** (from patch_dim = patch_t × patch_h × patch_w × latent_channels; we use 1×2×2×32 or 128 for video, and 8×16=128 for audio).

- **Video**: Latent shape `[T_lat, H_lat, W_lat, C]` with C=32 (DiT latent_channels). Patch size (1,2,2) → **n_video_tok** = (T_lat/1) × (H_lat/2) × (W_lat/2).
- **Audio**: Latent shape `[T_audio, C_audio, mel_bins]` with C_audio=8, mel_bins=16 (Lightricks `AudioLatentShape`). Patchify: `(B, C, T, F) → (B, T, C*F)` → **n_audio_tok** = T_audio. We align **T_audio = T_lat** so that one audio token per latent frame matches the Comfy “same length as video” rule.
- **Combined**: Input to DiT is `[n_video_tok + n_audio_tok, Pd]`; output is the same shape. We **split** the DiT output: first `n_video_tok` tokens → video velocity; remaining `n_audio_tok` tokens → audio velocity.

References:

- Comfy: “AV latent: model sees **concatenated video + audio latent**; after sampling, **LTXVSeparateAVLatent** splits back to video and audio.”
- Lightricks: `AudioPatchifier.get_token_count` = `tgt_shape.frames`; `patchify` is `b c t f → b t (c f)` so Pd_audio = C_audio × mel_bins = 128.

## Pipeline steps

1. **Latent init**: Video latent `[T_lat, H_lat, W_lat, C]` and audio latent `[T_lat, C_audio, mel_bins]` (i.e. `[T_lat, 8, 16]`) filled with noise.
2. **Per step**:
   - Patchify video → `[n_video_tok, 128]`; patchify audio → `[n_audio_tok, 128]`.
   - Concat → `[n_video_tok + n_audio_tok, 128]`.
   - DiT forward on combined sequence.
   - Split output: video part → unpatchify → video velocity; audio part → unpatchify_audio → audio velocity.
   - Euler step on video latent; Euler step on audio latent.
   - (Optional) Frame conditioning on video as today (start/end frame pinning).
3. **Decode**:
   - **Video**: Existing video VAE decode → frames → write PPM/PNG.
   - **Audio**: Audio VAE decode (latent → spectrogram) → vocoder or Griffin–Lim (spectrogram → waveform) → write WAV.

## Audio VAE

- **File**: `ltx-2.3-22b-dev_audio_vae.safetensors` (unsloth/LTX-2.3-GGUF, `vae/`).
- **Role**: Decoder maps audio latent `[B, C, T, F]` (C=8, F=16, T=T_lat) to spectrogram (e.g. `[B, 1, T×4, 64]` mel bins; exact from Lightricks `LATENT_DOWNSAMPLE_FACTOR=4` and `_adjust_output_shape`).
- **Implementation**: 2D conv decoder (conv_in → mid block → up blocks → norm_out → conv_out), loaded from safetensors; tensor names follow the Lightricks Decoder (e.g. `decoder.conv_in.*`, `decoder.mid.*`, `decoder.up.*`, `decoder.norm_out`, `decoder.conv_out`). Our loader may add a `vae.` prefix.

## Spectrogram → WAV

- **Preferred**: Full vocoder (spectrogram → waveform) as in Lightricks; if not available in C++, use **Griffin–Lim** (inverse STFT from magnitude) for a first milestone.
- **Params**: sample_rate 16000, hop_length 160, mel_bins 64 (from audio VAE / pipeline config).

## CLI

- `--av` : Enable audio+video path (allocate audio latent, concat/split, decode both).
- `--audio-vae <path>` : Optional path to `ltx-2.3-22b-dev_audio_vae.safetensors`; when omitted, audio is synthesized from the denoised latent (fallback path).
- `--out-wav <path>` : Output WAV path (default: `<out prefix>.wav` when `--av`).

## Summary

| Step           | Video path (existing)              | Audio path (new)                          |
|----------------|------------------------------------|-------------------------------------------|
| Init           | Noise `[T,H,W,C]`                  | Noise `[T_lat, 8, 16]`                    |
| Patchify       | `patchify()` → `[n_v, 128]`        | `patchify_audio()` → `[n_a, 128]`        |
| DiT            | —                                  | Single forward on `[n_v+n_a, 128]`       |
| Split          | —                                  | First n_v → video, last n_a → audio      |
| Unpatchify     | `unpatchify()` → velocity          | `unpatchify_audio()` → audio velocity    |
| Step           | Euler on video latent              | Euler on audio latent                    |
| Decode         | Video VAE → frames                 | Audio VAE → spectrogram → WAV            |

This yields a single, proven audio+video pipeline using the same GGUF models as ComfyUI.
