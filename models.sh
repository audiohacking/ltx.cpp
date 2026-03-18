#!/usr/bin/env bash
# models.sh – Download LTX models from Hugging Face (no conversion).
#
# All from unsloth/LTX-2.3-GGUF:
#   DiT:  ltx-2.3-22b-dev-*.gguf (root)
#   VAE:  vae/ltx-2.3-22b-dev_video_vae.safetensors (video decoding)
# T5:    city96/t5-v1_1-xxl-encoder-gguf (t5-v1_1-xxl-encoder-*.gguf)
#
# Note: ltx.cpp loads VAE from GGUF only; the repo provides VAE as safetensors.
#       Use a VAE GGUF if you have one, or the safetensors path for other tools.
#
# Usage:
#   ./models.sh              # DiT + T5 + VAE + full extras (audio VAE, embeddings, LoRA, upscaler, Gemma)
#   ./models.sh --quant Q8_0
#   ./models.sh --all        # All DiT quantizations + T5 + VAE + extras
#   ./models.sh --dit-only   # DiT only
#   ./models.sh --minimal    # DiT + T5 + VAE only (no extras)

set -euo pipefail

HF_REPO="unsloth/LTX-2.3-GGUF"
HF_REPO_T5="city96/t5-v1_1-xxl-encoder-gguf"
HF_REPO_LIGHTRICKS="Lightricks/LTX-2.3"
HF_REPO_GEMMA="unsloth/gemma-3-12b-it-qat-GGUF"
MODELS_DIR="${MODELS_DIR:-./models}"
QUANT="${QUANT:-Q4_K_M}"
DOWNLOAD_ALL=0
DIT_ONLY=0
MINIMAL=0
VAE_VIDEO_FILE="vae/ltx-2.3-22b-dev_video_vae.safetensors"

usage() {
    echo "Usage: $0 [--quant QUANT] [--all] [--dit-only] [--minimal]"
    echo ""
    echo "Options:"
    echo "  --quant QUANT    DiT quantization (default: Q4_K_M)"
    echo "                   Choices: Q2_K, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S,"
    echo "                            Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0, BF16, F16"
    echo "  --all            Download all DiT quantizations"
    echo "  --dit-only       Download DiT only (skip T5, VAE, extras)"
    echo "  --minimal        DiT + T5 + VAE only (skip extras: audio VAE, embeddings, LoRA, upscaler, Gemma)"
    echo ""
    echo "Environment:"
    echo "  MODELS_DIR       Directory for model files (default: ./models)"
    echo "  HF_TOKEN         HuggingFace token (if required)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --quant)    QUANT="$2"; shift 2 ;;
        --all)      DOWNLOAD_ALL=1; shift ;;
        --dit-only) DIT_ONLY=1; shift ;;
        --minimal)  MINIMAL=1; shift ;;
        --help|-h)  usage; exit 0 ;;
        *)          echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$MODELS_DIR"

# ── Download helper (curl/wget only, no Python) ───────────────────────────────

hf_download() {
    local repo="$1"
    local filename="$2"
    local dest="$3"

    if [[ -f "$dest" ]]; then
        echo "  already exists: $dest"
        return 0
    fi

    local url="https://huggingface.co/${repo}/resolve/main/${filename}"
    echo "  downloading: $url"
    echo "  → $dest"

    local curl_args=(-L -# -o "$dest")
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl_args+=(-H "Authorization: Bearer $HF_TOKEN")
    fi

    if command -v curl &>/dev/null; then
        curl "${curl_args[@]}" "$url"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        echo "Error: neither curl nor wget found. Install one."
        exit 1
    fi
}

# ── DiT filename by quant (case statement: safe with set -u) ───────────────────

get_dit_filename() {
    case "$1" in
        Q2_K)   echo "ltx-2.3-22b-dev-Q2_K.gguf" ;;
        Q3_K_M) echo "ltx-2.3-22b-dev-Q3_K_M.gguf" ;;
        Q3_K_S) echo "ltx-2.3-22b-dev-Q3_K_S.gguf" ;;
        Q4_0)   echo "ltx-2.3-22b-dev-Q4_0.gguf" ;;
        Q4_1)   echo "ltx-2.3-22b-dev-Q4_1.gguf" ;;
        Q4_K_M) echo "ltx-2.3-22b-dev-Q4_K_M.gguf" ;;
        Q4_K_S) echo "ltx-2.3-22b-dev-Q4_K_S.gguf" ;;
        Q5_0)   echo "ltx-2.3-22b-dev-Q5_0.gguf" ;;
        Q5_1)   echo "ltx-2.3-22b-dev-Q5_1.gguf" ;;
        Q5_K_M) echo "ltx-2.3-22b-dev-Q5_K_M.gguf" ;;
        Q5_K_S) echo "ltx-2.3-22b-dev-Q5_K_S.gguf" ;;
        Q6_K)   echo "ltx-2.3-22b-dev-Q6_K.gguf" ;;
        Q8_0)   echo "ltx-2.3-22b-dev-Q8_0.gguf" ;;
        BF16)   echo "ltx-2.3-22b-dev-BF16.gguf" ;;
        F16)    echo "ltx-2.3-22b-dev-F16.gguf" ;;
        *)     echo "" ;;
    esac
}

T5_FILE="t5-v1_1-xxl-encoder-Q8_0.gguf"

# ── Download ──────────────────────────────────────────────────────────────────

echo "Models directory: $MODELS_DIR"
echo "DiT quant:        $QUANT"
echo ""

if [[ $DOWNLOAD_ALL -eq 1 ]]; then
    while IFS= read -r line; do
        q="${line%% *}"
        f="${line#* }"
        echo "Downloading DiT [$q]: $f"
        hf_download "$HF_REPO" "$f" "$MODELS_DIR/$f"
    done <<'DIT_LIST'
Q2_K ltx-2.3-22b-dev-Q2_K.gguf
Q3_K_M ltx-2.3-22b-dev-Q3_K_M.gguf
Q3_K_S ltx-2.3-22b-dev-Q3_K_S.gguf
Q4_0 ltx-2.3-22b-dev-Q4_0.gguf
Q4_1 ltx-2.3-22b-dev-Q4_1.gguf
Q4_K_M ltx-2.3-22b-dev-Q4_K_M.gguf
Q4_K_S ltx-2.3-22b-dev-Q4_K_S.gguf
Q5_0 ltx-2.3-22b-dev-Q5_0.gguf
Q5_1 ltx-2.3-22b-dev-Q5_1.gguf
Q5_K_M ltx-2.3-22b-dev-Q5_K_M.gguf
Q5_K_S ltx-2.3-22b-dev-Q5_K_S.gguf
Q6_K ltx-2.3-22b-dev-Q6_K.gguf
Q8_0 ltx-2.3-22b-dev-Q8_0.gguf
BF16 ltx-2.3-22b-dev-BF16.gguf
F16 ltx-2.3-22b-dev-F16.gguf
DIT_LIST
    DIT_EXAMPLE="ltx-2.3-22b-dev-Q4_K_M.gguf"
else
    fn="$(get_dit_filename "$QUANT")"
    if [[ -z "$fn" ]]; then
        echo "Unknown quant: $QUANT. Choose from: Q2_K, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0, BF16, F16"
        exit 1
    fi
    echo "Downloading DiT [$QUANT]: $fn"
    hf_download "$HF_REPO" "$fn" "$MODELS_DIR/$fn"
    DIT_EXAMPLE="$fn"
fi

if [[ $DIT_ONLY -eq 0 ]]; then
    echo ""
    echo "Downloading T5 text encoder: $T5_FILE"
    hf_download "$HF_REPO_T5" "$T5_FILE" "$MODELS_DIR/$T5_FILE"

    echo ""
    echo "Downloading VAE (video): $VAE_VIDEO_FILE"
    VAE_DEST="$MODELS_DIR/ltx-2.3-22b-dev_video_vae.safetensors"
    hf_download "$HF_REPO" "$VAE_VIDEO_FILE" "$VAE_DEST"
fi

# ── Full project extras (skip with --minimal or --dit-only) ───────────────────

if [[ $DIT_ONLY -eq 0 && $MINIMAL -eq 0 ]]; then
    echo ""
    echo "Downloading extras (audio VAE, embeddings, LoRA, upscaler, Gemma, mmproj) ..."
    hf_download "$HF_REPO" "vae/ltx-2.3-22b-dev_audio_vae.safetensors" "$MODELS_DIR/ltx-2.3-22b-dev_audio_vae.safetensors"
    hf_download "$HF_REPO" "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors" "$MODELS_DIR/ltx-2.3-22b-dev_embeddings_connectors.safetensors"
    hf_download "$HF_REPO_LIGHTRICKS" "ltx-2.3-22b-distilled-lora-384.safetensors" "$MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors"
    hf_download "$HF_REPO_LIGHTRICKS" "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    hf_download "$HF_REPO_GEMMA" "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf" "$MODELS_DIR/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
    hf_download "$HF_REPO_GEMMA" "mmproj-BF16.gguf" "$MODELS_DIR/mmproj-BF16.gguf"
fi

echo ""
echo "Done. Models are in: $MODELS_DIR"
echo ""
echo "VAE is from unsloth/LTX-2.3-GGUF (vae/); downloaded as safetensors."
echo "ltx-generate expects a VAE GGUF; use the safetensors path with tools that support it."
echo ""
echo "Quick start (DiT + T5 + VAE, flat in $MODELS_DIR):"
echo "  mkdir -p output"
echo "  ./build/ltx-generate \\"
echo "    --dit   $MODELS_DIR/$DIT_EXAMPLE \\"
echo "    --vae   $MODELS_DIR/ltx-2.3-22b-dev_video_vae.safetensors \\"
echo "    --t5    $MODELS_DIR/$T5_FILE \\"
echo "    --prompt \"A beautiful sunrise over mountain peaks\" \\"
echo "    --frames 25 --height 480 --width 704 \\"
echo "    --steps 40 --cfg 3.0 --out output/frame"
