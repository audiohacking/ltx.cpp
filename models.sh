#!/usr/bin/env bash
# models.sh – Download LTX-Video GGUF models from Unsloth/HuggingFace
#
# Usage:
#   ./models.sh              # Download recommended Q8_0 models (~7 GB)
#   ./models.sh --quant Q4_K_M  # Pick a specific quantization
#   ./models.sh --all        # Download all quantizations
#   ./models.sh --vae-only   # VAE only

set -euo pipefail

HF_REPO_DIT="unsloth/LTX-2.3-GGUF"
HF_REPO_VAE="unsloth/LTX-2.3-GGUF"
HF_REPO_T5="unsloth/LTX-2.3-GGUF"
MODELS_DIR="${MODELS_DIR:-./models}"
QUANT="${QUANT:-Q8_0}"
DOWNLOAD_ALL=0
VAE_ONLY=0

usage() {
    echo "Usage: $0 [--quant QUANT] [--all] [--vae-only]"
    echo ""
    echo "Options:"
    echo "  --quant QUANT    Quantization level (default: Q8_0)"
    echo "                   Choices: Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16"
    echo "  --all            Download all quantizations"
    echo "  --vae-only       Download VAE only"
    echo ""
    echo "Environment:"
    echo "  MODELS_DIR       Directory for model files (default: ./models)"
    echo "  HF_TOKEN         HuggingFace access token (if required)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --quant)    QUANT="$2"; shift 2 ;;
        --all)      DOWNLOAD_ALL=1; shift ;;
        --vae-only) VAE_ONLY=1; shift ;;
        --help|-h)  usage; exit 0 ;;
        *)          echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

mkdir -p "$MODELS_DIR"

# ── Download helper ───────────────────────────────────────────────────────────

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
        echo "Error: neither curl nor wget found. Please install one."
        exit 1
    fi
}

pip_hf_download() {
    local repo="$1"
    local filename="$2"
    local dest="$3"

    if [[ -f "$dest" ]]; then
        echo "  already exists: $dest"
        return 0
    fi

    if python3 -c "import huggingface_hub" 2>/dev/null; then
        python3 - <<EOF
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(repo_id="$repo", filename="$filename")
shutil.copy(path, "$dest")
print(f"  saved: $dest")
EOF
    else
        hf_download "$repo" "$filename" "$dest"
    fi
}

# ── Model file definitions ────────────────────────────────────────────────────

declare -A DIT_FILES=(
    ["Q4_K_M"]="ltxv-2b-0.9.6-dev-Q4_K_M.gguf"
    ["Q5_K_M"]="ltxv-2b-0.9.6-dev-Q5_K_M.gguf"
    ["Q6_K"]="ltxv-2b-0.9.6-dev-Q6_K.gguf"
    ["Q8_0"]="ltxv-2b-0.9.6-dev-Q8_0.gguf"
    ["BF16"]="ltxv-2b-0.9.6-dev-BF16.gguf"
)

VAE_FILE="ltxv-vae-Q8_0.gguf"
T5_FILE="t5-xxl-Q8_0.gguf"

# ── Download ──────────────────────────────────────────────────────────────────

echo "Models directory: $MODELS_DIR"
echo "Quantization:     $QUANT"
echo ""

if [[ $VAE_ONLY -eq 0 ]]; then
    if [[ $DOWNLOAD_ALL -eq 1 ]]; then
        for q in "${!DIT_FILES[@]}"; do
            fn="${DIT_FILES[$q]}"
            echo "Downloading DiT [$q]: $fn"
            pip_hf_download "$HF_REPO_DIT" "$fn" "$MODELS_DIR/$fn"
        done
    else
        fn="${DIT_FILES[$QUANT]:-}"
        if [[ -z "$fn" ]]; then
            echo "Unknown quant: $QUANT. Choose from: ${!DIT_FILES[*]}"
            exit 1
        fi
        echo "Downloading DiT [$QUANT]: $fn"
        pip_hf_download "$HF_REPO_DIT" "$fn" "$MODELS_DIR/$fn"
    fi

    echo ""
    echo "Downloading T5 text encoder: $T5_FILE"
    pip_hf_download "$HF_REPO_T5" "$T5_FILE" "$MODELS_DIR/$T5_FILE"
fi

echo ""
echo "Downloading VAE: $VAE_FILE"
pip_hf_download "$HF_REPO_VAE" "$VAE_FILE" "$MODELS_DIR/$VAE_FILE"

echo ""
echo "Done. Models are in: $MODELS_DIR"
echo ""
echo "Quick start:"
echo "  mkdir -p output"
echo "  ./build/ltx-generate \\"
echo "    --dit   $MODELS_DIR/${DIT_FILES[$QUANT]:-ltxv-2b-Q8_0.gguf} \\"
echo "    --vae   $MODELS_DIR/$VAE_FILE \\"
echo "    --t5    $MODELS_DIR/$T5_FILE \\"
echo "    --prompt \"A beautiful sunrise over mountain peaks\" \\"
echo "    --frames 25 --height 480 --width 704 \\"
echo "    --steps 40 --cfg 3.0 --out output/frame"
