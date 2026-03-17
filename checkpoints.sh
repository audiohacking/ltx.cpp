#!/usr/bin/env bash
# checkpoints.sh – Download raw HuggingFace checkpoints for conversion
#
# Downloads safetensors weights from the official Lightricks LTX-Video repo
# and from the Google T5-XXL repo, then you can run convert.py to produce GGUFs.
#
# Usage:
#   ./checkpoints.sh           # DiT + VAE + T5 (default)
#   ./checkpoints.sh --dit     # DiT only
#   ./checkpoints.sh --vae     # VAE only
#   ./checkpoints.sh --t5      # T5-XXL only

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-./checkpoints}"
DL_DIT=0 DL_VAE=0 DL_T5=0

if [[ $# -eq 0 ]]; then
    DL_DIT=1; DL_VAE=1; DL_T5=1
fi

for arg in "$@"; do
    case "$arg" in
        --dit) DL_DIT=1 ;;
        --vae) DL_VAE=1 ;;
        --t5)  DL_T5=1  ;;
        --all) DL_DIT=1; DL_VAE=1; DL_T5=1 ;;
        *)     echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

mkdir -p "$CKPT_DIR"

hf_dl() {
    local repo="$1" fn="$2" dest="$3"
    [[ -f "$dest" ]] && { echo "  exists: $dest"; return; }
    echo "  huggingface-cli download $repo $fn → $dest"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$repo" "$fn" --local-dir "$(dirname "$dest")"
        mv -f "$(dirname "$dest")/$fn" "$dest" 2>/dev/null || true
    else
        echo "  huggingface-cli not found. Install: pip install huggingface_hub[cli]"
        exit 1
    fi
}

if [[ $DL_DIT -eq 1 ]]; then
    echo "=== LTX-Video DiT checkpoint ==="
    hf_dl "Lightricks/LTX-Video" \
        "ltxv-2b-0.9.6-dev.safetensors" \
        "$CKPT_DIR/ltxv-2b-0.9.6-dev.safetensors"
fi

if [[ $DL_VAE -eq 1 ]]; then
    echo "=== LTX-Video VAE checkpoint ==="
    hf_dl "Lightricks/LTX-Video" \
        "vae.safetensors" \
        "$CKPT_DIR/ltxv-vae.safetensors"
fi

if [[ $DL_T5 -eq 1 ]]; then
    echo "=== T5-XXL checkpoint ==="
    mkdir -p "$CKPT_DIR/t5-xxl"
    # T5-XXL is large; download via HF hub or point to your own copy.
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "google/t5-v1_1-xxl" \
            --local-dir "$CKPT_DIR/t5-xxl" \
            --include "*.safetensors" "*.json"
    else
        echo "  install huggingface-cli: pip install huggingface_hub[cli]"
        exit 1
    fi
fi

echo ""
echo "Checkpoints are in: $CKPT_DIR"
echo ""
echo "Now run convert.py:"
[[ $DL_DIT -eq 1 ]] && echo "  python3 convert.py --model dit --input $CKPT_DIR/ltxv-2b-0.9.6-dev.safetensors --output models/ltxv-2b-BF16.gguf"
[[ $DL_VAE -eq 1 ]] && echo "  python3 convert.py --model vae --input $CKPT_DIR/ltxv-vae.safetensors          --output models/ltxv-vae-BF16.gguf"
[[ $DL_T5  -eq 1 ]] && echo "  python3 convert.py --model t5  --input $CKPT_DIR/t5-xxl/                       --output models/t5-xxl-BF16.gguf"
echo ""
echo "Then quantize:"
echo "  ./quantize.sh Q8_0"
