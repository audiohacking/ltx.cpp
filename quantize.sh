#!/usr/bin/env bash
# quantize.sh – Quantize BF16 GGUF models to lower-precision formats
#
# Requires ltx-quantize to be built first: cmake --build build

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-./build}"
MODELS_DIR="${MODELS_DIR:-./models}"
QUANT="${1:-Q8_0}"

usage() {
    echo "Usage: $0 [QUANT]"
    echo ""
    echo "QUANT choices: Q4_K_M | Q5_K_M | Q6_K | Q8_0 (default: Q8_0)"
}

case "$QUANT" in
    Q4_K_M|Q5_K_M|Q6_K|Q8_0) ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown quant: $QUANT"; usage; exit 1 ;;
esac

QUANTIZE="$BUILD_DIR/ltx-quantize"
if [[ ! -x "$QUANTIZE" ]]; then
    echo "Error: $QUANTIZE not found. Build the project first:"
    echo "  mkdir -p build && cd build && cmake .. && cmake --build . -j\$(nproc)"
    exit 1
fi

echo "Quantizing models in $MODELS_DIR to $QUANT ..."

for src in "$MODELS_DIR"/*-BF16.gguf; do
    [[ -f "$src" ]] || continue
    base="${src%-BF16.gguf}"
    dst="${base}-${QUANT}.gguf"
    if [[ -f "$dst" ]]; then
        echo "  skip (exists): $dst"
        continue
    fi
    echo "  $src → $dst"
    "$QUANTIZE" "$src" "$dst" "$QUANT"
done

echo "Done."
