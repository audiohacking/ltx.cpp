#!/usr/bin/env bash
# test-gpu-migration.sh — Run ltx-generate with full GPU migration attempt and report results.
#
# Use on a machine with enough memory (e.g. 32GB+ unified/VRAM) to test whether
# DiT weights can be moved to the backend (Metal/CUDA/etc.) when the per-tensor
# cap is disabled. Run from repo root; paste the script output when reporting.
#
# Usage (run from anywhere; script switches to repo root):
#   ./scripts/test-gpu-migration.sh
#   bash scripts/test-gpu-migration.sh
#
# Default paths match ./models.sh output (flat under models/). Override: DIT=... VAE=... T5=... if needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BIN="${BIN:-$REPO_ROOT/build/ltx-generate}"
DIT="${DIT:-$REPO_ROOT/models/ltx-2.3-22b-dev-Q4_K_M.gguf}"
VAE="${VAE:-$REPO_ROOT/models/ltx-2.3-22b-dev_video_vae.safetensors}"
T5="${T5:-$REPO_ROOT/models/t5-v1_1-xxl-encoder-Q8_0.gguf}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/output}"
LOG="$OUT_DIR/gpu_migration_test.log"

echo "=============================================="
echo "ltx-generate GPU migration test"
echo "=============================================="
echo "Repo root: $REPO_ROOT"
echo "BIN=$BIN"
echo "DIT=$DIT"
echo "VAE=$VAE"
echo "T5=$T5"
echo "LTX_MIGRATE_MAX_TENSOR_MB=0 (full migration attempt)"
echo "=============================================="

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: $BIN not found or not executable. Build with: cmake --build build -j"
  exit 1
fi

for f in "$DIT" "$VAE" "$T5"; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: Missing model: $f"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo ""
echo "--- Running ltx-generate (short run: 9 frames, 2 steps) ---"
START=$(date +%s)
set +e
LTX_MIGRATE_MAX_TENSOR_MB=0 "$BIN" \
  --dit "$DIT" \
  --vae "$VAE" \
  --t5  "$T5" \
  --prompt "A peaceful waterfall in a lush forest, cinematic, 4K" \
  --frames 9 \
  --height 256 --width 384 \
  --steps 2 \
  --out "$OUT_DIR/frame" \
  -v 2>&1 | tee "$LOG"
EXIT=$?
set -e
END=$(date +%s)
DURATION=$((END - START))

echo ""
echo "=============================================="
echo "RESULTS (paste this when reporting)"
echo "=============================================="
echo "Exit code:    $EXIT"
echo "Duration:     ${DURATION}s"
echo ""

if grep -q "DiT weights on backend" "$LOG"; then
  echo "Migration:    SUCCESS (DiT weights on backend)"
  grep "DiT weights on backend" "$LOG" || true
elif grep -q "using CPU for DiT" "$LOG"; then
  echo "Migration:    SKIPPED/FALLBACK (DiT on CPU)"
else
  echo "Migration:    (check log — migration line not found)"
fi

if grep -q "backend:" "$LOG"; then
  echo "Backend:      $(grep 'backend:' "$LOG" | head -1)"
fi

echo ""
echo "Log file:     $LOG"
echo "=============================================="

exit $EXIT
