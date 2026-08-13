#!/usr/bin/env bash
# Example: depth on one image
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python "$ROOT/infer.py" \
  --input "${1:?usage: $0 image_or_video [out_dir]}" \
  --weights "$ROOT/weights/model.safetensors" \
  --out_dir "${2:-$ROOT/outputs/demo}" \
  --window_size 5 \
  --bank_size 32 \
  --width 280 \
  --height 224
