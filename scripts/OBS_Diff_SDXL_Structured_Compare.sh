#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/content/models/hyper_sdxl_4step_471056.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/obs_diff_sdxl_structured_results}"
COCO_CAPTIONS="${COCO_CAPTIONS:-/content/coco/annotations/captions_val2017.json}"

python -u obs_diff_sdxl_structured.py \
  --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --ratios 0.20,0.30,0.40,0.50 \
  --steps 4 \
  --guidance-scale 0.0 \
  --dtype float16 \
  --calibration-prompts 16 \
  --calibration-size 512 \
  --compare-size 1024 \
  --coco-captions "$COCO_CAPTIONS" \
  --package-hessian-gib 1.0 \
  --max-tokens 128 \
  --percdamp 0.01 \
  --ff-prune-chunk 512
