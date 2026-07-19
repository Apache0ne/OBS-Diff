#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/content/models/hyper_sdxl_4step_471056.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/obs_diff_sdxl_hybrid_results}"
COCO_CAPTIONS="${COCO_CAPTIONS:-/content/coco/annotations/captions_val2017.json}"

python -u obs_diff_sdxl_hybrid_colab.py \
  --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --target-reductions 0.20,0.30,0.40,0.50 \
  --steps 4 \
  --guidance-scale 0.0 \
  --dtype float16 \
  --calibration-prompts 12 \
  --calibration-size 1024 \
  --compare-size 1024 \
  --coco-captions "$COCO_CAPTIONS" \
  --package-hessian-gib 0.80 \
  --max-tokens 96 \
  --percdamp 0.01 \
  --ff-prune-chunk 256 \
  --population 72 \
  --generations 100 \
  --elite 12 \
  --mutation-rate 0.08 \
  --protect-fraction 0.15 \
  --recovery-steps 12 \
  --recovery-size 512 \
  --recovery-records 12 \
  --recovery-lr 3e-5
