#!/usr/bin/env bash
set -euo pipefail

python -m ethiopian_food_vision.train \
  --model baseline --epochs "${BASELINE_EPOCHS:-8}" \
  --run-name baseline --output-dir artifacts/baseline

python -m ethiopian_food_vision.train \
  --model mobilenet_v3_small --epochs "${TRANSFER_EPOCHS:-8}" \
  --freeze-epochs "${FREEZE_EPOCHS:-2}" \
  --run-name mobilenet_v3_small --output-dir artifacts/mobilenet_v3_small
