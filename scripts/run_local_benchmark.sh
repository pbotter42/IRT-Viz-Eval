#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m irt_viz_eval.cli all \
  --data data/benchmark \
  --analysis-output output/analysis \
  --n-per-model 1 \
  --styles all \
  --models oracle,noisy,blind \
  --seed 20260711
