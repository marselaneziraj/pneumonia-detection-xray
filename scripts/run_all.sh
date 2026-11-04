#!/usr/bin/env bash
# Full pipeline (Linux/macOS/Git-Bash). Assumes the venv is active and data/ is downloaded.
set -euo pipefail
pneumonia prepare --config configs/densenet121.yaml
pneumonia train --config configs/densenet121.yaml
pneumonia train --config configs/resnet18.yaml
pneumonia train --config configs/resnet18_scratch.yaml
pneumonia evaluate
pneumonia gradcam --per-case 3
pneumonia report
