#!/bin/bash
# bc_v2 retrain: 8 epochs, 16 parallel episode streams. Run inside WSL via scripts/wsl.sh; log in runs/bc_v2/train.log.
mkdir -p runs/bc_v2
python -m dotasim.train.bc --data data/sf_mirror_v1 --epochs 8 --batch 16 --out runs/bc_v2 2>&1 | tee runs/bc_v2/train.log
