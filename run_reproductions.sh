#!/usr/bin/env bash
# Reproduce the sinter, calcine, and combined sinter+calcine DQN experiments
# from Karpovich et al. (npj Comput Mater 2024).
#
# Usage:
#   conda activate dqn
#   bash run_reproductions.sh [GPU] [RUN_ID]
#
#   GPU      CUDA_VISIBLE_DEVICES index on this machine (default: 0)
#   RUN_ID   seed/run id -> output folder suffix -N (default: 1)
#
# Outputs (per task) land in:
#   dqn_models/oxides_<task>-<RUN_ID>/        model checkpoints (every 10 iter)
#   training_data/oxides_<task>-<RUN_ID>/     reward/compound/loss curves + final 1000 candidates
#   data/oxides_<task>-<RUN_ID>/              random-init Q-data + scaler
set -euo pipefail

GPU="${1:-0}"
RUN_ID="${2:-1}"

echo "GPU=$GPU  RUN_ID=$RUN_ID"

echo "==================== [1/3] SINTER ===================="
python oxides_driver.py --tasks sinter         --run-id "$RUN_ID" --gpu "$GPU"

echo "==================== [2/3] CALCINE ===================="
python oxides_driver.py --tasks calcine        --run-id "$RUN_ID" --gpu "$GPU"

echo "==================== [3/3] SINTER + CALCINE ===================="
python oxides_driver.py --tasks sinter calcine --run-id "$RUN_ID" --gpu "$GPU"

echo "All three reproductions complete."
# For multiple seeds, re-run with a different RUN_ID, e.g.:
#   bash run_reproductions.sh 0 2
#   bash run_reproductions.sh 0 3
