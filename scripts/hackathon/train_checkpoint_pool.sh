#!/usr/bin/env bash
# Train the reduced hackathon checkpoint pool with the ORIGINAL repo training
# script (diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh).
# Each checkpoint gets N_TEST=50 simulator rollouts at save time (fixed test
# seeds 100000..100049), exactly like the paper's retrospective protocol.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/hackathon/env.sh

OUT_ROOT=${OUT_ROOT:-data/outputs/hackathon_pool}
GROUP=${GROUP:-hackathon_cosmos_plv}
mkdir -p "$OUT_ROOT" data/outputs/hackathon_logs

launch() {
  local task="$1" seed="$2" epochs="$3" every="$4"
  local log="data/outputs/hackathon_logs/${task}_seed${seed}.log"
  DATASET_TYPE=ph TASKS="$task" SEEDS="$seed" OUT_ROOT="$OUT_ROOT" GROUP="$GROUP" \
  N_ENVS=${N_ENVS:-25} N_TEST=50 N_TRAIN=0 NUM_EPOCHS="$epochs" MAX_TRAIN_STEPS=200 \
  MAX_VAL_STEPS=20 ROLLOUT_EVERY="$every" CHECKPOINT_EVERY="$every" \
  NUM_INFERENCE_STEPS=100 BATCH_SIZE=128 LOGGING_MODE=offline PYTHON="$(which python)" \
  nice -n 10 bash diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh > "$log" 2>&1 &
  echo "launched $task seed=$seed epochs=$epochs every=$every pid=$! log=$log"
}

# NUM_EPOCHS is passed as N+1 so that epoch N is the last (saved) epoch.
launch can_lowdim_abs 1 201 5
launch can_lowdim_abs 2 201 10
launch square_lowdim_abs 1 301 10
wait
