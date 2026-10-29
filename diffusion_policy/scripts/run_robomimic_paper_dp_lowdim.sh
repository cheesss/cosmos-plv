#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
PROJECT=${PROJECT:-plv_robomimic_validation}
DATASET_TYPE=${DATASET_TYPE:-ph}
GROUP=${GROUP:-robomimic_paper_dp_lowdim_abs_${DATASET_TYPE}_v1}
OUT_ROOT=${OUT_ROOT:-data/outputs/robomimic_paper_dp_lowdim_abs_${DATASET_TYPE}_v1}
LOGGING_MODE=${LOGGING_MODE:-online}

# Diffusion Policy low-dimensional PH matrix for PLV architecture checks.
# Defaults are intentionally smaller than the full BC-RNN matrix so that the
# first architecture-comparison pool finishes in a reasonable time.
TASKS=${TASKS:-"can_lowdim_abs square_lowdim_abs"}
SEEDS=${SEEDS:-"0"}
NUM_EPOCHS=${NUM_EPOCHS:-1001}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-200}
MAX_VAL_STEPS=${MAX_VAL_STEPS:-20}
ROLLOUT_EVERY=${ROLLOUT_EVERY:-50}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}
N_TEST=${N_TEST:-50}
N_TRAIN=${N_TRAIN:-6}
N_ENVS=${N_ENVS:-28}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-100}
BATCH_SIZE=${BATCH_SIZE:-128}
SLOT_LIMIT_FILE=${SLOT_LIMIT_FILE:-data/robomimic/plv_assets/.max_parallel_train_runs}
SLOT_POLL_SECONDS=${SLOT_POLL_SECONDS:-120}
START_LOCK=${START_LOCK:-data/robomimic/plv_assets/.paper_run_start.lock}

mkdir -p "$OUT_ROOT" data/outputs/robomimic_paper_logs data/robomimic/plv_assets

export WANDB_START_METHOD=thread

active_train_count() {
  "$PYTHON" - <<'PY'
import glob
import os
import re
import subprocess

run_dirs = set()
try:
    out = subprocess.check_output(["ps", "-eo", "args="], text=True, errors="ignore")
except Exception:
    out = ""
for line in out.splitlines():
    if not re.search(r"(^|/)(python[0-9.]*)\s+train\.py(\s|$)", line):
        continue
    if "hydra.run.dir=" not in line:
        continue
    m = re.search(r"hydra\.run\.dir=([^ ]+)", line)
    if m:
        run_dirs.add(m.group(1).rstrip("/"))

for marker in glob.glob("data/outputs/**/.paper_run_active", recursive=True):
    try:
        pid = int(open(marker).read().strip())
        os.kill(pid, 0)
    except Exception:
        continue
    run_dirs.add(os.path.dirname(marker).rstrip("/"))

print(len(run_dirs))
PY
}

wait_for_train_slot() {
  if [[ ! -f "$SLOT_LIMIT_FILE" ]]; then
    return 0
  fi
  local limit
  limit="$(tr -dc '0-9' < "$SLOT_LIMIT_FILE" || true)"
  if [[ -z "$limit" ]] || [[ "$limit" == "0" ]]; then
    return 0
  fi
  while true; do
    local active
    active="$(active_train_count)"
    if (( active < limit )); then
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] WAIT train slot active=${active} limit=${limit}"
    sleep "$SLOT_POLL_SECONDS"
  done
}

run_one() {
  local task="$1"
  local seed="$2"
  local task_cfg="$task"
  if [[ "$task" != *_lowdim* ]]; then
    task_cfg="${task}_lowdim"
  fi
  local task_label="${task_cfg%_lowdim_abs}"
  task_label="${task_label%_lowdim}"
  local action_label="delta"
  if [[ "$task_cfg" == *_abs ]]; then
    action_label="abs"
  fi
  local run_name="paper_dp_${task_label}_${action_label}_${DATASET_TYPE}_seed${seed}_e$((NUM_EPOCHS - 1))_s${MAX_TRAIN_STEPS}_r${ROLLOUT_EVERY}_inf${NUM_INFERENCE_STEPS}"
  local run_dir="${OUT_ROOT}/${run_name}"
  local done_file="${run_dir}/.paper_run_done"
  local active_file="${run_dir}/.paper_run_active"

  if [[ -f "$done_file" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP ${run_name} (done marker exists)"
    return 0
  fi

  mkdir -p "$run_dir"
  if [[ -f "$active_file" ]]; then
    local active_pid
    active_pid="$(cat "$active_file" 2>/dev/null || true)"
    if [[ -n "$active_pid" ]] && kill -0 "$active_pid" 2>/dev/null; then
      echo "[$(date --iso-8601=seconds)] SKIP ${run_name} (active marker pid=${active_pid})"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] REMOVE stale active marker for ${run_name}"
    rm -f "$active_file"
  fi

  exec 8>"$START_LOCK"
  flock 8
  wait_for_train_slot
  if [[ -f "$done_file" ]]; then
    echo "[$(date --iso-8601=seconds)] SKIP ${run_name} (done marker exists after slot wait)"
    flock -u 8
    return 0
  fi
  if [[ -f "$active_file" ]]; then
    local locked_active_pid
    locked_active_pid="$(cat "$active_file" 2>/dev/null || true)"
    if [[ -n "$locked_active_pid" ]] && kill -0 "$locked_active_pid" 2>/dev/null; then
      echo "[$(date --iso-8601=seconds)] SKIP ${run_name} (active marker pid=${locked_active_pid} after slot wait)"
      flock -u 8
      return 0
    fi
    rm -f "$active_file"
  fi
  echo "[$(date --iso-8601=seconds)] START ${run_name}"
  echo "$$" > "$active_file"
  flock -u 8

  "$PYTHON" train.py \
    --config-name=train_diffusion_unet_lowdim_workspace \
    task="$task_cfg" \
    task.dataset_type="$DATASET_TYPE" \
    training.seed="$seed" \
    training.num_epochs="$NUM_EPOCHS" \
    training.max_train_steps="$MAX_TRAIN_STEPS" \
    training.max_val_steps="$MAX_VAL_STEPS" \
    training.rollout_every="$ROLLOUT_EVERY" \
    training.checkpoint_every="$CHECKPOINT_EVERY" \
    training.val_every=1 \
    training.sample_every=10 \
    dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers=0 \
    val_dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.num_workers=0 \
    task.dataset.val_ratio=0.0 \
    +task.dataset.train_filter_key=train \
    +task.dataset.val_filter_key=valid \
    task.env_runner.n_test="$N_TEST" \
    task.env_runner.n_test_vis=0 \
    task.env_runner.n_train="$N_TRAIN" \
    task.env_runner.n_train_vis=0 \
    task.env_runner.n_envs="$N_ENVS" \
    policy.num_inference_steps="$NUM_INFERENCE_STEPS" \
    checkpoint.topk.k=50 \
    logging.mode="$LOGGING_MODE" \
    logging.project="$PROJECT" \
    logging.group="$GROUP" \
    logging.name="$run_name" \
    hydra.run.dir="$run_dir"
  rm -f "$active_file"
  touch "$done_file"
  echo "[$(date --iso-8601=seconds)] DONE ${run_name}"
}

for seed in $SEEDS; do
  for task in $TASKS; do
    run_one "$task" "$seed"
  done
done

echo "[$(date --iso-8601=seconds)] ALL PAPER DP LOWDIM ${DATASET_TYPE} RUNS DONE"
