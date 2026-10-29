#!/usr/bin/env bash
# Same training call as diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh
# (unchanged original), but with configurable robomimic filter keys so that the
# low-data subsets shipped with robomimic v1.5 (e.g. 20_percent_train /
# 20_percent_valid) can be used. Exploration only (wave 4: does PLV beat the epoch
# baseline when long low-data training makes "later" not better?).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/hackathon/env.sh
TASK=${TASK:?can_lowdim_abs or square_lowdim_abs}
SEED=${SEED:-1}
TRAIN_KEY=${TRAIN_KEY:-20_percent_train}
VAL_KEY=${VAL_KEY:-20_percent_valid}
NUM_EPOCHS=${NUM_EPOCHS:-1001}
EVERY=${EVERY:-25}
OUT_ROOT=${OUT_ROOT:-data/outputs/hackathon_pool_lowdata}
name="paper_dp_${TASK%_lowdim_abs}_abs_ph_${TRAIN_KEY%_train}_seed${SEED}_e$((NUM_EPOCHS - 1))_r${EVERY}"
mkdir -p "$OUT_ROOT/$name"
nice -n 10 python train.py --config-name=train_diffusion_unet_lowdim_workspace \
  task="$TASK" task.dataset_type=ph training.seed="$SEED" training.num_epochs="$NUM_EPOCHS" \
  training.max_train_steps=200 training.max_val_steps=20 training.rollout_every="$EVERY" \
  training.checkpoint_every="$EVERY" training.val_every=1 training.sample_every=10 \
  dataloader.batch_size=128 dataloader.num_workers=0 val_dataloader.batch_size=128 val_dataloader.num_workers=0 \
  task.dataset.val_ratio=0.0 +task.dataset.train_filter_key="$TRAIN_KEY" +task.dataset.val_filter_key="$VAL_KEY" \
  task.env_runner.n_test=50 task.env_runner.n_test_vis=0 task.env_runner.n_train=0 task.env_runner.n_train_vis=0 \
  task.env_runner.n_envs=25 policy.num_inference_steps=100 checkpoint.topk.k=50 \
  logging.mode=offline logging.project=plv_robomimic_validation logging.group=hackathon_lowdata logging.name="$name" \
  hydra.run.dir="$OUT_ROOT/$name"
touch "$OUT_ROOT/$name/.paper_run_done"
