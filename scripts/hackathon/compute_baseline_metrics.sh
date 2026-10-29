#!/usr/bin/env bash
# Experiment 1: run the ORIGINAL PLV metric scripts with the paper flags on the
# hackathon checkpoint pool. Writes into data/outputs/hackathon_metrics/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/hackathon/env.sh
POOL=${POOL:-data/outputs/hackathon_pool}
# RUN_GLOB selects run directories (default: all); TAG names the output batch.
RUN_GLOB=${RUN_GLOB:-*}
TAG=${TAG:-all}
OUT=${OUT:-data/outputs/hackathon_metrics/$TAG}
mkdir -p "$OUT"
ROOTS=$(ls -d "$(pwd)/$POOL"/$RUN_GLOB/ | sed 's:/$::')
echo "roots: $ROOTS"
DEV=${DEVICE:-cuda:0}
run() { local name="$1"; shift; ( /usr/bin/time -f "$name %e s" nice -n 10 python "$@" ) > "$OUT/$name.log" 2>&1; echo "$name exit=$?"; }
run plv1 diffusion_policy/scripts/plv_lowdim_checkpoint_metrics.py --roots $ROOTS --output "$(pwd)/$OUT/plv1.csv" \
  --device "$DEV" --max-eval-samples 64 --rollout-starts 32 --plv-horizon 10 \
  --dyn-type mlp --dyn-models 5 --dyn-max-transitions 30000 --dyn-epochs 8 --dyn-hidden-dim 256 --skip-latest &
run open_loop diffusion_policy/scripts/plv_open_loop_smoothness_baseline.py --roots $ROOTS --output "$(pwd)/$OUT/open_loop.csv" \
  --device "$DEV" --max-eval-samples 64 --horizon 10 --policy-seed 0 --skip-latest &
run chunk diffusion_policy/scripts/plv_chunk_aware_checkpoint_metrics.py --roots $ROOTS --output "$(pwd)/$OUT/chunk.csv" \
  --device "$DEV" --max-eval-samples 64 --rollout-starts 32 --action-horizons 10 16 --sampling-seeds 0 1 2 3 \
  --dyn-type mlp --dyn-models 5 --dyn-max-transitions 30000 --dyn-hidden-dim 256 --dyn-epochs 8 \
  --dyn-batch-size 1024 --dyn-lr 0.001 --progress-mode semantic --skip-latest &
wait
tail -n 2 "$OUT"/*.log
