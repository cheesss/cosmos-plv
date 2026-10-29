#!/usr/bin/env bash
# y3_visual_robustness: run ONE GPU process under the shared explore lock.
#   y3_gpu_run.sh envA <python args...>   (robosuite/EGL, Environment A)
#   y3_gpu_run.sh envB <python args...>   (Cosmos3-Edge, Environment B)
# Waits up to 20 min (60 s polls) for >= 9000 MiB free GPU memory.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
which_env=$1; shift
for i in $(seq 1 20); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 9000 ] && break
  echo "[y3] GPU free ${free} MiB < 9000, waiting ($i/20)"; sleep 60
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 9000 ]; then echo "[y3] GPU still busy (${free} MiB); abort"; exit 3; fi
echo "[y3] GPU free ${free} MiB; starting under flock"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COSMOS_MAX_GPU_FRACTION=0.25
if [ "$which_env" = envA ]; then
  args=$(printf ' %q' "$@")
  flock /tmp/claude-1000/gpu_explore.lock bash -lc "source scripts/hackathon/env.sh && python $args"
else
  export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
  flock /tmp/claude-1000/gpu_explore.lock ~/.venvs/cosmos-plv/bin/python "$@"
fi
