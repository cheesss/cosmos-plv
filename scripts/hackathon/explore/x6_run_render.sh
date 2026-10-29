#!/usr/bin/env bash
# x6_visual_plan_critic stage 1 launcher: GPU-free check, then one flock'ed Env A process.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
for i in $(seq 1 20); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge 9000 ] && break
  echo "[x6] GPU free ${free} MiB < 9000, waiting ($i/20)"; sleep 60
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt 9000 ]; then echo "[x6] GPU still busy (${free} MiB); abort"; exit 3; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COSMOS_MAX_GPU_FRACTION=0.25
flock /tmp/claude-1000/gpu_explore.lock bash -lc "source scripts/hackathon/env.sh && python scripts/hackathon/explore/x6_plan_render.py $*"
