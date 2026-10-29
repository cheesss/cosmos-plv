#!/usr/bin/env bash
# Record the eval set one checkpoint per fresh process (avoids the EGL render-context
# memory growth seen when one process records many checkpoints). Stops when every
# expected eval checkpoint has a summary.json. Waits for GPU headroom first.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/hackathon/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MIN_FREE=${MIN_FREE:-6000}
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -lt "$MIN_FREE" ]; then sleep 60; continue; fi
  out=$(nice -n 5 python scripts/hackathon/record_eval_set.py --max-checkpoints 1 2>&1 | grep -E '^\{|recorded|Error|Traceback' | tail -3)
  echo "$(date +%T) $out"
  missing=$(python3 - <<'PY'
import pathlib
done = {p.parent.name for p in pathlib.Path('data/outputs/hackathon_rollouts_eval').glob('*/summary.json')}
exp = {f'can_s1_e{e:04d}' for e in range(0, 201, 20)} | {f'can_s2_e{e:04d}' for e in range(0, 201, 20)} | {f'square_s1_e{e:04d}' for e in range(0, 301, 30)}
print(len(exp - done))
PY
)
  [ "$missing" = "0" ] && { echo "ALL EVAL CHECKPOINTS RECORDED"; break; }
  echo "$out" | grep -q '^{' || sleep 120
done
