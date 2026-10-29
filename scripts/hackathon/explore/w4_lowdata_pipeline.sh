#!/usr/bin/env bash
# Wave 4 (exploratory): once the low-data long runs finish, compute the original PLV
# metrics with the paper flags and the within-run ranking analysis, separately from
# the main pool. Outputs: results/hackathon/explore/w4_lowdata/.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
POOL=data/outputs/hackathon_pool_lowdata
for r in $POOL/*/; do while [ ! -f "$r/.paper_run_done" ]; do sleep 60; done; done
echo "$(date +%T) low-data training done"
for r in $POOL/*/; do
  n=$(basename "$r")
  POOL=$POOL RUN_GLOB="$n" TAG="$n" OUT="data/outputs/hackathon_metrics_lowdata/$n" \
    bash scripts/hackathon/logged_run.sh "w4_metrics_$n" bash scripts/hackathon/compute_baseline_metrics.sh &
done
wait
mkdir -p results/hackathon/explore/w4_lowdata
bash scripts/hackathon/logged_run.sh w4_analyze env PYTHONNOUSERSITE=1 $HOME/.venvs/cosmos-plv/bin/python \
  scripts/hackathon/analyze_baseline.py --metrics-dir data/outputs/hackathon_metrics_lowdata --pool $POOL \
  --out-dir results/hackathon/explore/w4_lowdata
echo "$(date +%T) W4 DONE"
