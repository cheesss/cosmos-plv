#!/usr/bin/env bash
# c1 amendment A1, secondary 8: after the recording finishes, re-judge every failure plus 40 sampled successes per
# stratum with the compartment-explicit Nemotron prompt (v2c_compartment), then re-run the analysis once the main
# c1 pipeline (c1_run.sh) has finished.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
LOG=data/outputs/hackathon_logs/c1_run.log
J=data/outputs/hackathon_explore/c1_judge
PY="env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 $HOME/.venvs/cosmos-plv/bin/python"
until grep -q "RECORD DONE" $LOG; do sleep 30; done
$PY scripts/hackathon/explore/c1_analyze.py --select-v2c
nice -n 10 $PY scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_explore/c1_coverage --no-reasoning \
  --backend nebius --variant v2c_compartment --max-new-tokens 700 --episodes $J/v2c_episodes.json \
  --evidence results/hackathon/explore/nebius_calls_explore.jsonl --out $J/nebius_v2c.jsonl > $J/nebius_v2c.log 2>&1
echo "$(date +%T) V2C DONE $(wc -l < $J/nebius_v2c.jsonl)"
until grep -q "C1 DONE" $LOG; do sleep 30; done
$PY scripts/hackathon/explore/c1_analyze.py > $J/analyze.log 2>&1
echo "$(date +%T) C1 ANALYSIS WITH V2C DONE (exit $?)"
