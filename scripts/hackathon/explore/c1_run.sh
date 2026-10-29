#!/usr/bin/env bash
# c1 coverage shift: record the pre-registered rollouts (4 workers, one checkpoint per fresh process, low priority),
# then judge them with the frozen Experiment 2 settings (local Cosmos and Nebius Nemotron in parallel), then analyse.
# Protocol: docs/hackathon/explore/c1_coverage_shift_protocol.md. Safe to re-run: every step skips finished work.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
ROOT=$(pwd)
OUT=data/outputs/hackathon_explore/c1_coverage
J=data/outputs/hackathon_explore/c1_judge
mkdir -p "$OUT" "$J"
echo "$(date +%T) RECORD START"
(
  source scripts/hackathon/env.sh
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python scripts/hackathon/explore/c1_record.py | awk '$4 == "todo" {print $1}' \
    | xargs -P 4 -I{} sh -c 'nice -n 10 python scripts/hackathon/explore/c1_record.py --checkpoint {} 2>&1 | grep -E "^\{|already|Error|Traceback" | tail -3'
)
n=$(ls "$OUT"/*/summary.json 2>/dev/null | wc -l)
echo "$(date +%T) RECORD DONE ($n/31 checkpoints)"
B="env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 $HOME/.venvs/cosmos-plv/bin/python scripts/hackathon/run_cosmos_diagnostics.py --rollouts $OUT --no-reasoning"
nice -n 10 $B --variant v6_perception_last --max-new-tokens 300 --out $J/local_v6.jsonl > $J/local_v6.log 2>&1 &
nice -n 10 $B --backend nebius --variant v2_evidence_first --max-new-tokens 700 \
  --evidence results/hackathon/explore/nebius_calls_explore.jsonl --out $J/nebius_v2.jsonl > $J/nebius_v2.log 2>&1 &
wait
echo "$(date +%T) JUDGES DONE local $(wc -l < $J/local_v6.jsonl) nebius $(wc -l < $J/nebius_v2.jsonl)"
env PYTHONNOUSERSITE=1 $HOME/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/c1_analyze.py > $J/analyze.log 2>&1
echo "$(date +%T) C1 DONE (exit $?)"
