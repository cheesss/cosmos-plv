#!/usr/bin/env bash
# s2 self-training, end to end (protocol: docs/hackathon/explore/s2_self_improve_protocol.md). Safe to re-run.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
S=scripts/hackathon/explore/s2_selfimprove.py
O=data/outputs/hackathon_explore/s2_selfimprove
J=$O/judge
mkdir -p $J
envA() { nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c "source scripts/hackathon/env.sh >/dev/null 2>&1; python $S $*"; }
B="env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 $HOME/.venvs/cosmos-plv/bin/python"
echo "$(date +%T) COLLECT START"
envA collect --shard 0 2>&1 | grep -E "^\{|already|Error|Traceback" | tail -3 &
sleep 60
envA collect --shard 1 2>&1 | grep -E "^\{|already|Error|Traceback" | tail -3 &
wait
echo "$(date +%T) COLLECT DONE"
envA eval --arm base 2>&1 | grep -E "^\{|already|Error|Traceback" | tail -3 &
D="$B scripts/hackathon/run_cosmos_diagnostics.py --rollouts $O/collect --no-reasoning --backend nebius --variant v2_evidence_first --max-new-tokens 700 --evidence results/hackathon/explore/nebius_calls_explore.jsonl"
nice -n 10 $D --seeds 710000:710200 --out $J/nebius_v2_a.jsonl > $J/nebius_v2_a.log 2>&1 &
nice -n 10 $D --seeds 710200:710400 --out $J/nebius_v2_b.jsonl > $J/nebius_v2_b.log 2>&1 &
wait
echo "$(date +%T) JUDGE+BASE DONE $(cat $J/nebius_v2_*.jsonl | wc -l)"
envA build 2>&1 | grep -E "^\{|Error|Traceback" | tail -3
for arm in continue nemotron oracle human; do envA finetune --arm $arm > $O/finetune_$arm.log 2>&1 & sleep 20; done
wait
echo "$(date +%T) FINETUNE DONE $(ls $O/finetune/*/final.ckpt | wc -l)"
for arm in continue nemotron oracle human; do envA eval --arm $arm 2>&1 | grep -E "^\{|Error|Traceback" | tail -3 & done
wait
echo "$(date +%T) EVAL DONE"
$B $S analyze > $O/analyze.log 2>&1
echo "$(date +%T) S2 DONE (exit $?)"
