#!/usr/bin/env bash
# s1 self-training, end to end (protocol: docs/hackathon/explore/s1_self_improve_protocol.md). Safe to re-run: each
# step skips finished work. Waits for the c1 recording to free the GPU first.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
S=scripts/hackathon/explore/s1_selfimprove.py
O=data/outputs/hackathon_explore/s1_selfimprove
J=$O/judge
mkdir -p $J
A="nice -n 10 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash -c"
envA() { $A "source scripts/hackathon/env.sh >/dev/null 2>&1; python $S $*"; }
B="env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 $HOME/.venvs/cosmos-plv/bin/python"
until grep -q "RECORD DONE" data/outputs/hackathon_logs/c1_run.log; do sleep 30; done
echo "$(date +%T) COLLECT START"
envA collect 2>&1 | grep -E "^\{|already|Error|Traceback" | tail -3
echo "$(date +%T) COLLECT DONE"
envA eval --arm base 2>&1 | grep -E "^\{|already|Error|Traceback" | tail -3 &
D="$B scripts/hackathon/run_cosmos_diagnostics.py --rollouts $O/collect --no-reasoning"
nice -n 10 $D --backend nebius --variant v2_evidence_first --max-new-tokens 700 \
  --evidence results/hackathon/explore/nebius_calls_explore.jsonl --out $J/nebius_v2.jsonl > $J/nebius_v2.log 2>&1 &
nice -n 10 $D --backend nebius --variant v2c_compartment --max-new-tokens 700 \
  --evidence results/hackathon/explore/nebius_calls_explore.jsonl --out $J/nebius_v2c.jsonl > $J/nebius_v2c.log 2>&1 &
nice -n 10 $D --variant v6_perception_last --max-new-tokens 300 --out $J/local_v6.jsonl > $J/local_v6.log 2>&1 &
wait
echo "$(date +%T) JUDGES DONE v2 $(wc -l < $J/nebius_v2.jsonl) v2c $(wc -l < $J/nebius_v2c.jsonl) v6 $(wc -l < $J/local_v6.jsonl)"
envA build 2>&1 | grep -E "^\{|Error|Traceback" | tail -3
for arm in continue nemotron oracle all nemotron_v2c; do
  envA finetune --arm $arm > $O/finetune_$arm.log 2>&1 &
  sleep 20
done
wait
echo "$(date +%T) FINETUNE DONE $(ls $O/finetune/*/final.ckpt | wc -l)"
for arm in continue nemotron oracle; do envA eval --arm $arm 2>&1 | grep -E "^\{|Error|Traceback" | tail -3 & done
wait
for arm in all nemotron_v2c; do envA eval --arm $arm 2>&1 | grep -E "^\{|Error|Traceback" | tail -3 & done
wait
echo "$(date +%T) EVAL DONE"
$B $S analyze > $O/analyze.log 2>&1
echo "$(date +%T) S1 DONE (exit $?)"
