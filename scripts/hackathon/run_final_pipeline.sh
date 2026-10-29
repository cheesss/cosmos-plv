#!/usr/bin/env bash
# Finish the pipeline once training is complete. Every step is logged with its exit
# code in results/hackathon/commands_log.jsonl (scripts/hackathon/logged_run.sh).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
L=scripts/hackathon/logged_run.sh
PYB="env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 $HOME/.venvs/cosmos-plv/bin/python"
POOL=data/outputs/hackathon_pool
wait_for() { while [ ! -f "$1" ]; do sleep 60; done; }
echo "$(date +%T) waiting for training"
wait_for $POOL/paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100/.paper_run_done
( RUN_GLOB='paper_dp_can_abs_ph_seed1_*' TAG=can_s1 bash $L baseline_metrics_can_s1 bash scripts/hackathon/compute_baseline_metrics.sh ) &
wait_for $POOL/paper_dp_square_abs_ph_seed1_e300_s200_r10_inf100/.paper_run_done
( RUN_GLOB='paper_dp_square_abs_ph_seed1_*' TAG=square_s1 bash $L baseline_metrics_square_s1 bash scripts/hackathon/compute_baseline_metrics.sh ) &
echo "$(date +%T) waiting for eval recordings"
while ! grep -q "ALL EVAL CHECKPOINTS RECORDED" data/outputs/hackathon_logs/rec_eval_loop.log; do sleep 60; done
export COSMOS_MAX_GPU_FRACTION=0.3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash $L diag_local_v6 $PYB scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval --variant v6_perception_last --no-reasoning --max-new-tokens 300 --out data/outputs/hackathon_cosmos/eval_local_v6.jsonl > data/outputs/hackathon_logs/final_v6.log 2>&1
bash $L diag_nebius_v2 $PYB scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval --backend nebius --variant v2_evidence_first --no-reasoning --max-new-tokens 700 --out data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl > data/outputs/hackathon_logs/final_neb.log 2>&1 &
bash $L diag_nebius_v2_pad201 $PYB scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval --backend nebius --variant v2_evidence_first --no-reasoning --max-new-tokens 700 --pad-to-frames 201 --out data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl > data/outputs/hackathon_logs/final_nebpad.log 2>&1 &
bash $L diag_local_v7 $PYB scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval --variant v7_perception_track --no-reasoning --keyframes 8 --max-new-tokens 300 --out data/outputs/hackathon_cosmos/eval_local_v7.jsonl > data/outputs/hackathon_logs/final_v7.log 2>&1
wait
bash $L refresh_truth $PYB scripts/hackathon/refresh_truth.py
bash $L pool_manifest $PYB scripts/hackathon/make_pool_manifest.py
bash $L analyze_baseline $PYB scripts/hackathon/analyze_baseline.py
bash $L analyze_hybrid bash -lc "source scripts/hackathon/env.sh && python scripts/hackathon/analyze_hybrid.py --diag data/outputs/hackathon_cosmos/eval_local_v6.jsonl data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl"
bash $L compare_backends $PYB scripts/hackathon/compare_backends.py
bash $L compare_backends_pad201 $PYB scripts/hackathon/compare_backends.py --nebius data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl --out results/hackathon/experiment2_backend_comparison_pad201.json
bash $L error_analysis $PYB scripts/hackathon/error_analysis.py
bash $L cascade $PYB scripts/hackathon/explore/x0_cascade.py
bash $L finalize $PYB scripts/hackathon/finalize_results.py
bash $L figures $PYB scripts/hackathon/make_figures.py
echo "$(date +%T) FINAL PIPELINE DONE"
