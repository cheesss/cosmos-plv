#!/usr/bin/env bash
# End-to-end smoke test with the smallest valid inputs. Prints one PASS/FAIL line
# per stage and writes a JSON report to $OUT/smoke_report.json. Exits 1 (no SMOKE PASS)
# when any stage fails or a model output is invalid; the Nebius call is skipped without a key.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$(pwd)
OUT=${OUT:-/tmp/cosmos_plv_smoke}
rm -rf "$OUT"; mkdir -p "$OUT"
declare -A RC
step() { local name="$1"; shift; local t0=$SECONDS; ( "$@" ) > "$OUT/$name.log" 2>&1; RC[$name]=$?;
         printf '%-22s %s  (%ds, log %s)\n' "$name" "$([[ ${RC[$name]} == 0 ]] && echo PASS || echo FAIL:${RC[$name]})" $((SECONDS - t0)) "$OUT/$name.log"; }

envA() { source scripts/hackathon/env.sh; "$@"; }
envB() { source scripts/hackathon/envb.sh; "$@"; }

step unit_tests_envA envA python -m pytest -q -p no:cacheprovider tests/hackathon
step unit_tests_envB envB python -m pytest -q -p no:cacheprovider tests/hackathon
step dp_train_smoke envA env DATASET_TYPE=ph TASKS=can_lowdim_abs SEEDS=1 OUT_ROOT="$OUT/train" GROUP=smoke \
     N_ENVS=1 N_TEST=1 N_TRAIN=0 NUM_EPOCHS=1 MAX_TRAIN_STEPS=1 MAX_VAL_STEPS=1 ROLLOUT_EVERY=1000 \
     CHECKPOINT_EVERY=1000 NUM_INFERENCE_STEPS=2 BATCH_SIZE=16 LOGGING_MODE=offline PYTHON=python \
     bash diffusion_policy/scripts/run_robomimic_paper_dp_lowdim.sh
CKPT=$(ls "$OUT"/train/*/checkpoints/epoch=0000-*.ckpt 2>/dev/null | head -1)
step record_rollout envA python scripts/hackathon/record_rollouts.py --checkpoint "$CKPT" --checkpoint-id smoke \
     --out "$OUT/rollouts" --seeds 200000:200001 --batch-size 1 --render 256x256
step cosmos_local envB env HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} python scripts/hackathon/run_cosmos_diagnostics.py \
     --rollouts "$OUT/rollouts" --variant v6_perception_last --no-reasoning --max-new-tokens 200 --out "$OUT/cosmos.jsonl"
if envB python -c "import sys; sys.path.insert(0,'.'); from cosmos_plv.nebius_backend import load_api_key; sys.exit(0 if load_api_key() else 1)"; then
  step nebius_call envB python scripts/hackathon/run_cosmos_diagnostics.py --rollouts "$OUT/rollouts" --backend nebius \
       --variant v2_evidence_first --no-reasoning --max-new-tokens 600 --out "$OUT/nebius.jsonl"
else
  echo "nebius_call            SKIPPED (no NEBIUS_API_KEY)"; RC[nebius_call]=skipped
fi
# run_cosmos_diagnostics.py exits 0 even when the model call errored or its output did not
# parse, so check every written row: a backend_error, a parse_status other than ok/ok_repaired,
# or no rows at all from a model that was called fails the smoke test (RC[model_outputs]).
models=(cosmos); [[ ${RC[nebius_call]} == skipped ]] || models+=(nebius)
python3 - "$OUT" "${models[@]}" <<'PY'
import json, sys, pathlib
out, expected = pathlib.Path(sys.argv[1]), sys.argv[2:]
rep, bad = {}, []
for name in ["cosmos", "nebius"]:
    p = out / f"{name}.jsonl"
    if p.exists() and p.read_text().strip():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        r = rows[-1]
        rep[name] = {k: r.get(k) for k in ("model_id", "model_revision", "served_model", "request_id", "parse_status",
                                            "cosmos_task_success", "truth_success", "latency_s", "backend_error")}
        invalid = [x for x in rows if x.get("backend_error") or x.get("parse_status") not in ("ok", "ok_repaired")]
        if invalid:
            x = invalid[0]
            print(f"{name}: model call returned but output invalid ({len(invalid)}/{len(rows)} rows): "
                  f"{x.get('parse_status')} {x.get('backend_error')}")
            bad.append(name)
    elif name in expected:
        print(f"{name}: no output rows in {p}")
        bad.append(name)
(out / "smoke_report.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
sys.exit(1 if bad else 0)
PY
RC[model_outputs]=$?
printf '%-22s %s\n' model_outputs "$([[ ${RC[model_outputs]} == 0 ]] && echo PASS || echo FAIL:${RC[model_outputs]})"
fail=0; for k in "${!RC[@]}"; do [[ ${RC[$k]} == 0 || ${RC[$k]} == skipped ]] || fail=1; done
echo "SMOKE $([[ $fail == 0 ]] && echo PASS || echo FAIL)"; exit $fail
