#!/usr/bin/env python3
"""r1 counterfactual replanning, step 2 — branch rollouts (Environment A, GPU for the policy).

For every selected event (checkpoint, env seed, boundary c -> c+1 at T):
  replay the recorded executed actions 0..T-1 (bit-exact), then
  NEW: execute the new plan's steps action_pred[c+1][1:8]   (what the policy did)
  OLD: execute the old plan's steps action_pred[c][9:16]    (ignore the replan)
and continue closed-loop with the same checkpoint from T+7. K continuations per branch;
NEW and OLD use the same batch layout and the same sampling seed, so replicate k of both
branches draws the same diffusion noise (common random numbers).
Protocol: docs/hackathon/explore/r1_counterfactual_replan_protocol.md

Work units are (checkpoint, branch); --shard/--nshard split them across processes.
Output: data/outputs/hackathon_explore/r1_branch/<split>/<checkpoint>__<branch>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/outputs/hackathon_explore/r1_branch"
K = 4
SEED = 4242
STAGE_AT = 40  # steps after the branch ends


def run_unit(ck, branch, events, dataset_cache, device="cuda"):
    import torch

    from cosmos_plv.checkpoint_adapter import load_policy
    from cosmos_plv.rollout_recorder import _make_env, _stack_last_n
    from cosmos_plv.sim_probe import make_probe
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    from diffusion_policy.scripts.plv_chunk_aware_checkpoint_metrics import set_sampling_seed
    from diffusion_policy.scripts.plv_policy_induced_sim_calibration import undo_abs_action

    t_start = time.time()
    task_cfg, policy, _ds, ex = load_policy(ROOT / events[0]["checkpoint_path"], device, dataset_cache)
    task = str(task_cfg.task.task_name)
    dataset_path = ROOT / ex["dataset_path"]
    tf = RotationTransformer("axis_angle", "rotation_6d") if ex["abs_action"] else None
    n_obs, n_act, max_steps, n_lat = ex["n_obs_steps"], ex["n_action_steps"], ex["max_steps"], ex["n_latency_steps"]

    envs, probes, hist, t_now, done, slots, checks = [], [], [], [], [], [], []
    res = [{"event": i, "k": k, "success": False, "first_success_t": -1, "grasped40": None, "objz40": None,
            "success40": None} for i in range(len(events)) for k in range(K)]
    for i, ev in enumerate(events):
        rec = np.load(ROOT / ev["episode_npz"])
        acts, ap, c, T = rec["action"], rec["action_pred"], ev["c"], ev["T"]
        seg = ap[c + 1][None, 1:8] if branch == "NEW" else ap[c][None, 9:16]
        seg = undo_abs_action(seg, tf)[0]
        new_vs_rec = float(np.abs(undo_abs_action(ap[c + 1][None, 1:8], tf)[0] - acts[T:T + 7]).max())
        for k in range(K):
            env = _make_env(dataset_path, ex["obs_keys"], ex["abs_action"], False, (84, 84), "agentview")
            env.seed(int(ev["env_seed"]))
            obs = env.reset()
            probe = make_probe(task, env.env.env)
            h = deque([obs], maxlen=n_obs + 1)
            for t in range(T):
                obs, *_ = env.step(acts[t])
                h.append(obs)
            if k == 0:
                checks.append({"event": i, "replay_eef_err_m": float(np.abs(probe.sample()["eef_pos"] - rec["eef_pos"][T]).max()),
                               "new_plan_vs_recorded_actions": new_vs_rec})
            r = res[i * K + k]
            z0 = float(rec["object_pos"][0][2])
            for j, a in enumerate(seg):
                obs, *_ = env.step(a)
                h.append(obs)
                if probe.sample()["success"] and not r["success"]:
                    r["success"], r["first_success_t"] = True, T + 1 + j
            r["z0"] = z0
            envs.append(env); probes.append(probe); hist.append(h); t_now.append(T + 7); done.append(r["success"])
            slots.append(r)
    if hasattr(policy, "reset"):
        policy.reset()
    set_sampling_seed(SEED)
    n_calls = 0
    while not all(done):
        obs_hist = np.stack([_stack_last_n(h, n_obs) for h in hist]).astype(np.float32)
        with torch.no_grad():
            out = policy.predict_action({"obs": torch.from_numpy(obs_hist).to(device)})
        n_calls += 1
        action = undo_abs_action(out["action"].detach().cpu().numpy()[:, n_lat:], tf)
        for kk in range(action.shape[1]):
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                obs, *_ = env.step(action[i, kk])
                hist[i].append(obs)
                t_now[i] += 1
                s = probes[i].sample()
                r = slots[i]
                start = events[r["event"]]["T"] + 7
                if t_now[i] == start + STAGE_AT:
                    r["grasped40"], r["objz40"], r["success40"] = bool(s["grasped"]), float(s["object_pos"][2]), bool(s["success"])
                if s["success"]:
                    r["success"], r["first_success_t"] = True, t_now[i]
                    done[i] = True
                if t_now[i] >= max_steps:
                    done[i] = True
    for env in envs:
        try:
            env.env.env.close()
        except Exception:
            pass
    return {"checkpoint_id": ck, "branch": branch, "task": task, "K": K, "sampling_seed": SEED, "max_steps": int(max_steps),
            "policy_calls": n_calls, "wall_s": round(time.time() - t_start, 1),
            "events": [{k: ev[k] for k in ("checkpoint_id", "env_seed", "c", "T", "kind")} for ev in events],
            "replay_checks": checks, "replicates": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-events", type=int, default=0, help="smoke test: first N events per checkpoint")
    ap.add_argument("--out-tag", default="")
    a = ap.parse_args()
    evs = json.loads((ROOT / f"results/hackathon/explore/r1_events_{a.split}.json").read_text())["events"]
    by = {}
    for ev in evs:
        by.setdefault(ev["checkpoint_id"], []).append(ev)
    units = [(ck, br) for ck in sorted(by) for br in ("NEW", "OLD")]
    if a.max_events:
        by = {ck: v[: a.max_events] for ck, v in list(by.items())[:1]}
        units = [(ck, br) for ck in sorted(by) for br in ("NEW", "OLD")]
    out = OUT / (a.split + a.out_tag)
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for j, (ck, br) in enumerate(units):
        if j % a.nshard != a.shard:
            continue
        f = out / f"{ck}__{br}.json"
        if f.exists():
            continue
        r = run_unit(ck, br, by[ck], cache)
        f.write_text(json.dumps(r))
        print(json.dumps({"unit": f.name, "events": len(by[ck]), "wall_s": r["wall_s"], "calls": r["policy_calls"],
                          "max_replay_err": max(c["replay_eef_err_m"] for c in r["replay_checks"]),
                          "max_newplan_vs_rec": max(c["new_plan_vs_recorded_actions"] for c in r["replay_checks"])}), flush=True)


if __name__ == "__main__":
    main()
