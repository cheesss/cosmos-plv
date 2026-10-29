#!/usr/bin/env python3
"""s2 VLM-filtered self-training, redesigned after the s1 null (Environment A, GPU; `analyze` runs in Environment B).
Protocol: docs/hackathon/explore/s2_self_improve_protocol.md

  collect --shard 0|1   400 episodes of B2 (epoch 125) on seeds 710000-710399, two halves in parallel
  build                 one hdf5: 36 + 4 original demos, 36 held-out human demos, 400 collected episodes, a key per arm
  finetune --arm        8,000 steps from B2's EMA weights (normalizer fixed), warm-up 200 then cosine 1e-4 -> 0
  eval --arm            500 episodes on seeds 810000-810499 (no video); --arm base evaluates B2 itself
  analyze               endpoints -> results/hackathon/explore/s2_self_improve.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "data/outputs/hackathon_explore/s2_selfimprove"
BASE = ROOT / ("data/outputs/hackathon_pool_lowdata/paper_dp_can_abs_ph_20_percent_seed1_e1000_r25/checkpoints/"
               "epoch=0125-test_mean_score=0.580.ckpt")
SRC_H5 = ROOT / "data/robomimic/datasets/can/ph/low_dim_abs.hdf5"
H5 = OUT / "s2_can_20pct_plus_self.hdf5"
OBS_KEYS = (("object", 14), ("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2))
COLLECT_SEEDS = list(range(710000, 710400))
EVAL_SEEDS = list(range(810000, 810500))
ARMS = ("continue", "nemotron", "oracle", "human")
STEPS, PEAK_LR, WARMUP, BATCH, N_HUMAN = 8000, 1e-4, 200, 128, 36


def shard_seeds(shard):
    return COLLECT_SEEDS[:200] if shard == 0 else COLLECT_SEEDS[200:]


def collect(shard):
    from cosmos_plv.provenance import git_commit, utc_now, write_json
    from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint

    out = OUT / "collect" / f"s2_base_{'ab'[shard]}"
    if (out / "summary.json").exists():
        print("already collected", shard)
        return
    seeds = shard_seeds(shard)
    cfg = RecorderConfig(batch_size=25, save_obs=True)
    rows = record_checkpoint(BASE, out, seeds, seeds, cfg, {}, checkpoint_id="s2_base")
    write_json(out / "summary.json", {"study": "s2_self_improve", "checkpoint_id": "s2_base", "shard": shard,
                                      "checkpoint_path": str(BASE.relative_to(ROOT)), "seeds": seeds,
                                      "recorder": cfg.__dict__, "git_commit": git_commit(), "finished_utc": utc_now(),
                                      "episodes": rows, "success_rate": sum(x["success"] for x in rows) / len(rows)})
    print(json.dumps({"shard": shard, "collected": len(rows), "success": sum(x["success"] for x in rows) / len(rows)}))


def collected():
    eps = {}
    for summ in sorted((OUT / "collect").glob("*/summary.json")):
        for e in json.loads(summ.read_text())["episodes"]:
            eps[int(e["env_seed"])] = e
    return eps


def nemotron_success():
    v = {}
    for f in sorted((OUT / "judge").glob("nebius_v2_*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            ok = r.get("parse_status") in ("ok", "ok_repaired") and isinstance(r.get("cosmos_task_success"), bool)
            v[int(r["env_seed"])] = bool(r["cosmos_task_success"]) if ok else False  # unparsed -> not selected
    return v


def human_pool(src):
    m = lambda k: {x.decode() for x in src[f"mask/{k}"][:]}  # noqa: E731
    pool = sorted(m("train") - m("20_percent_train") - m("20_percent_valid"), key=lambda s: int(s.split("_")[1]))
    rng = np.random.default_rng(0)
    return [pool[i] for i in np.sort(rng.choice(len(pool), N_HUMAN, replace=False))]


def build():
    import h5py

    eps = collected()
    assert sorted(eps) == COLLECT_SEEDS, "collection incomplete"
    labels = {"oracle": {s: bool(e["success"]) for s, e in eps.items()}, "nemotron": nemotron_success()}
    with h5py.File(SRC_H5, "r") as src, h5py.File(H5, "w") as dst:
        train = [x.decode() for x in src["mask/20_percent_train"][:]]
        valid = [x.decode() for x in src["mask/20_percent_valid"][:]]
        human = human_pool(src)
        g = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            g.attrs[k] = v
        for k in train + valid + human:
            src.copy(src[f"data/{k}"], g, name=k)
        self_keys, n_frames, total = {}, {}, 0
        for i, seed in enumerate(COLLECT_SEEDS):
            e = eps[seed]
            d = np.load(ROOT / e["episode_npz"])
            obs, act = d["obs"], d["action"]
            n = int(np.argmax(np.asarray(d["success"]).astype(bool))) if e["success"] else len(act)
            key = f"demo_{1000 + i}"
            dg = g.create_group(key)
            o = dg.create_group("obs")
            off = 0
            for name, dim in OBS_KEYS:
                o.create_dataset(name, data=obs[:n, off:off + dim])
                off += dim
            dg.create_dataset("actions", data=act[:n])
            dg.attrs["num_samples"] = n
            dg.attrs["s2_env_seed"] = seed
            dg.attrs["s2_sim_success"] = bool(e["success"])
            self_keys[seed], n_frames[seed] = key, n
            total += n
        g.attrs["total"] = int(sum(src[f"data/{k}"].attrs["num_samples"] for k in train + valid + human) + total)
        m = dst.create_group("mask")
        m.create_dataset("s2_valid", data=np.array(valid, dtype="S"))
        sel = {"continue": [], "human": list(human)}
        for arm in ("nemotron", "oracle"):
            sel[arm] = [self_keys[s] for s in COLLECT_SEEDS if labels[arm][s]]
        for arm, keys in sel.items():
            m.create_dataset(f"s2_{arm}", data=np.array(train + keys, dtype="S"))
    inv = {k: s for s, k in self_keys.items()}
    info = {}
    for arm, keys in sel.items():
        fails = [inv[k] for k in keys if k in inv and not eps[inv[k]]["success"]]
        frames = sum(n_frames[inv[k]] for k in keys if k in inv)
        info[arm] = {"added_episodes": len(keys), "of_which_sim_failures": len(fails),
                     "self_frames": frames, "failure_frames": sum(n_frames[s] for s in fails)}
    info["human_demo_keys"] = human
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "human_demo_keys"}))


def finetune(arm, steps=STEPS, h5=None, filter_key=None, tag=None):
    import copy

    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    from diffusion_policy.common.pytorch_util import dict_apply

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    out = OUT / "finetune" / (tag or arm)
    final = out / "final.ckpt"
    if final.exists():
        print("already trained", arm)
        return
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(open(BASE, "rb"), pickle_module=dill, map_location="cpu")
    cfg = copy.deepcopy(payload["cfg"])
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(out))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    workspace.model.load_state_dict(workspace.ema_model.state_dict())  # start from B2's EMA weights (the ones evaluated)
    torch.manual_seed(1)
    np.random.seed(1)
    # train on the arm's data, but keep B2's dataset config in the saved checkpoint: load_policy() rebuilds that dataset
    # and the checkpoint's own normalizer (B2's, never changed here) is loaded on top of it
    cfg2 = copy.deepcopy(cfg)
    cfg2.task.dataset.dataset_path = str(h5 or H5)
    cfg2.task.dataset.train_filter_key = filter_key or f"s2_{arm}"
    cfg2.task.dataset.val_filter_key = "s2_valid" if filter_key is None else "20_percent_valid"
    dataset = hydra.utils.instantiate(cfg2.task.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=True)
    device = torch.device("cuda:0")
    model, ema_model = workspace.model, workspace.ema_model
    model.to(device)
    ema_model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps,
                            weight_decay=1e-6)

    def lr_factor(i):  # linear warm-up, then cosine to 0 at `steps`
        if i < WARMUP:
            return (i + 1) / WARMUP
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (i - WARMUP) / max(1, steps - WARMUP))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
    log, step = [], 0
    model.train()
    while step < steps:
        for batch in loader:
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            loss = model.compute_loss(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            sched.step()
            ema.step(model)
            step += 1
            if step % 200 == 0:
                log.append({"step": step, "loss": float(loss.item()), "lr": float(sched.get_last_lr()[0])})
                print(arm, step, round(float(loss.item()), 5), flush=True)
            if step >= steps:
                break
    workspace.global_step, workspace.epoch = step, -1
    workspace.save_checkpoint(path=str(final), use_thread=False)
    (out / "train_log.json").write_text(json.dumps({"arm": arm, "steps": steps, "peak_lr": PEAK_LR, "warmup": WARMUP,
                                                    "batch": BATCH, "n_train_episodes": int(dataset.train_mask.sum()),
                                                    "n_train_samples": len(dataset), "log": log}, indent=1))


def evaluate(arm):
    from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint

    ckpt = BASE if arm == "base" else OUT / "finetune" / arm / "final.ckpt"
    out = OUT / "eval" / arm
    if (out / "summary.json").exists():
        print("already evaluated", arm)
        return
    rows = record_checkpoint(ckpt, out, EVAL_SEEDS, [], RecorderConfig(batch_size=50), {}, checkpoint_id=f"s2_{arm}")
    (out / "summary.json").write_text(json.dumps({"arm": arm, "checkpoint": str(ckpt.relative_to(ROOT)), "seeds": EVAL_SEEDS,
                                                  "episodes": rows,
                                                  "success_rate": sum(x["success"] for x in rows) / len(rows)}, indent=1))
    print(json.dumps({"arm": arm, "success": sum(x["success"] for x in rows) / len(rows)}))


def analyze():
    from scipy.stats import binomtest

    res = {"label": "EXPLORATORY; pre-registered in docs/hackathon/explore/s2_self_improve_protocol.md"}
    succ, ftypes = {}, {}
    for arm in ("base",) + ARMS:
        f = OUT / "eval" / arm / "summary.json"
        if f.exists():
            e = json.loads(f.read_text())["episodes"]
            succ[arm] = {int(x["env_seed"]): bool(x["success"]) for x in e}
            ftypes[arm] = {}
            for x in e:
                if not x["success"]:
                    ftypes[arm][x["failure_type"]] = ftypes[arm].get(x["failure_type"], 0) + 1
    res["success_rate"] = {a: float(np.mean(list(v.values()))) for a, v in succ.items()}
    res["failure_types_by_arm"] = ftypes

    def paired(a, b, n=10000):
        x = np.array([succ[a][s] for s in EVAL_SEEDS], float)
        y = np.array([succ[b][s] for s in EVAL_SEEDS], float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(x), (n, len(x)))
        diffs = (x[idx] - y[idx]).mean(1)
        a_only, b_only = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        return {"diff": float((x - y).mean()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "discordant": [a_only, b_only],
                "mcnemar_exact_p": float(binomtest(a_only, a_only + b_only).pvalue) if a_only + b_only else 1.0}

    pairs = {"primary_nemotron_minus_continue": ("nemotron", "continue"), "human_minus_continue": ("human", "continue"),
             "oracle_minus_continue": ("oracle", "continue"), "nemotron_minus_oracle": ("nemotron", "oracle"),
             **{f"{a}_minus_base": (a, "base") for a in ARMS}}
    res["paired"] = {k: paired(a, b) for k, (a, b) in pairs.items() if a in succ and b in succ}
    p = res["paired"].get("primary_nemotron_minus_continue")
    res["primary_positive"] = bool(p and p["ci95"][0] > 0)
    pc = res["paired"].get("human_minus_continue")
    res["positive_control_passed"] = bool(pc and pc["ci95"][0] > 0)
    oc = res["paired"].get("oracle_minus_continue")
    res["h2_interpretable"] = bool(oc and oc["ci95"][0] > 0)
    # secondary 5: Nemotron label quality on the 400 collected episodes
    eps = collected()
    truth = {s: bool(e["success"]) for s, e in eps.items()}
    sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
    from c1_analyze import final_place

    v = nemotron_success()
    fp = [s for s in truth if v.get(s) and not truth[s]]
    tp = sum(1 for s in truth if v.get(s) and truth[s])
    fn = sum(1 for s in truth if not v.get(s) and truth[s])
    res["collection_success_rate"] = float(np.mean(list(truth.values())))
    res["nemotron_labels"] = {"precision": tp / max(tp + len(fp), 1), "recall": tp / max(tp + fn, 1),
                              "false_successes": len(fp), "failures": sum(not t for t in truth.values()),
                              "false_success_failure_types": {t: sum(eps[s]["failure_type"] == t for s in fp)
                                                              for t in {eps[s]["failure_type"] for s in fp}},
                              "false_success_final_place": {q: sum(final_place(eps[s]["episode_npz"]) == q for s in fp)
                                                            for q in {final_place(eps[s]["episode_npz"]) for s in fp}}}
    bi = OUT / "build_info.json"
    if bi.exists():
        info = json.loads(bi.read_text())
        res["training_data"] = {k: v for k, v in info.items() if k != "human_demo_keys"}
        n = info.get("nemotron", {})
        if n.get("self_frames"):
            res["nemotron_labels"]["contamination_frames"] = n["failure_frames"] / n["self_frames"]
            res["nemotron_labels"]["contamination_episodes"] = n["of_which_sim_failures"] / max(n["added_episodes"], 1)
    (ROOT / "results/hackathon/explore/s2_self_improve.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "build", "finetune", "eval", "analyze"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="finetune: 50 steps on the original 20%% subset (pipeline check)")
    a = ap.parse_args()
    if a.step == "collect":
        collect(a.shard)
    elif a.step == "build":
        build()
    elif a.step == "finetune" and a.smoke:
        finetune("continue", steps=50, h5=SRC_H5, filter_key="20_percent_train", tag="smoke")
    elif a.step == "finetune":
        finetune(a.arm)
    elif a.step == "eval":
        evaluate(a.arm)
    else:
        analyze()


if __name__ == "__main__":
    main()
