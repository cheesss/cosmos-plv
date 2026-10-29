#!/usr/bin/env python3
"""s1 VLM-filtered self-training of a plateaued Can policy (Environment A, GPU; `analyze` runs in Environment B).
Protocol: docs/hackathon/explore/s1_self_improve_protocol.md

  collect          200 episodes of base B on seeds 700000-700199 (video + low-dim obs + absolute actions)
  build            one hdf5: the 36 + 4 original 20%-subset demos, the 200 collected episodes, one filter key per arm
  finetune --arm   8,000 steps from B's weights (normalizer fixed from B), constant lr 3e-5, EMA as configured
  eval --arm       200 episodes on seeds 800000-800199 (no video); --arm base evaluates B itself
  analyze          endpoints -> results/hackathon/explore/s1_self_improve.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "data/outputs/hackathon_explore/s1_selfimprove"
BASE = ROOT / ("data/outputs/hackathon_pool_lowdata/paper_dp_can_abs_ph_20_percent_seed1_e1000_r25/checkpoints/"
               "epoch=1000-test_mean_score=0.860.ckpt")
SRC_H5 = ROOT / "data/robomimic/datasets/can/ph/low_dim_abs.hdf5"
H5 = OUT / "s1_can_20pct_plus_self.hdf5"
OBS_KEYS = (("object", 14), ("robot0_eef_pos", 3), ("robot0_eef_quat", 4), ("robot0_gripper_qpos", 2))
COLLECT_SEEDS = list(range(700000, 700200))
EVAL_SEEDS = list(range(800000, 800200))
ARMS = ("continue", "nemotron", "oracle", "all", "nemotron_v2c")
JUDGE = {"nemotron": "nebius_v2.jsonl", "nemotron_v2c": "nebius_v2c.jsonl", "cosmos": "local_v6.jsonl"}
STEPS, LR, BATCH = 8000, 3e-5, 128


def collect():
    from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint
    from cosmos_plv.provenance import git_commit, utc_now, write_json

    out = OUT / "collect" / "s1_base"
    if (out / "summary.json").exists():
        print("already collected")
        return
    cfg = RecorderConfig(batch_size=25, save_obs=True)
    rows = record_checkpoint(BASE, out, COLLECT_SEEDS, COLLECT_SEEDS, cfg, {}, checkpoint_id="s1_base")
    write_json(out / "summary.json", {"study": "s1_self_improve", "checkpoint_id": "s1_base", "checkpoint_path": str(BASE.relative_to(ROOT)),
                                      "seeds": COLLECT_SEEDS, "recorder": cfg.__dict__, "git_commit": git_commit(),
                                      "finished_utc": utc_now(), "episodes": rows,
                                      "success_rate": sum(x["success"] for x in rows) / len(rows)})
    print(json.dumps({"collected": len(rows), "success": sum(x["success"] for x in rows) / len(rows)}))


def judge_success(name):
    f = OUT / "judge" / JUDGE[name]
    v = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        ok = r.get("parse_status") in ("ok", "ok_repaired") and isinstance(r.get("cosmos_task_success"), bool)
        v[int(r["env_seed"])] = bool(r["cosmos_task_success"]) if ok else False  # unparsed -> not selected
    return v


def build():
    import h5py

    summ = json.loads((OUT / "collect/s1_base/summary.json").read_text())
    eps = {int(e["env_seed"]): e for e in summ["episodes"]}
    labels = {"oracle": {s: bool(e["success"]) for s, e in eps.items()}, "all": {s: True for s in eps},
              "nemotron": judge_success("nemotron"), "nemotron_v2c": judge_success("nemotron_v2c")}
    with h5py.File(SRC_H5, "r") as src, h5py.File(H5, "w") as dst:
        train = [x.decode() for x in src["mask/20_percent_train"][:]]
        valid = [x.decode() for x in src["mask/20_percent_valid"][:]]
        g = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            g.attrs[k] = v
        for k in train + valid:
            src.copy(src[f"data/{k}"], g, name=k)
        self_keys = {}
        total = 0
        for i, seed in enumerate(COLLECT_SEEDS):
            e = eps[seed]
            d = np.load(ROOT / e["episode_npz"])
            obs, act = d["obs"], d["action"]
            n = len(act)
            if e["success"]:
                n = int(np.argmax(np.asarray(d["success"]).astype(bool)))  # trace index of first success = steps taken
            key = f"demo_{1000 + i}"
            dg = g.create_group(key)
            o = dg.create_group("obs")
            off = 0
            for name, dim in OBS_KEYS:
                o.create_dataset(name, data=obs[:n, off:off + dim])
                off += dim
            dg.create_dataset("actions", data=act[:n])
            dg.attrs["num_samples"] = n
            dg.attrs["s1_env_seed"] = seed
            dg.attrs["s1_sim_success"] = bool(e["success"])
            self_keys[seed] = key
            total += n
        g.attrs["total"] = int(sum(src[f"data/{k}"].attrs["num_samples"] for k in train + valid) + total)
        m = dst.create_group("mask")
        m.create_dataset("s1_valid", data=np.array(valid, dtype="S"))
        sel = {"continue": []}
        for arm in ("nemotron", "oracle", "all", "nemotron_v2c"):
            sel[arm] = [self_keys[s] for s in COLLECT_SEEDS if labels[arm][s]]
        for arm, keys in sel.items():
            m.create_dataset(f"s1_{arm}", data=np.array(train + keys, dtype="S"))
    info = {arm: {"self_episodes": len(k),
                  "of_which_sim_failures": sum(1 for s in COLLECT_SEEDS if self_keys[s] in set(k) and not eps[s]["success"])}
            for arm, k in sel.items()}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


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
    torch.manual_seed(1)
    np.random.seed(1)
    # train on the arm's data, but keep B's dataset config in the saved checkpoint: load_policy() rebuilds that dataset
    # and the checkpoint's own normalizer (B's, never changed here) is loaded on top of it
    cfg2 = copy.deepcopy(cfg)
    cfg2.task.dataset.dataset_path = str(h5 or H5)
    cfg2.task.dataset.train_filter_key = filter_key or f"s1_{arm}"
    cfg2.task.dataset.val_filter_key = "s1_valid" if filter_key is None else "20_percent_valid"
    dataset = hydra.utils.instantiate(cfg2.task.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True,
                                         drop_last=True)
    device = torch.device("cuda:0")
    model, ema_model = workspace.model, workspace.ema_model  # normalizer stays B's (loaded from the payload)
    model.to(device)
    ema_model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=tuple(cfg.optimizer.betas), eps=cfg.optimizer.eps,
                            weight_decay=1e-6)
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
            ema.step(model)
            step += 1
            if step % 200 == 0:
                log.append({"step": step, "loss": float(loss.item())})
                print(arm, step, round(float(loss.item()), 5), flush=True)
            if step >= steps:
                break
    workspace.global_step, workspace.epoch = step, -1
    workspace.save_checkpoint(path=str(final), use_thread=False)
    (out / "train_log.json").write_text(json.dumps({"arm": arm, "steps": steps, "lr": LR, "batch": BATCH,
                                                    "n_train_episodes": int(dataset.train_mask.sum()),
                                                    "n_train_samples": len(dataset), "log": log}, indent=1))


def evaluate(arm):
    from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint

    ckpt = BASE if arm == "base" else OUT / "finetune" / arm / "final.ckpt"
    out = OUT / "eval" / arm
    if (out / "summary.json").exists():
        print("already evaluated", arm)
        return
    rows = record_checkpoint(ckpt, out, EVAL_SEEDS, [], RecorderConfig(batch_size=50), {}, checkpoint_id=f"s1_{arm}")
    (out / "summary.json").write_text(json.dumps({"arm": arm, "checkpoint": str(ckpt.relative_to(ROOT)), "seeds": EVAL_SEEDS,
                                                  "episodes": rows,
                                                  "success_rate": sum(x["success"] for x in rows) / len(rows)}, indent=1))
    print(json.dumps({"arm": arm, "success": sum(x["success"] for x in rows) / len(rows)}))


def analyze():
    from scipy.stats import binomtest

    res = {"label": "EXPLORATORY; pre-registered in docs/hackathon/explore/s1_self_improve_protocol.md"}
    succ = {}
    for arm in ("base",) + ARMS:
        f = OUT / "eval" / arm / "summary.json"
        if f.exists():
            succ[arm] = {int(e["env_seed"]): bool(e["success"]) for e in json.loads(f.read_text())["episodes"]}
    res["success_rate"] = {a: float(np.mean(list(v.values()))) for a, v in succ.items()}

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

    pairs = {"primary_nemotron_minus_continue": ("nemotron", "continue"), "oracle_minus_continue": ("oracle", "continue"),
             "nemotron_minus_oracle": ("nemotron", "oracle"), "v2c_minus_nemotron": ("nemotron_v2c", "nemotron"),
             "all_minus_continue": ("all", "continue"), "continue_minus_base": ("continue", "base"),
             "v2c_minus_continue": ("nemotron_v2c", "continue")}
    res["paired"] = {k: paired(a, b) for k, (a, b) in pairs.items() if a in succ and b in succ}
    p = res["paired"].get("primary_nemotron_minus_continue")
    res["primary_positive"] = bool(p and p["ci95"][0] > 0)
    # label quality on the 200 collected episodes (secondary 5)
    summ = json.loads((OUT / "collect/s1_base/summary.json").read_text())
    truth = {int(e["env_seed"]): bool(e["success"]) for e in summ["episodes"]}
    ftype = {int(e["env_seed"]): e["failure_type"] for e in summ["episodes"]}
    sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
    from c1_analyze import final_place

    place = {int(e["env_seed"]): final_place(e["episode_npz"]) for e in summ["episodes"]}
    res["collection_success_rate"] = float(np.mean(list(truth.values())))
    res["labels"] = {}
    for name in JUDGE:
        if not (OUT / "judge" / JUDGE[name]).exists():
            continue
        v = judge_success(name)
        tp = sum(v[s] and truth[s] for s in truth)
        fp = [s for s in truth if v[s] and not truth[s]]
        fn = sum((not v[s]) and truth[s] for s in truth)
        res["labels"][name] = {"precision": tp / max(tp + len(fp), 1), "recall": tp / max(tp + fn, 1),
                               "false_successes": len(fp), "failures": sum(not t for t in truth.values()),
                               "false_success_failure_types": {t: sum(ftype[s] == t for s in fp) for t in set(ftype[s] for s in fp)},
                               "false_success_final_place": {q: sum(place[s] == q for s in fp) for q in set(place[s] for s in fp)}}
    bi = OUT / "build_info.json"
    if bi.exists():
        res["training_data"] = json.loads(bi.read_text())
    (ROOT / "results/hackathon/explore/s1_self_improve.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "build", "finetune", "eval", "analyze"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--smoke", action="store_true", help="finetune: 50 steps on the original 20%% subset (pipeline check)")
    a = ap.parse_args()
    {"collect": collect, "build": build, "analyze": analyze}.get(a.step, lambda: None)()
    if a.step == "finetune" and a.smoke:
        finetune("continue", steps=50, h5=SRC_H5, filter_key="20_percent_train", tag="smoke")
    elif a.step == "finetune":
        finetune(a.arm)
    elif a.step == "eval":
        evaluate(a.arm)


if __name__ == "__main__":
    main()
