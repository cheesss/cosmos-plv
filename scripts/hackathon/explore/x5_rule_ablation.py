#!/usr/bin/env python3
"""x5 side analysis (EXPLORATORY, CPU only, no new model calls): aggregation-rule ablation on the
EXISTING v7 per-keyframe Cosmos3-Edge outputs.

Three rules fixed before looking at eval, chosen on dev, reported on eval:
  R0_v7            v7 rule (held = held_by_gripper OR location == in_gripper, any frame)
  R1_loc_only      held = location == in_gripper (ignore the held_by_gripper flag)
  R2_two_consec    held needs >= 2 consecutive keyframes with (held_by_gripper OR in_gripper)
Success is the last-frame location in all three (identical to v7), so only failure types change.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_rule_ablation.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x5_common as C  # noqa: E402
from x5_analyze import truth_map  # noqa: E402
from x5_cosmos_modes import load_v7_frames  # noqa: E402

OUT = C.ROOT / "results/hackathon/explore/x5_cosmos_modes_rule_ablation.json"
RULES = ("R0_v7", "R1_loc_only", "R2_two_consec")


def pred(rule, task, frames):
    if not frames or frames[-1]["parse_status"] not in C.OK:
        return {"parse_status": "last_frame_unparsed"}
    valid = [f for f in frames if f["parse_status"] in C.OK]
    flag = [bool(f.get("p_held_by_gripper")) or f.get("p_object_location") == "in_gripper" for f in valid]
    loc = [f.get("p_object_location") == "in_gripper" for f in valid]
    if rule == "R0_v7":
        held = flag
    elif rule == "R1_loc_only":
        held = loc
    else:
        held = [flag[i] and ((i > 0 and flag[i - 1]) or (i + 1 < len(flag) and flag[i + 1])) for i in range(len(flag))]
    success = valid[-1].get("p_object_location") == C.SUCCESS_LOCATION[task]
    ever, last = any(held), held[-1]
    ft = "none" if success else "no_grasp" if not ever else "not_placed" if last else "dropped"
    return {"parse_status": "ok", "task_success": success, "failure_type": ft, "object_lifted": ever}


def rows_for(split_root, v7, prefix_dev):
    tm = truth_map(split_root)
    out = {r: [] for r in RULES}
    for key, frs in v7.items():
        if key[0].startswith("dev_") != prefix_dev or key not in tm:
            continue
        ep = tm[key]
        frames = [frs[j] for j in sorted(frs)]
        for r in RULES:
            out[r].append({"checkpoint_id": key[0], "env_seed": key[1], "task": ep["task"],
                           "truth_success": ep["success"], "truth_failure_type": ep["failure_type"],
                           "truth_lifted": ep["lifted"], "pred": pred(r, ep["task"], frames),
                           "latency_s": sum(f.get("latency_s") or 0 for f in frames)})
    return out


def main():
    v7 = load_v7_frames()
    dev = rows_for(C.DEV_ROOT, v7, True)
    ev = rows_for(C.EVAL_ROOT, v7, False)
    dev_m = {r: C.metrics(dev[r], has_ftype=True) for r in RULES}
    choice = sorted(RULES, key=lambda r: -dev_m[r]["failure_type_acc_on_failures_all"])[0]
    ev_m = {r: C.metrics(ev[r], has_ftype=True) for r in RULES}
    res = {
        "label": "EXPLORATORY / post-hoc CPU rule ablation on existing v7 outputs; chosen on dev, reported on eval",
        "dev_choice": choice,
        "dev": dev_m, "eval": ev_m,
        "eval_paired_ftype_diff_choice_minus_R0": C.paired_diff(ev[choice], ev["R0_v7"], field="ft"),
        "majority_always_no_grasp": {
            sp: (lambda f: {"n_failures": len(f), "acc": round(sum(x == "no_grasp" for x in f) / len(f), 3),
                            "ci95": C.wilson(sum(x == "no_grasp" for x in f), len(f))})(
                [r["truth_failure_type"] for r in rows["R0_v7"] if not r["truth_success"]])
            for sp, rows in (("dev", dev), ("eval", ev))},
        "eval_episodes": sorted([list(k) for k in v7 if not k[0].startswith("dev_")]),
    }
    OUT.write_text(json.dumps(res, indent=2))
    for split, m in (("dev", dev_m), ("eval", ev_m)):
        for r in RULES:
            x = m[r]
            print(split, r, x["episodes"], "ft_acc", x["failure_type_acc_on_failures_all"], x["failure_type_acc_ci95"],
                  "merged", x["failure_type_acc_square_other_as_no_grasp"], "lifted", x["lifted_acc_scored"])
    print("majority", json.dumps(res["majority_always_no_grasp"]))
    print("choice", choice, json.dumps(res["eval_paired_ftype_diff_choice_minus_R0"]))


if __name__ == "__main__":
    main()
