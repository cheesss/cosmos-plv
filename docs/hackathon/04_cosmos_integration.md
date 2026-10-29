# 04 — NVIDIA Cosmos integration

## Summary

| | Local (this machine) | Hosted (Nebius Token Factory) |
|---|---|---|
| Model | `nvidia/Cosmos3-Edge` (Reasoner tower) | `nvidia/Nemotron-3.5-Super-VL` |
| Revision | `344d602b128d1bbdacb43b08d0a3626f46343e29` (HF, 2026-09-16) | served model id as returned by the API |
| License | OpenMDW-1.1 (model card), not gated | OpenMDW-1.1 (catalog) |
| Weights | Hugging Face `nvidia/Cosmos3-Edge`, Reasoner files only (~7.7 GB download; 4.87 GB of weights per `model.safetensors.index.json`) | not downloaded (serverless) |
| Backend | Hugging Face Transformers 5.21.0, `AutoModelForImageTextToText`, bfloat16, single RTX 3090 | OpenAI-compatible `chat.completions` at `https://api.tokenfactory.nebius.com/v1/` |
| Input actually used | one 448x448 agentview frame per call (`v7`: 8 keyframes per episode) | one episode MP4 re-encoded to 4 fps, 320x320 (`video_url` data URI) |
| Output | JSON `{object_location, held_by_gripper, explanation}` per frame | JSON with the 8-field failure schema |
| Measured GPU memory | 4.93 GB peak (image), 8.39 GB peak (20 s video) | — |
| Code | `cosmos_plv/cosmos_backend.py` | `cosmos_plv/nebius_backend.py` |

Both are NVIDIA open models; Cosmos3-Edge runs locally, Nemotron-3.5-Super-VL is
the NVIDIA model that Nebius Token Factory serves (no Cosmos model is served
there: the public catalog fetched on 2026-10-29 lists Cosmos3-Super-Reasoner as
retired on 2026-08-31).

## Why Cosmos3-Edge

Sources: <https://github.com/NVIDIA/cosmos>,
<https://docs.nvidia.com/cosmos/latest/cosmos3/model_matrix.html>, the
`nvidia/Cosmos3-Edge` model card and the Transformers `cosmos3_edge` docs.

* Cosmos 3 (May 2026) replaces Predict / Reason / Transfer with one omni model
  (Reasoner + Generator towers). The Reason2 / Predict2.5 repositories are marked
  as no longer under active development.
* Edge is the smallest Cosmos 3 checkpoint (~3.9 B BF16 parameters) and the only
  candidate whose card lists NVIDIA Ampere; it is not gated.
* Alternatives considered and not used: Cosmos-Reason2-2B (gated, card lists
  Hopper/Blackwell and a 24 GB minimum), Cosmos-Reason1-7B (older generation,
  ~16.6 GB), Cosmos-Reason2-8B (needs >= 32 GB), Cosmos-Predict2.5-2B
  action-conditioned (WidowX/Bridge 7-D only, needs glibc >= 2.35; this host has
  Ubuntu 20.04 / glibc 2.31), Cosmos3-Nano (larger; generator path needs CUDA 13
  images that do not run on driver 570).

Prompt format follows the Cosmos 3 reasoner guide: system prompt
`You are a helpful assistant.`, media before text, the `<think>` answer suffix
when reasoning is on, official sampling settings
(`temperature 0.6 / top_p 0.95 / top_k 20` with reasoning, `0.7 / 0.8 / 20`
without), video at 4 fps.

## What Cosmos is used for (Mode A — visual diagnostic augmentation)

```
policy checkpoint --(real robosuite rollout, Environment A)--> agentview MP4 + state trace
        |                                                        |
        |                                   privileged sim state -> verified labels (failure_annotator)
        v                                                        |
cheap PLV metrics                  keyframes / video --> Cosmos3-Edge (local)  ----> diagnostic JSON
(Action L1, smoothness, PLV-1,                       --> Nemotron VL (Nebius) ----> diagnostic JSON
 replanning)                                                     |
        \___________________ hybrid screening + report (dashboard) _______/
```

This mode needs simulator rollouts to produce the videos; it is **not**
rollout-free evaluation. What it tests is whether an NVIDIA vision-language
model can recover physically meaningful outcomes (grasp, lift, drop, placement)
from video alone — i.e. without the privileged success signal a simulator gives
for free but a real robot does not — and whether that signal helps choose which
checkpoint to evaluate.

### Prompt development (dev split only; see 06_experiment_protocol.md)

Seven prompt variants were tried on 36 dev episodes (3 dev checkpoints, seeds
300000+). Key observations, all reported in `07_experiment_results.md`:

* Giving Cosmos3-Edge the full video plus the task goal (`v1`) produced
  "success" for 12/12 dev episodes although only 3 succeeded. Its reasoning was
  written as a first-person plan ("I close my gripper ... and carefully lower the
  can"), i.e. it narrated the intended task instead of the observed one.
* Removing the task goal and asking only *where is the red can in this image?*
  (`v6`, one frame) or tracking that answer over 8 keyframes (`v7`) removed most
  of this bias. Success and failure type are then derived by a fixed rule from the
  per-frame visual states (`visual_diagnostics.aggregate_track`), mirroring the
  simulator annotator.
* The hosted Nemotron-3.5-Super-VL handled the full-video failure schema
  (`v2_evidence_first`) well and was frozen for the Nebius backend.

### Output validation

`cosmos_plv/visual_diagnostics.py` extracts the answer after `</think>`,
requires exactly the schema fields with the right types and enums, and reports
logical inconsistencies without repairing them. One repair is allowed and
recorded (`parse_status = ok_repaired`): an object closed with `]` instead of `}`
— a recurring Cosmos3-Edge formatting slip. Unparsable outputs stay in the data
with their raw text and count as wrong in the headline accuracy.

## Mode B — action-conditioned forward dynamics: exploratory (Experiment 4 / 4b)

The Cosmos3-Edge generator is used through diffusers (`Cosmos3OmniPipeline` +
`CosmosActionCondition`, environment C, `configs/hackathon/requirements-cosmos-gen-lock.txt`).
It gets a real frame plus a 16-step action chunk and generates 17 frames.

**Which action slots are trained.** Actions enter through a per-domain linear projection.
`scripts/hackathon/explore/e5_domain_weights.py` reads those tensors on CPU:
* Trained slots: 1, 2, 3, 6, 7, 8, 12, 13, 15 and 20, the model card's embodiment list.
* `libero` (5) and `robocasa` (30) are **untrained**. They have zero biases and
  initialization-level weights.
* diffusers rejects `libero`.
* Our first run (E4) forced it, so E4 tested a random projection and is retracted
  (`07_experiment_results.md` §C3).

**Working path (E4b).** We use the trained Franka slot 8 with the RoboMIND encoding:
* actions are frame-to-frame relative end-effector poses `inv(T_k) T_{k+1}` of the
  **achieved** poses, at 10 fps, i.e. every second 20 Hz step;
* rotation is column rot6d and the gripper is 1 = open;
* quantile normalisation uses the framework's `robomind_franka_stats.json`, downloaded
  from the pinned framework commit with the hash checked;
* the action frame `R_site @ diag(−1, −1, 1)` comes from a synthetic axis sweep.

Without normalisation, i.e. the DROID raw encoding, the arm does not move.

**Results** are in `07_experiment_results.md` §C3: a coarse directional signal (strong
on Can, weak on Square), compressed motion, and no pixel-fidelity gain.

**Not done:**
* Commanded OSC targets are still not achieved poses, so a policy's *planned* chunk
  would need a controller model before it could be imagined.
* No closed-loop "imagined rollout" ranking is claimed.
* `cosmos_plv/action_adapter.py` keeps its blocked stance for the original 10-D
  absolute policy output.

## Reproduce

```bash
source scripts/hackathon/envb.sh
python scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval \
    --variant v7_perception_track --no-reasoning --keyframes 8 --out data/outputs/hackathon_cosmos/eval_local_v7.jsonl
python scripts/hackathon/run_cosmos_diagnostics.py --rollouts data/outputs/hackathon_rollouts_eval \
    --backend nebius --variant v2_evidence_first --no-reasoning --max-new-tokens 700 \
    --out data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl
```
