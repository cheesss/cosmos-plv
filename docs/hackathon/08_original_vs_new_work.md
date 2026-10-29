# 08 — Original research vs. work done during the hackathon

The hackathon submission period started on 2026-08-26 09:00 PT. This project
builds on earlier research, so this page separates what existed before from
what was built during the hackathon.

## Before the hackathon (not claimed as hackathon work)

* **Research:** "Cheap Metrics First, Replanning Diagnostics Second: Offline
  Checkpoint Screening for Robot Imitation Learning", by Hyun Joon Cho and Taehwan
  Yun (Sungkyunkwan University). The evidence was frozen on 2026-08-25: 1,365
  checkpoints across 45 runs, 50 simulator rollouts each in the primary cohort.
* **Code:** a fork of Diffusion Policy (Chi et al., MIT license) with the PLV
  metric scripts under `diffusion_policy/scripts/`: Action L1, open-loop
  smoothness, PLV-1 with a learned MLP dynamics ensemble, and execution-matched
  replanning.
* **Historical numbers:** reported in the paper and quoted in this repo only as
  history. Examples: Diffusion Policy PLV-1 Spearman 0.556 and Regret@30 0.006;
  Action L1 0.462 and 0.017. None of them were re-run here (the paper checkpoints
  were deleted on 2026-09-28).
* **Commits 2026-08-26 to 2026-10-02** (`cc2bc95` … `bbdcece`): manuscript, figure
  and project-site editing plus storage cleanup. They contain no Cosmos, Nebius or
  evaluation-pipeline code.

## Built during the hackathon (branch `hackathon/cosmos-plv-2026`, from 2026-10-29)

All of it is new code, kept in separate directories. No original research file was
modified, apart from `.gitignore` additions.

1. **Recovery of the original pipeline on new hardware**
   * Datasets pinned by revision and sha256, with a download script.
   * A one-line fix for the robodiff environment (libffi preload).
   * A new reduced checkpoint pool, trained with the unchanged original training
     script.
   * All three original metric scripts verified on the new pool.
2. **`cosmos_plv/`, a new package with six stages.** It covers checkpoint
   discovery and validation, and real robosuite rollouts that record synchronized
   video, privileged state and every predicted action chunk. It also produces
   verified failure labels from simulator state.
3. **NVIDIA Cosmos integration:** `nvidia/Cosmos3-Edge` runs locally through
   Transformers, pinned to a revision. Its outputs pass through schema-validated
   JSON diagnostics.
4. **Prompt protocol:** seven prompt variants were tried on a dev split, and the
   rule for choosing one was written down before any evaluation episode was
   scored. The variant that worked tracks task-free per-keyframe object state.
   This variant came from an observed failure mode: given the task goal,
   Cosmos3-Edge narrated the intended plan instead of what the video showed.
5. **Nebius integration:** runtime calls to the Nebius Token Factory API serve
   `nvidia/Nemotron-3.5-Super-VL` as a hosted second-opinion diagnostic.
   * Regional fallback.
   * A sanitized evidence log of every call (`results/hackathon/nebius_calls.jsonl`).
6. **Execution-matched replanning on real simulator feedback:**
   * The paper's old-plan/new-plan alignment, computed on recorded closed-loop
     plans instead of the learned dynamics model.
7. **Hybrid screening experiment:** a cheap top-q screen followed by VLM-judged
   rollouts, scored on within-run Spearman, Regret@30 and the regret of the final
   pick, with no score fitted on outcomes.
8. **Interactive dashboard (Streamlit):**
   * Recompute cheap metrics with the original script.
   * Record a new rollout.
   * Run Cosmos3-Edge live.
   * Call Nemotron on Nebius live.
   * Recommend which checkpoint to evaluate next, and explain why.
9. **Action-space analysis:** the PLV-to-Cosmos action adapter (geometry plus
   tests), with explicit blockers. The adapter refuses to fake 7-D to 10-D
   padding.
10. **Tests:** unit tests under `tests/hackathon/`. **Docs:** `docs/hackathon/00–10`.
    **Figures:** `media/hackathon/`, all generated from result files.

## Not done (see 10_final_status.md)

* Closed-loop world-model feedback, i.e. ranking checkpoints from imagined rollouts.
  The exploratory Experiment 4b shows that Cosmos3-Edge's trained Franka slot carries
  a coarse directional action signal for this robot. It is not faithful enough to
  replace rollouts (`07_experiment_results.md` §C3).
* Cosmos running on Nebius AI Cloud GPUs. Only Token Factory serverless inference
  was used; no paid GPU instance was created.
* Re-running the paper's 1,365-checkpoint study.
