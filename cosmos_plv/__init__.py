"""Cosmos-PLV: world-model-guided robot policy evaluation.

Hackathon-era integration package (Nebius x NVIDIA Global AI Hackathon, 2026).
It sits next to the original PLV research code and only imports from it; the
original scripts under ``diffusion_policy/scripts`` are not modified.

Stages (see docs/hackathon/04_cosmos_integration.md):
    A  checkpoint_adapter   load a Diffusion Policy checkpoint + its config
    B  cheap_metrics        Action L1 / open-loop smoothness / replanning (cheap)
    C  rollout_recorder     real robosuite rollouts with video + state traces
    D  cosmos_backend       NVIDIA Cosmos-Reason inference (local or Nebius)
    E  visual_diagnostics   structured, validated physical-failure diagnostics
    F  evaluation_pipeline  per-checkpoint reports with provenance
"""

__version__ = "0.1.0"
