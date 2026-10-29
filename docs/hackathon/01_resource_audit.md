# 01 — Resource and infrastructure audit

Measured on 2026-10-29 (KST) on the build machine. Commands: `nvidia-smi`, `df -h`,
`free -h`, `lscpu`, `python --version`, `docker info`, `ldd --version`.

## Hardware

| Item | Value |
|---|---|
| OS | Ubuntu 20.04.6 LTS (glibc 2.31), Linux 5.15 |
| GPU | 1x NVIDIA GeForce RTX 3090, 24 576 MiB, compute capability 8.6 (Ampere, BF16 yes, FP8 no) |
| NVIDIA driver / CUDA | 570.133.07 / CUDA 12.8 (driver); nvcc 12.4 toolkit installed |
| CPU | Intel Core i9-10940X, 28 logical cores |
| RAM | 251 GiB |
| Disk | 916 GB root volume, ~350 GB free at start |
| Containers | Docker 28.0.1 with the `nvidia` runtime |
| Network | Hugging Face, GitHub (SSH), Nebius Token Factory reachable |

The machine is shared with an unrelated always-on service. All training and
rollout jobs were started with `nice -n 10` (rollouts `nice -n 5`), W&B in
offline mode, and the GPU was never reserved exclusively.

## Software environments

Two isolated environments are used because the PLV stack (torch 2.1, robomimic
0.3, transformers 4.30) and the Cosmos stack (transformers >= 5.15, Python
>= 3.11, torch >= 2.6) cannot coexist.

| | Environment A — PLV | Environment B — Cosmos |
|---|---|---|
| Location | conda env `robodiff` (pre-existing) | `~/.venvs/cosmos-plv` (created for the hackathon with mamba) |
| Python | 3.10.19 | 3.11.18 |
| Key packages | torch 2.1.2+cu121, robosuite 1.5.1, robomimic 0.3.0, mujoco 3.4.0, hydra-core 1.3.2, diffusers 0.11.1, numpy 1.24.4 | torch 2.8.0+cu128, transformers 5.21.0, accelerate 1.15.0, av 18.1.0, openai 3.40.0, streamlit 1.66.1 |
| Activate | `source scripts/hackathon/env.sh` | `source scripts/hackathon/envb.sh` |
| Lock file | `configs/hackathon/requirements-plv-lock.txt` | `configs/hackathon/requirements-cosmos-lock.txt` |

The two environments exchange data only through files: rollout MP4 videos,
`.npz` state traces and `summary.json` (written by A), and diagnostic JSONL
(written by B).

### Fix needed for Environment A

`robomimic 0.3.0` imports `mujoco_py`; its compiled extension pulls in the system
`libLLVM-12`, which needs `libffi` symbol version `LIBFFI_BASE_7.0`. In the conda
env `libffi.so.7` is a symlink to `libffi.so.8.1.2`, so the import failed with
`undefined symbol: ffi_type_sint32, version LIBFFI_BASE_7.0`. Fix (no package
change): `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`, set in `scripts/hackathon/env.sh`.

## Hugging Face cache

Empty for Cosmos before the hackathon. Downloaded: `nvidia/Cosmos3-Edge` at
revision `344d602b128d1bbdacb43b08d0a3626f46343e29`, Reasoner files only
(transformer + vision encoder + tokenizer/processor, ~7.7 GB; VAE not
downloaded), 68 s. The model is not gated; no HF token was used.

## Nebius

* No Nebius credentials existed on the machine at the start. The owner created a
  Token Factory API key on 2026-10-29; it is stored outside the repo in
  `~/.config/cosmos-plv/nebius.env` (mode 600) and is never logged.
* Token Factory (serverless inference, OpenAI-compatible) is used. The public
  catalog (`https://tokenfactory.nebius.com/api/public/models_info`) lists five
  NVIDIA models; the only vision model is `nvidia/Nemotron-3.5-Super-VL` (status
  active, us-north1, OpenMDW-1.1). No Cosmos model is served on Token Factory
  (Cosmos3-Super-Reasoner was retired on 2026-08-31 according to the catalog
  research in `04_cosmos_integration.md`).
* Nebius AI Cloud GPU instances were **not** provisioned (no authorization for
  chargeable cloud resources). Estimated cost if needed later: L40S 48 GB
  on-demand about $1.35 per GPU-hour (published price, not measured).

## Financial safety

Only Token Factory pay-per-token calls are made. Measured token usage is logged
per call in `results/hackathon/nebius_calls.jsonl` and summarized in
`05_nebius_execution.md`.
