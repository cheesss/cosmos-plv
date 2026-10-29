import json

import pytest

omegaconf = pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

from cosmos_plv.checkpoint_adapter import discover_checkpoints, validate_checkpoint_config  # noqa: E402


def _cfg(**over):
    base = {"horizon": 16, "n_obs_steps": 2, "n_action_steps": 8, "n_latency_steps": 0,
            "policy": {"num_inference_steps": 100},
            "task": {"obs_keys": ["object"], "abs_action": True, "dataset_path": "d.hdf5",
                     "env_runner": {"max_steps": 400}, "action_dim": 10, "obs_dim": 23}}
    base.update(over)
    return OmegaConf.create(base)


def test_execution_config_read_from_checkpoint():
    ex = validate_checkpoint_config(_cfg())
    assert ex["action_offset"] == 1 and ex["horizon"] == 16 and ex["n_action_steps"] == 8
    with pytest.raises(ValueError):
        validate_checkpoint_config(_cfg(n_action_steps=16))


def test_discover_checkpoints(tmp_path):
    run = tmp_path / "run_a"
    (run / "checkpoints").mkdir(parents=True)
    (run / ".hydra").mkdir()
    (run / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(
        {"task": {"task_name": "can", "env_runner": {"n_test": 50}}, "training": {"seed": 3}})))
    for ep, sc in [(0, 0.0), (10, 0.42)]:
        (run / "checkpoints" / f"epoch={ep:04d}-test_mean_score={sc:.3f}.ckpt").write_bytes(b"x")
    (run / "checkpoints" / "latest.ckpt").write_bytes(b"x")
    (run / "logs.json.txt").write_text("\n".join(json.dumps(r) for r in [
        {"epoch": 0, "test/mean_score": 0.0, "val_loss": 0.5},
        {"epoch": 10, "test/mean_score": 0.42, "val_loss": 0.1}]))
    recs = discover_checkpoints(tmp_path, hash_files=True)
    assert [r.epoch for r in recs] == [0, 10]
    assert recs[1].train_rollout_success == pytest.approx(0.42)
    assert recs[1].checkpoint_id == "can_s3_e0010" and recs[1].sha256 and recs[1].n_train_rollouts == 50
