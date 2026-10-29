from typing import Dict
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
import os
import shutil

import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


register_codecs()


class ACTReplayImageDataset(BaseImageDataset):
    """Image + qpos dataset for ACT-style dual-arm episodes.

    Expected HDF5 layout:

    /data/demo_i/obs/<rgb_key>      (T, H, W, C), uint8
    /data/demo_i/obs/<lowdim_key>   (T, D), float
    /data/demo_i/actions            (T, action_dim), float

    Unlike SonReplayImageDataset, this class does not assume
    action = xyz + rotation_6d + gripper. That keeps the imitation_learning
    dual-arm joint action space intact.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
    ):
        replay_buffer = None
        if use_cache:
            shape_hash = hashlib.md5(
                json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True).encode("utf-8")
            ).hexdigest()[:8]
            cache_zarr_path = f"{dataset_path}.act_{shape_hash}.zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_act_hdf5_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as exc:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise exc
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_act_hdf5_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
            )

        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer["action"])
        )

        for key in self.lowdim_keys:
            normalizer[key] = get_range_normalizer_from_stat(
                array_to_stats(self.replay_buffer[key])
            )

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][t_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }


def _convert_act_hdf5_to_replay(store, shape_meta, dataset_path, n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys = []
    lowdim_keys = []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            rgb_keys.append(key)
        elif obs_type == "low_dim":
            lowdim_keys.append(key)
        else:
            raise RuntimeError(f"Unsupported obs type: {obs_type}")

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path, "r") as file:
        demos = file["data"]
        episode_ends = []
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array("episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True)

        for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
            data_key = f"obs/{key}" if key != "action" else "actions"
            chunks = []
            for i in range(len(demos)):
                chunks.append(demos[f"demo_{i}"][data_key][:].astype(np.float32))
            this_data = np.concatenate(chunks, axis=0)
            expected_shape = (
                (n_steps,) + tuple(shape_meta["action"]["shape"])
                if key == "action"
                else (n_steps,) + tuple(shape_meta["obs"][key]["shape"])
            )
            assert this_data.shape == expected_shape, (
                f"{key} shape mismatch: got {this_data.shape}, expected {expected_shape}"
            )
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    c, h, w = tuple(shape_meta["obs"][key]["shape"])
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=Jpeg2k(level=50),
                        dtype=np.uint8,
                    )
                    for episode_idx in range(len(demos)):
                        hdf5_arr = demos[f"demo_{episode_idx}"]["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                completed, futures = concurrent.futures.wait(
                                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                                )
                                for future in completed:
                                    if not future.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for future in completed:
                    if not future.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    return ReplayBuffer(root)
