"""Shared utilities: seeding, device, config, obs flattening, checkpoint I/O."""
from __future__ import annotations
import logging
import random
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def checkpoint_steps(total_steps: int, fracs) -> dict[int, str]:
    """Map step index -> checkpoint filename suffix ('' for the final save).

    The final (frac 1.0) checkpoint is always included, so every trainer
    produces its end-of-training encoder even if the config omits 1.0.
    """
    out = {}
    for f in sorted(set(list(fracs) + [1.0])):
        step = max(1, int(round(f * total_steps))) - 1
        out[step] = "" if f >= 1.0 else f"_ckpt{int(round(f * 100))}"
    return out


# Fixed key order for Dict observation spaces — defined here and nowhere
# else, so every objective sees the exact same input vector.
# achieved_goal is redundant with observation.
FLATTEN_KEYS = ("observation", "desired_goal")


def flatten_obs(obs: Any) -> np.ndarray:
    """Turn a single observation (Dict or array) into a 1-D float32 vector."""
    if isinstance(obs, dict):
        keys = [k for k in FLATTEN_KEYS if k in obs] or sorted(obs)
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).ravel() for k in keys])
    return np.asarray(obs, dtype=np.float32).ravel()


def flatten_obs_batch(obs_seq) -> np.ndarray:
    """Flatten a sequence/array of observations into (N, obs_dim) float32."""
    return np.stack([flatten_obs(o) for o in obs_seq], axis=0).astype(np.float32)


def save_encoder(path: str, encoder: torch.nn.Module, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder.state_dict(), "meta": meta}, path)


def load_encoder_state(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def dump_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
