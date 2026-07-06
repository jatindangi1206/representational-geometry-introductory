"""Null baselines: random-init shared encoder (architecture, no objective)
and dimension-matched Gaussian (the metric floor for unstructured data).
Metric values are read against these, never in absolute terms."""
from __future__ import annotations

import numpy as np
import torch

from common import set_seed
from encoder import build_encoder


@torch.no_grad()
def random_init_features(cfg: dict, obs_dim: int, states: np.ndarray,
                         device: torch.device, seed: int = 0) -> np.ndarray:
    """Features from an untrained shared encoder (random init null)."""
    set_seed(seed)
    enc = build_encoder(cfg, obs_dim).to(device)
    enc.eval()
    x = torch.as_tensor(states, device=device)
    return enc(x).cpu().numpy().astype(np.float32)


def gaussian_dim_features(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Dimension-matched i.i.d. Gaussian features."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def null_distribution(values: list[float]) -> dict:
    """Summarize a set of metric values into mean/std (descriptive context)."""
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if v.size > 1 else 0.0),
            "n": int(v.size), "values": v.tolist()}
