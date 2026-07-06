"""The single shared encoder. Every objective uses THIS class as its
observation backbone; objective-specific parameters live in heads that
consume the latent z. Do not subclass or fork per-objective."""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class Encoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden: Sequence[int] = (256, 256),
        latent_dim: int = 128,
        activation: str = "relu",
        layernorm: bool = True,
    ):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act())
            d = h
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(cfg: dict, obs_dim: int) -> Encoder:
    """Construct an encoder from the `encoder:` block of the config."""
    ec = cfg["encoder"]
    return Encoder(
        obs_dim=obs_dim,
        hidden=ec["hidden"],
        latent_dim=ec["latent_dim"],
        activation=ec.get("activation", "relu"),
        layernorm=ec.get("layernorm", True),
    )


class MLPHead(nn.Module):
    """Generic MLP head that consumes latent z (and optionally an action)."""

    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int,
                 activation: str = "relu"):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
