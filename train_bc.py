"""Behavior Cloning — the observational baseline: predict dataset actions
from observations. Only the encoder is saved; heads are discarded."""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn

from common import (load_config, set_seed, resolve_device, save_encoder,
                    get_logger, checkpoint_steps)
from encoder import build_encoder, MLPHead
from data import load_episodes, split_episodes, TransitionBuffer

log = get_logger("bc")


def train(cfg: dict, buffer: TransitionBuffer, seed: int, out_path: str,
          device: torch.device) -> str:
    set_seed(seed)
    tr = cfg["train"]
    encoder = build_encoder(cfg, buffer.obs_dim).to(device)
    head = MLPHead(encoder.latent_dim, cfg["bc"]["action_head_hidden"],
                   buffer.act_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                           lr=tr["lr"], weight_decay=tr["weight_decay"])
    rng = np.random.default_rng(seed)
    mse = nn.MSELoss()
    ckpts = checkpoint_steps(tr["steps"], tr.get("checkpoint_fracs", [1.0]))

    encoder.train(); head.train()
    for step in range(tr["steps"]):
        obs, act, *_ = buffer.sample(tr["batch_size"], rng)
        obs_t = torch.as_tensor(obs, device=device)
        act_t = torch.as_tensor(act, device=device)
        pred = head(encoder(obs_t))
        loss = mse(pred, act_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % tr["log_every"] == 0:
            log.info(f"seed={seed} step {step:6d}  loss {loss.item():.4f}")
        if step in ckpts:
            p = out_path.replace(".pt", f"{ckpts[step]}.pt")
            save_encoder(p, encoder,
                         meta={"objective": "bc", "seed": seed,
                               "obs_dim": buffer.obs_dim,
                               "latent_dim": encoder.latent_dim})
            log.info(f"seed={seed} saved -> {p}")
    return out_path


def _build_buffer(cfg):
    eps, meta = load_episodes(cfg["data"]["dataset_id"], cfg["data"]["max_episodes"])
    tr, _, _ = split_episodes(eps, cfg["data"]["split"], cfg["experiment"]["seed"])
    return TransitionBuffer(tr), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    out = args.out or f"{cfg['analyze']['out_dir']}/encoders/bc_seed{seed}.pt"
    device = resolve_device(cfg["train"]["device"])
    buffer, _ = _build_buffer(cfg)
    train(cfg, buffer, seed, out, device)


if __name__ == "__main__":
    main()
