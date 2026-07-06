"""Reward-access control arm: supervised (z, a) -> r regression. Same reward
signal as CQL but no bootstrapping — separates "sees reward" from "does
credit assignment"."""
from __future__ import annotations
import argparse

import numpy as np
import torch
import torch.nn as nn

from common import (load_config, set_seed, resolve_device, save_encoder,
                    get_logger, checkpoint_steps)
from encoder import build_encoder, MLPHead
from data import load_episodes, split_episodes, TransitionBuffer

log = get_logger("reward")


def train(cfg: dict, buffer: TransitionBuffer, seed: int, out_path: str,
          device: torch.device) -> str:
    set_seed(seed)
    tr = cfg["train"]
    encoder = build_encoder(cfg, buffer.obs_dim).to(device)
    head = MLPHead(encoder.latent_dim + buffer.act_dim,
                   cfg["reward"]["head_hidden"], 1).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                           lr=tr["lr"], weight_decay=tr["weight_decay"])
    rng = np.random.default_rng(seed)
    mse = nn.MSELoss()
    ckpts = checkpoint_steps(tr["steps"], tr.get("checkpoint_fracs", [1.0]))

    encoder.train(); head.train()
    for step in range(tr["steps"]):
        obs, act, rew, *_ = buffer.sample(tr["batch_size"], rng)
        obs_t = torch.as_tensor(obs, device=device)
        act_t = torch.as_tensor(act, device=device)
        rew_t = torch.as_tensor(rew, device=device).unsqueeze(-1)
        pred = head(torch.cat([encoder(obs_t), act_t], dim=-1))
        loss = mse(pred, rew_t)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % tr["log_every"] == 0:
            log.info(f"seed={seed} step {step:6d}  loss {loss.item():.4f}")
        if step in ckpts:
            p = out_path.replace(".pt", f"{ckpts[step]}.pt")
            save_encoder(p, encoder,
                         meta={"objective": "reward", "seed": seed,
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
    out = args.out or f"{cfg['analyze']['out_dir']}/encoders/reward_seed{seed}.pt"
    device = resolve_device(cfg["train"]["device"])
    buffer, _ = _build_buffer(cfg)
    train(cfg, buffer, seed, out, device)


if __name__ == "__main__":
    main()
