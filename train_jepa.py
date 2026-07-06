"""Latent predictive baseline: BYOL/EMA world model on (z_t, a_t) -> z_{t+1}
with a variance regularizer against collapse. The online encoder is saved."""
from __future__ import annotations
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (load_config, set_seed, resolve_device, save_encoder,
                    get_logger, checkpoint_steps)
from encoder import build_encoder, MLPHead
from data import load_episodes, split_episodes, TransitionBuffer

log = get_logger("jepa")


def variance_reg(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style hinge on per-dim std; discourages latent collapse."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def train(cfg: dict, buffer: TransitionBuffer, seed: int, out_path: str,
          device: torch.device) -> str:
    set_seed(seed)
    tr, j = cfg["train"], cfg["jepa"]

    encoder = build_encoder(cfg, buffer.obs_dim).to(device)          # online f
    tgt_encoder = copy.deepcopy(encoder)                             # target f_ema
    for p in tgt_encoder.parameters():
        p.requires_grad_(False)
    predictor = MLPHead(encoder.latent_dim + buffer.act_dim,
                        j["predictor_hidden"], encoder.latent_dim).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()),
                           lr=tr["lr"], weight_decay=tr["weight_decay"])
    rng = np.random.default_rng(seed)
    ckpts = checkpoint_steps(tr["steps"], tr.get("checkpoint_fracs", [1.0]))

    for step in range(tr["steps"]):
        obs, act, _, next_obs, _ = buffer.sample(tr["batch_size"], rng)
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        next_obs = torch.as_tensor(next_obs, device=device)

        z = encoder(obs)
        with torch.no_grad():
            z_next_tgt = tgt_encoder(next_obs)
        z_pred = predictor(torch.cat([z, act], dim=-1))

        pred_loss = F.mse_loss(z_pred, z_next_tgt)
        reg = variance_reg(z) + variance_reg(z_pred)
        loss = pred_loss + j["var_coeff"] * reg
        opt.zero_grad(); loss.backward(); opt.step()

        # EMA update of the target encoder
        with torch.no_grad():
            d = j["ema_decay"]
            for ps, pt in zip(encoder.parameters(), tgt_encoder.parameters()):
                pt.data.mul_(d).add_((1 - d) * ps.data)

        if step % tr["log_every"] == 0:
            log.info(f"seed={seed} step {step:6d}  pred {pred_loss.item():.4f}  "
                     f"reg {reg.item():.4f}")
        if step in ckpts:
            p = out_path.replace(".pt", f"{ckpts[step]}.pt")
            save_encoder(p, encoder,
                         meta={"objective": "jepa", "seed": seed,
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
    out = args.out or f"{cfg['analyze']['out_dir']}/encoders/jepa_seed{seed}.pt"
    device = resolve_device(cfg["train"]["device"])
    buffer, _ = _build_buffer(cfg)
    train(cfg, buffer, seed, out, device)


if __name__ == "__main__":
    main()
