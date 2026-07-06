"""Conservative Q-Learning — the decision baseline. Self-contained
(SAC backbone + CQL penalty) so the encoder is provably the shared module.
Correct in form, tuned for the pilot, not for SOTA returns."""
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

log = get_logger("cql")

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class GaussianPolicy(nn.Module):
    """Actor over the shared latent z -> tanh-squashed Gaussian action."""

    def __init__(self, latent_dim, hidden, act_dim):
        super().__init__()
        self.body = MLPHead(latent_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, z):
        mu, log_std = self.body(z).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, z):
        mu, log_std = self(z)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        # tanh change-of-variables correction
        logp = dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, logp.sum(-1, keepdim=True)


class QHead(nn.Module):
    """Critic: consumes latent z and action -> scalar Q."""

    def __init__(self, latent_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = MLPHead(latent_dim + act_dim, list(hidden), 1)

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


def train(cfg: dict, buffer: TransitionBuffer, seed: int, out_path: str,
          device: torch.device) -> str:
    set_seed(seed)
    tr, c = cfg["train"], cfg["cql"]
    ad = buffer.act_dim

    # Shared encoder is used by the (online) critic to embed observations.
    encoder = build_encoder(cfg, buffer.obs_dim).to(device)
    q1 = QHead(encoder.latent_dim, ad).to(device)
    q2 = QHead(encoder.latent_dim, ad).to(device)
    policy = GaussianPolicy(encoder.latent_dim, c["action_head_hidden"], ad).to(device)

    # Targets (encoder + critics) via polyak averaging.
    tgt_encoder = copy.deepcopy(encoder)
    tgt_q1, tgt_q2 = copy.deepcopy(q1), copy.deepcopy(q2)
    for p in list(tgt_encoder.parameters()) + list(tgt_q1.parameters()) + list(tgt_q2.parameters()):
        p.requires_grad_(False)

    q_params = list(encoder.parameters()) + list(q1.parameters()) + list(q2.parameters())
    opt_q = torch.optim.Adam(q_params, lr=tr["lr"])
    opt_pi = torch.optim.Adam(policy.parameters(), lr=tr["lr"])
    rng = np.random.default_rng(seed)
    ckpts = checkpoint_steps(tr["steps"], tr.get("checkpoint_fracs", [1.0]))

    def polyak(src, dst, tau):
        for ps, pd in zip(src.parameters(), dst.parameters()):
            pd.data.mul_(1 - tau).add_(tau * ps.data)

    for step in range(tr["steps"]):
        obs, act, rew, next_obs, term = buffer.sample(tr["batch_size"], rng)
        obs = torch.as_tensor(obs, device=device)
        act = torch.as_tensor(act, device=device)
        rew = torch.as_tensor(rew, device=device).unsqueeze(-1)
        next_obs = torch.as_tensor(next_obs, device=device)
        term = torch.as_tensor(term, device=device).unsqueeze(-1)

        z = encoder(obs)

        # ---- critic target ----
        with torch.no_grad():
            zn = tgt_encoder(next_obs)
            zn_pi = encoder(next_obs)          # policy uses online encoder for actions
            a2, logp2 = policy.sample(zn_pi)
            q_next = torch.min(tgt_q1(zn, a2), tgt_q2(zn, a2)) - c["alpha"] * logp2
            # term = TRUE terminals only; truncations must still bootstrap.
            target = rew + c["gamma"] * (1 - term) * q_next

        q1_pred, q2_pred = q1(z, act), q2(z, act)
        bellman = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        # ---- CQL penalty: push down Q off-dataset, pull up on-dataset ----
        n = c["n_action_samples"]
        rand_a = torch.empty(obs.size(0), n, ad, device=device).uniform_(-1, 1)
        z_rep = z.unsqueeze(1).expand(-1, n, -1)
        q1_rand = q1(z_rep.reshape(-1, z.size(-1)), rand_a.reshape(-1, ad)).view(obs.size(0), n)
        pi_a, _ = policy.sample(z)
        q1_pi = q1(z, pi_a)
        cat = torch.cat([q1_rand, q1_pi], dim=1)
        logsumexp = torch.logsumexp(cat, dim=1, keepdim=True)
        cql_pen = (logsumexp - q1_pred).mean()

        q_loss = bellman + c["cql_alpha"] * cql_pen
        opt_q.zero_grad(); q_loss.backward(); opt_q.step()

        # ---- actor ----
        z_det = encoder(obs).detach()          # don't shape encoder with actor loss
        a_pi, logp_pi = policy.sample(z_det)
        q_pi = torch.min(q1(z_det, a_pi), q2(z_det, a_pi))
        pi_loss = (c["alpha"] * logp_pi - q_pi).mean()
        opt_pi.zero_grad(); pi_loss.backward(); opt_pi.step()

        polyak(encoder, tgt_encoder, c["tau"])
        polyak(q1, tgt_q1, c["tau"]); polyak(q2, tgt_q2, c["tau"])

        if step % tr["log_every"] == 0:
            log.info(f"seed={seed} step {step:6d}  q {q_loss.item():.3f}  "
                     f"pi {pi_loss.item():.3f}  cql {cql_pen.item():.3f}")
        if step in ckpts:
            p = out_path.replace(".pt", f"{ckpts[step]}.pt")
            save_encoder(p, encoder,
                         meta={"objective": "cql", "seed": seed,
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
    out = args.out or f"{cfg['analyze']['out_dir']}/encoders/cql_seed{seed}.pt"
    device = resolve_device(cfg["train"]["device"])
    buffer, _ = _build_buffer(cfg)
    train(cfg, buffer, seed, out, device)


if __name__ == "__main__":
    main()
