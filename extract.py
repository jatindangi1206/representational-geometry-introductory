"""Freeze encoders and extract z on the SAME held-out probe: identical
extraction point, identical states in identical row order, spans preserved."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from common import (load_config, resolve_device, load_encoder_state,
                    set_seed, get_logger)
from encoder import build_encoder
from data import load_episodes, split_episodes, held_out_states

log = get_logger("extract")


@torch.no_grad()
def extract_features(cfg: dict, encoder_path: str, states: np.ndarray,
                     device: torch.device) -> np.ndarray:
    ckpt = load_encoder_state(encoder_path)
    obs_dim = ckpt["meta"]["obs_dim"]
    enc = build_encoder(cfg, obs_dim).to(device)
    enc.load_state_dict(ckpt["state_dict"])
    enc.eval()
    x = torch.as_tensor(states, device=device)
    z = enc(x).cpu().numpy().astype(np.float32)
    return z


def save_probe(path, states, spans, meta) -> None:
    """Persist the shared probe with its data provenance."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, states=states, spans=np.array(spans, dtype=np.int64),
             source=meta["source"], dataset_id=meta["dataset_id"])


def build_probe(cfg):
    """Return the shared probe (states, spans, meta). Reuses a saved
    probe.npz — a correctness guard, not a cache: rebuilding could swap the
    probe under a comparison. run_pilot.py rewrites it each full run."""
    probe_path = Path(cfg["analyze"]["out_dir"]) / "probe.npz"
    if probe_path.exists():
        d = np.load(probe_path)
        meta = {"source": str(d["source"]), "dataset_id": str(d["dataset_id"])}
        log.info(f"reusing probe {probe_path} (source={meta['source']}, "
                 f"n={d['states'].shape[0]})")
        return d["states"], [tuple(s) for s in d["spans"]], meta
    eps, meta = load_episodes(cfg["data"]["dataset_id"], cfg["data"]["max_episodes"])
    _, _, test = split_episodes(eps, cfg["data"]["split"], cfg["experiment"]["seed"])
    states, spans = held_out_states(test, cfg["data"]["n_eval_states"],
                                    cfg["experiment"]["seed"])
    save_probe(probe_path, states, spans, meta)
    return states, spans, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    states, spans, _ = build_probe(cfg)
    feats = extract_features(cfg, args.encoder, states, device)
    np.savez(args.out, features=feats,
             spans=np.array(spans, dtype=np.int64))
    log.info(f"{args.encoder} -> {args.out}  features {feats.shape}")


if __name__ == "__main__":
    main()
