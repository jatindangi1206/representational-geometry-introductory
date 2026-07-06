"""End-to-end round-one driver: data -> train all objectives (all seeds) ->
nulls -> extract on the shared probe -> geometry battery + verdict.

    python run_pilot.py           # full run from config.yaml
    python run_pilot.py --quick   # tiny budget for a fast plumbing check
"""
from __future__ import annotations
import argparse
import os

import numpy as np

from common import (load_config, resolve_device, dump_json, get_logger,
                    checkpoint_steps)
from data import load_episodes, split_episodes, TransitionBuffer, held_out_states
import train_bc, train_cql, train_jepa, train_reward
from extract import extract_features, save_probe
from nulls import random_init_features, gaussian_dim_features
import analyze as A

log = get_logger("pilot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget for a fast plumbing check")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.quick:
        cfg["train"]["steps"] = 300
        cfg["train"]["log_every"] = 100
        cfg["data"]["max_episodes"] = 120
        cfg["data"]["n_eval_states"] = 600
        cfg["experiment"]["extra_seeds"] = [1]

    out_dir = cfg["analyze"]["out_dir"]
    enc_dir = os.path.join(out_dir, "encoders")
    feat_dir = os.path.join(out_dir, "features")
    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    device = resolve_device(cfg["train"]["device"])
    main_seed = cfg["experiment"]["seed"]
    seeds = [main_seed] + list(cfg["experiment"]["extra_seeds"])

    log.info("== loading data ==")
    eps, meta = load_episodes(cfg["data"]["dataset_id"], cfg["data"]["max_episodes"])
    tr_eps, _, te_eps = split_episodes(eps, cfg["data"]["split"], main_seed)
    buffer = TransitionBuffer(tr_eps)
    states, spans = held_out_states(te_eps, cfg["data"]["n_eval_states"], main_seed)
    spans_arr = np.array(spans, dtype=np.int64)
    # Persist probe + provenance; standalone extract.py MUST reuse this file.
    save_probe(os.path.join(out_dir, "probe.npz"), states, spans, meta)
    log.info(f"source={meta['source']} obs_dim={buffer.obs_dim} "
             f"act_dim={buffer.act_dim} probe_states={states.shape[0]}")
    log.info(f"reward stats: mean={buffer.rew.mean():.4f} "
             f"min={buffer.rew.min():.3f} max={buffer.rew.max():.3f} "
             f"nonzero={float((buffer.rew != 0).mean()):.3%}  "
             f"terminal={float(buffer.term.mean()):.3%}")

    trainers = {"bc": train_bc.train, "cql": train_cql.train,
                "jepa": train_jepa.train, "reward": train_reward.train}
    suffixes = sorted(set(checkpoint_steps(
        cfg["train"]["steps"], cfg["train"].get("checkpoint_fracs", [1.0])).values()))

    for obj, fn in trainers.items():
        for seed in seeds:
            enc_path = os.path.join(enc_dir, f"{obj}_seed{seed}.pt")
            log.info(f"== train {obj} seed={seed} ==")
            fn(cfg, buffer, seed, enc_path, device)
            for sfx in suffixes:
                feats = extract_features(cfg, enc_path.replace(".pt", f"{sfx}.pt"),
                                         states, device)
                np.savez(os.path.join(feat_dir, f"{obj}_seed{seed}{sfx}.npz"),
                         features=feats, spans=spans_arr)

    log.info("== nulls ==")
    for seed in seeds:
        rf = random_init_features(cfg, buffer.obs_dim, states, device, seed)
        np.savez(os.path.join(feat_dir, f"null_random_seed{seed}.npz"),
                 features=rf, spans=spans_arr)
    gf = gaussian_dim_features(states.shape[0], cfg["encoder"]["latent_dim"], main_seed)
    np.savez(os.path.join(feat_dir, "null_gaussian.npz"),
             features=gf, spans=spans_arr)

    log.info("== analyze ==")
    feats = A.load_feature_files(feat_dir)
    ckpts = A.load_feature_files(feat_dir, A.CKPT_RE)
    report = A.analyze(cfg, feats, ckpts)
    A.attach_provenance(report, out_dir)
    dump_json(os.path.join(out_dir, "report.json"), report)
    A.write_summary(report, os.path.join(out_dir, "SUMMARY.md"))
    A.make_figures(report, feats, os.path.join(out_dir, "figures"))

    v = report["verdict"]
    log.info("=" * 60)
    log.info(f"VERDICT: {v['decision'].upper()}")
    log.info(v.get("reason", ""))
    log.info(f"see {out_dir}/SUMMARY.md and {out_dir}/report.json")


if __name__ == "__main__":
    main()
