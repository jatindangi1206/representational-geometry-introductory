"""Smoke checks for the pilot's correctness-critical paths.

Run: python test_pilot.py
Covers exactly the things that would silently corrupt the science if broken:
termination/truncation split, probe determinism + provenance round-trip,
train->extract shape sanity, metric identities.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

import train_bc
import train_reward
from common import load_config, checkpoint_steps
from data import Episode, TransitionBuffer, held_out_states
from extract import extract_features, save_probe
from metrics import linear_cka, mutual_knn, participation_ratio

cfg = load_config()
cfg["train"]["steps"] = 5
cfg["train"]["log_every"] = 5


def make_episodes(n=20, obs_dim=6, act_dim=2, seed=0):
    """Random test fixture; episodes end by time limit (truncation)."""
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        T = int(rng.integers(15, 40))
        obs = rng.normal(size=(T + 1, obs_dim)).astype(np.float32)
        acts = rng.normal(size=(T, act_dim)).astype(np.float32)
        rew = rng.normal(size=T).astype(np.float32)
        dones = np.zeros(T, bool); dones[-1] = True
        terms = np.zeros(T, bool)
        eps.append(Episode(obs[:T], acts, rew, dones, terms,
                           obs[1:], i, np.arange(T)))
    return eps


# 1. Truncation is NOT termination: time-limit episodes must yield an
#    all-zero Bellman mask from the buffer.
eps = make_episodes()
assert eps[0].dones[-1] and not eps[0].terminations.any(), \
    "time-limit end must set dones but never terminations"
buf = TransitionBuffer(eps)
*_, term = buf.sample(256, np.random.default_rng(0))
assert term.max() == 0.0, "Bellman mask must bootstrap through truncations"

# 2. Shared probe is deterministic and round-trips through probe.npz.
s1, sp1 = held_out_states(eps[:8], 100, seed=0)
s2, sp2 = held_out_states(eps[:8], 100, seed=0)
assert np.array_equal(s1, s2) and sp1 == sp2, "probe must be deterministic"
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "probe.npz"
    save_probe(p, s1, sp1, {"source": "test", "dataset_id": "fixture"})
    d = np.load(p)
    assert np.array_equal(d["states"], s1) and str(d["source"]) == "test"

# 3. Train tiny BC + reward encoders, checkpoints included, extract on probe.
cfg["train"]["checkpoint_fracs"] = [0.5, 1.0]
with tempfile.TemporaryDirectory() as td:
    enc_path = str(Path(td) / "bc_test.pt")
    train_bc.train(cfg, buf, seed=0, out_path=enc_path, device=torch.device("cpu"))
    assert Path(td, "bc_test_ckpt50.pt").exists(), "mid-training checkpoint missing"
    z = extract_features(cfg, enc_path, s1, torch.device("cpu"))
    rw_path = str(Path(td) / "reward_test.pt")
    train_reward.train(cfg, buf, seed=0, out_path=rw_path, device=torch.device("cpu"))
    zr = extract_features(cfg, rw_path, s1, torch.device("cpu"))
assert z.shape == zr.shape == (s1.shape[0], cfg["encoder"]["latent_dim"])
assert np.isfinite(z).all() and np.isfinite(zr).all()

# 3b. checkpoint_steps: final always present, suffixes as expected.
cs = checkpoint_steps(300, [0.25, 0.5, 1.0])
assert cs[299] == "" and cs[74] == "_ckpt25" and cs[149] == "_ckpt50"
assert "" in checkpoint_steps(100, [0.5]).values(), "final save must be implicit"

# 4. Metric identities on identical representations.
assert abs(linear_cka(z, z) - 1.0) < 1e-5
assert mutual_knn(z, z, k=5) == 1.0
assert 1.0 <= participation_ratio(z) <= z.shape[1]

# 5. Permutation test: a truly separated outlier group must give a small p
#    (min attainable with 3 seeds/objective is ~1/84), a homogeneous
#    similarity structure must give p ~ 1.
from analyze import permutation_outlier_test
groups = {o: [f"{o}_seed{i}" for i in range(3)] for o in ("bc", "cql", "jepa")}
labs = [l for g in groups.values() for l in g]
key = lambda a, b: "|".join(sorted((a, b)))
sep = {key(a, b): (0.1 if "cql" in (a[:3], b[:3]) else 0.9)
       for i, a in enumerate(labs) for b in labs[i + 1:]}
r = permutation_outlier_test(sep, groups["bc"], groups["jepa"], groups["cql"],
                             5000, np.random.default_rng(0))
assert r["gap"] > 0.7 and r["p_value"] < 0.03, r
flat = {k: 0.5 for k in sep}
r = permutation_outlier_test(flat, groups["bc"], groups["jepa"], groups["cql"],
                             5000, np.random.default_rng(0))
assert r["p_value"] > 0.9, r

print("all checks passed")
