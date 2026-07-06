"""Data layer: load ONE Minari dataset, standardize episodes, split, and
expose a training buffer + the single held-out probe set."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from common import flatten_obs_batch


@dataclass
class Episode:
    observations: np.ndarray       # (T, obs_dim)
    actions: np.ndarray            # (T, act_dim)
    rewards: np.ndarray            # (T,)
    dones: np.ndarray              # (T,) bool  episode boundary (terminated | truncated)
    terminations: np.ndarray       # (T,) bool  TRUE terminals only — the Bellman cut
    next_observations: np.ndarray  # (T, obs_dim)
    episode_id: int
    timestep: np.ndarray           # (T,) int

    @property
    def length(self) -> int:
        return self.observations.shape[0]


def _standardize_minari_episode(ep, ep_id: int) -> Episode:
    # Minari stores one extra observation (the terminal one).
    if isinstance(ep.observations, (dict, np.ndarray)):
        obs_all = _flatten_maybe_dict(ep.observations)
    else:
        obs_all = flatten_obs_batch(ep.observations)
    actions = np.asarray(ep.actions, dtype=np.float32)
    rewards = np.asarray(ep.rewards, dtype=np.float32).ravel()
    terminations = np.asarray(ep.terminations).ravel().astype(bool)
    truncations = np.asarray(ep.truncations).ravel().astype(bool)

    T = actions.shape[0]
    observations = obs_all[:T]
    next_observations = obs_all[1:T + 1] if obs_all.shape[0] >= T + 1 else obs_all[:T]
    dones = (terminations | truncations)[:T]
    return Episode(observations, actions, rewards, dones, terminations[:T],
                   next_observations, ep_id, np.arange(T, dtype=np.int64))


def _flatten_maybe_dict(obs):
    if isinstance(obs, dict):
        from common import FLATTEN_KEYS
        keys = [k for k in FLATTEN_KEYS if k in obs] or sorted(obs)
        parts = [np.asarray(obs[k], dtype=np.float32).reshape(len(obs[k]), -1)
                 for k in keys]
        return np.concatenate(parts, axis=1)
    return np.asarray(obs, dtype=np.float32)


def load_episodes(dataset_id: str, max_episodes: Optional[int] = None):
    """Load a Minari dataset. Any failure is fatal — there is no synthetic
    fallback; a real run must never continue on fake data."""
    import minari
    ds = minari.load_dataset(dataset_id, download=True)
    episodes = []
    for i, ep in enumerate(ds.iterate_episodes()):
        episodes.append(_standardize_minari_episode(ep, i))
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    meta = {"source": "minari", "dataset_id": dataset_id,
            "n_episodes": len(episodes),
            "obs_dim": int(episodes[0].observations.shape[1]),
            "act_dim": int(episodes[0].actions.shape[1])}
    return episodes, meta


def split_episodes(episodes, split: dict, seed: int = 0):
    """Deterministic episode-level train/val/test split."""
    idx = np.arange(len(episodes))
    np.random.default_rng(seed).shuffle(idx)
    n = len(idx)
    n_tr = int(split["train"] * n)
    n_va = int(split["val"] * n)
    pick = lambda ii: [episodes[i] for i in ii]
    return pick(idx[:n_tr]), pick(idx[n_tr:n_tr + n_va]), pick(idx[n_tr + n_va:])


class TransitionBuffer:
    """Episodes flattened to transition arrays. sample() yields
    (obs, act, rew, next_obs, term); term marks TRUE terminals only —
    Bellman targets must bootstrap through truncations."""

    def __init__(self, episodes):
        self.obs = np.concatenate([e.observations for e in episodes], axis=0)
        self.act = np.concatenate([e.actions for e in episodes], axis=0)
        self.rew = np.concatenate([e.rewards for e in episodes], axis=0)
        self.next_obs = np.concatenate([e.next_observations for e in episodes], axis=0)
        self.term = np.concatenate([e.terminations for e in episodes], axis=0).astype(np.float32)
        self.n = self.obs.shape[0]
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.act.shape[1]

    def sample(self, batch_size: int, rng: np.random.Generator):
        i = rng.integers(0, self.n, size=batch_size)
        return (self.obs[i], self.act[i], self.rew[i],
                self.next_obs[i], self.term[i])


def held_out_states(test_episodes, n_states: int, seed: int = 0):
    """The single shared probe: a fixed, ordered state set from the test
    split, with (start, length) spans so jPCA can rebuild trajectories.
    Whole trajectories are sampled so spans stay intact."""
    chunks, spans, cursor = [], [], 0
    for e in test_episodes:
        chunks.append(e.observations)
        spans.append((cursor, e.length))
        cursor += e.length
    all_states = np.concatenate(chunks, axis=0)

    if all_states.shape[0] <= n_states:
        idx = np.arange(all_states.shape[0])
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(spans))
        keep, total = [], 0
        for j in order:
            s, ln = spans[j]
            if total + ln > n_states and keep:
                break
            keep.append(j)
            total += ln
        keep.sort()
        idx_list, new_spans, c = [], [], 0
        for j in keep:
            s, ln = spans[j]
            idx_list.append(np.arange(s, s + ln))
            new_spans.append((c, ln))
            c += ln
        idx = np.concatenate(idx_list)
        spans = new_spans

    return all_states[idx].astype(np.float32), spans
