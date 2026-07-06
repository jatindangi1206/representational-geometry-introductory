# Repos & tools for the rep-geo pipeline

Curated for the "same dataset, different objective → same or different representational
geometry?" question. Grouped by the role each plays in the pilot. Starred (⭐) items are
the ones the smoke-test pipeline actually leans on.

## 1. The anchor — Platonic Representation & its critics

- ⭐ [minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep) — the paper you started from. Its `metrics.py` (mutual-kNN, CKA, and the alignment sweep) is the reference implementation to copy conventions from.
- ⭐ [mlbio-epfl/aristotelian](https://github.com/mlbio-epfl/aristotelian) — "Revisiting the Platonic Representation Hypothesis: An Aristotelian View." *This is the most important one for you.* It adds **calibrated** similarity metrics with statistical guarantees and shows that global spectral convergence (CKA) largely vanishes after calibration while local neighborhood (mutual-kNN) survives. This is exactly the null-calibration discipline your proposal demands. [Project page](https://brbiclab.epfl.ch/projects/aristotelian/).
- [nacloos/similarity-repository](https://github.com/nacloos/similarity-repository) — a standardized, unified collection of representational-similarity metric implementations (CKA, RSA, Procrustes, CCA families) with consistent APIs. Good for cross-checking your `metrics.py` against canonical implementations.

## 2. Representational similarity metrics (the geometry battery)

- [js-d/sim_metric](https://github.com/js-d/sim_metric) — reference CKA / CCA / Procrustes.
- [google-research/google-research/representation_similarity](https://github.com/google-research/google-research/tree/master/representation_similarity) — Kornblith et al. original CKA notebook (the canonical linear + RBF CKA).
- [nacloos/similarity-repository](https://github.com/nacloos/similarity-repository) — (again) the broadest single home for these.
- [rsatoolbox (rsagroup/rsatoolbox)](https://github.com/rsagroup/rsatoolbox) — mature RSA library from the neuroscience side; useful if you want RDM-based RSA done properly rather than a hand-rolled version.

## 3. Dynamical / population-geometry metrics

- ⭐ [bantin/jPCA](https://github.com/bantin/jPCA) — Python jPCA, mirrors Churchland's MATLAB pack. Use this (or port its core) for the rotational-structure arm. Input format: list of `T x N` arrays.
- [alexmorley/jPCA](https://github.com/alexmorley/jPCA) — Julia version, useful as a correctness reference.
- [buschman-lab/RotationalDynamics](https://github.com/buschman-lab/RotationalDynamics) — RNNs with rotational dynamics; good for building a *known-rotation* sanity check for your jPCA code.
- [NevVerVer/neural-dynamics-gyration](https://github.com/NevVerVer/neural-dynamics-gyration) — travelling-waves account of rotational dynamics; worth reading before over-interpreting a jPCA rotation.
- Participation ratio has no single canonical repo — it's a few lines (`(Σλ)² / Σλ²` over the covariance eigenvalues). Implemented directly in this pilot's `metrics.py`.

## 4. JEPA / predictive-latent objectives

- [facebookresearch/jepa](https://github.com/facebookresearch/jepa) — official V-JEPA (video). Reference for the EMA-target + predictor recipe.
- [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) — JEPA **World Models** (DINO-WM, V-JEPA-2-AC baselines). Closest to a *decision/world-model* JEPA, most relevant to your RL setting.
- [keon/jepa](https://github.com/keon/jepa) — minimal I-JEPA; good to read for the smallest correct version.
- [yunusskeete/jepas](https://github.com/yunusskeete/jepas) — demo JEPA world-model implementations for research.
- Note: the pilot's `train_jepa.py` is a **self-contained latent MLP predictor** (BYOL/EMA-style, same backbone as BC/CQL) so architecture is held fixed. The repos above are for when you want a more faithful JEPA in a later round.

## 5. Offline RL (the BC / CQL objective arm)

- ⭐ [takuseno/d3rlpy](https://github.com/takuseno/d3rlpy) — offline deep RL with sklearn-style API, custom `EncoderFactory`, `CQLConfig`, and `get_minari(...)`. The pilot ships a self-contained CQL so the encoder is provably identical across objectives, but d3rlpy is the drop-in if you'd rather use a battle-tested CQL.
- [Farama-Foundation/Minari](https://github.com/Farama-Foundation/Minari) — the offline-RL dataset API (PointMaze, Adroit, etc.), Gymnasium-style, episode-based. [Docs](https://minari.farama.org/).
- [Farama-Foundation/D4RL / minari datasets](https://minari.farama.org/main/datasets/) — dataset catalog (find the exact PointMaze / Adroit IDs here).
- [tinkoff-ai/CORL](https://github.com/tinkoff-ai/CORL) — clean single-file offline-RL baselines (BC, CQL, IQL, TD3+BC). Excellent to sanity-check your CQL numbers against.

## 6. Environments to scale into

- [Farama-Foundation/Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) — PointMaze + Adroit environments behind the Minari datasets.

---

### Suggested reading order
1. Skim **aristotelian** (calibration is your guardrail) and **platonic-rep** `metrics.py`.
2. Read **bantin/jPCA** input format before wiring the dynamical arm.
3. Keep **CORL** and **d3rlpy** open as reference implementations for BC/CQL numbers.
4. Save **jepa-wms** for round two, when you want a faithful decision-JEPA.
