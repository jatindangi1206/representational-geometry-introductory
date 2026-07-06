# substrate-pilot

Round-one smoke test for the decision-atlas project. **Not** the full atlas —
just enough plumbing to answer one falsifiable question end to end:

> Do different training objectives on the *same* decision dataset produce the
> *same* or *different* representational geometry?

Architecture, data, and evaluation are held fixed. The only moving part is the
training objective. If the pipeline runs clean on PointMaze, move to Adroit;
if Adroit also shows a clean decision-specific geometry, that justifies building
the multi-domain atlas.

## The design in one breath

One dataset → one shared encoder → several objectives (BC, CQL, JEPA, + nulls)
→ freeze → extract at one point on one held-out probe set → one metric battery
(CKA, mutual-kNN, participation ratio, jPCA) → calibrate against nulls → apply a
pre-registered success rule.

## Install

```bash
python3.12 -m venv .venv          # 3.10+ works; 3.12 recommended
.venv/bin/pip install -r requirements.txt
```

`minari[hf,hdf5]` is required — both extras, or the dataset download/read
fails. There is **no synthetic fallback anywhere**: a Minari failure is a
hard error, and every report stamps the data provenance it measured.

## Run

```bash
# fast plumbing check (tiny budget, real data)
.venv/bin/python run_pilot.py --quick

# sanity: unit checks for the correctness-critical paths
.venv/bin/python test_pilot.py

# THE real round-one run (GPU machine): 4 objectives x 5 seeds x 20k steps
# on D4RL/pointmaze/umaze-dense-v2. train.device: auto picks up CUDA.
.venv/bin/python run_pilot.py
```

Before trusting any result, check the run log for
`source=minari` and the reward stats line, and `SUMMARY.md` line 2 for the
data provenance stamp.

Outputs land in `results/`:
- `results/encoders/` — frozen encoder checkpoints (`{obj}_seed{n}.pt`)
- `results/features/` — extracted features on the shared probe set (`.npz`)
- `results/report.json` — every metric + null calibration + verdict
- `results/SUMMARY.md` — human-readable summary
- `results/figures/` — similarity heatmaps

You can also run stages individually:

```bash
python train_bc.py   --seed 0
python train_cql.py  --seed 0
python train_jepa.py --seed 0
python extract.py --encoder results/encoders/cql_seed0.pt --out results/features/cql_seed0.npz
python analyze.py
```

## Files

| file | role |
|------|------|
| `config.yaml` | dataset, seeds, encoder shape, objective knobs, success rule |
| `common.py` | seeding, device, obs flattening, checkpoint I/O (shared everywhere) |
| `data.py` | Minari load → standardized MDP → episode split → shared held-out probe |
| `encoder.py` | the **single** shared encoder + generic MLP head |
| `train_bc.py` | behavior cloning (observational baseline) |
| `train_cql.py` | conservative Q-learning (decision / interventional baseline) |
| `train_jepa.py` | latent predictive baseline (self-contained BYOL/EMA world-model) |
| `train_reward.py` | supervised reward prediction (reward-access control) |
| `extract.py` | freeze encoders, extract features on the SAME probe set |
| `metrics.py` | CKA, mutual-kNN, participation ratio, jPCA |
| `nulls.py` | random-init encoder + dimension-matched Gaussian floors |
| `analyze.py` | pairwise comparisons + null calibration + pre-registered verdict |
| `run_pilot.py` | one-command end-to-end driver |

## The two guardrails, enforced in code

1. **One extraction point.** Every objective uses `encoder.py::Encoder` and
   nothing else as its observation backbone; objective-specific parameters live
   in *heads* that consume the latent `z`. We always extract `z`.
2. **One probe distribution.** `data.py::held_out_states` builds a single,
   fixed, ordered held-out state set from the **test** split (never training
   trajectories), preserving trajectory spans for jPCA. `run_pilot.py`
   persists it (with data provenance) to `results/probe.npz`; standalone
   `extract.py` reuses that file, so nothing can silently swap the probe
   under a comparison.

## The pre-registered success rule

Written in `config.yaml` **before** looking at results:

- **Supports** the hypothesis if the decision encoders (CQL, all seeds) are
  separated from the BC/JEPA pack on the outlier metric (mutual-kNN), by a
  one-sided permutation test that shuffles objective labels across all trained
  encoders (`p <= p_threshold`). Under H0 — the objective doesn't shape
  geometry — encoder labels are exchangeable, so this is exact.
- **Weakens** it if all objectives cluster once architecture and dimensionality
  are matched.
- **Inconclusive** if fewer than 2 seeds per objective are available.

Two secondary controls qualify (but never move) the verdict:

- **Reward access**: the same permutation test with `train_reward.py` (pure
  supervised reward regression) as the outlier. If it also separates, a CQL
  separation may reflect *seeing rewards* rather than *credit assignment*.
- **Convergence**: encoders are checkpointed at `train.checkpoint_fracs`;
  the CQL-vs-pack gap is reported per training fraction. A verdict that only
  appears at one fraction is a training-speed artifact, not geometry.

`analyze.py` computes this automatically and writes the verdict to `SUMMARY.md`.

## Scaling up (round two+)

1. Swap `data.dataset_id` to an Adroit Minari id.
2. Add objectives (IQL, TD3+BC, a faithful JEPA from `jepa-wms`) — each just
   needs a new `train_*.py` that reuses `encoder.py`.
3. Only after a clean, null-surviving decision-specific geometry on multiple
   domains: build the multi-domain atlas.

See `REPOS.md` for the reference implementations to lean on at each step.
