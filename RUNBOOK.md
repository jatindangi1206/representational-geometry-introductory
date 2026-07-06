# Runbook

## Setup (once)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python test_pilot.py        # unit checks, ~30s, no network
.venv/bin/python run_pilot.py --quick # tiny end-to-end check, ~2 min
.venv/bin/python run_pilot.py         # full run: 4 objectives x 5 seeds x 20k steps
```

The dataset downloads automatically on first run (to `~/.minari/datasets/`),
then is cached. No separate download command. GPU is picked up automatically
(`train.device: auto`).

## Pipeline

load dataset → split episodes → save shared probe (`probe.npz`) →
train bc/cql/jepa/reward on every seed (checkpoints at 25/50/100%) →
extract features on the probe → nulls → metrics → permutation test → verdict.

## Files

| file | does |
|------|------|
| `config.yaml` | every knob: dataset, seeds, encoder, budgets, success rule |
| `common.py` | seeding, device, config load, obs flattening, checkpoint I/O |
| `data.py` | Minari load → Episodes → split → buffer → held-out probe |
| `encoder.py` | the single shared encoder + generic MLP head |
| `train_bc.py` | behavior cloning (observational arm) |
| `train_cql.py` | conservative Q-learning (decision arm) |
| `train_jepa.py` | latent world model (predictive arm) |
| `train_reward.py` | reward regression (reward-access control) |
| `extract.py` | frozen encoder → features on the shared probe |
| `metrics.py` | CKA, mutual kNN, participation ratio, jPCA |
| `nulls.py` | random-init encoder + Gaussian floor |
| `analyze.py` | pairwise battery, permutation test, controls, verdict |
| `run_pilot.py` | runs all of the above in order |
| `test_pilot.py` | unit checks for the correctness-critical paths |

## Outputs (`results/`)

- `SUMMARY.md` — verdict + controls, human-readable. Line 2 = data provenance.
- `report.json` — every number.
- `probe.npz` — the shared probe (do not delete between train and extract).
- `encoders/`, `features/`, `figures/`.

## Debugging

| symptom | fix |
|---------|-----|
| `Could not load Minari dataset` / download error | first run needs internet; check id against `minari.list_remote_datasets()`; needs `minari[hf,hdf5]` extras |
| `No module named h5py / huggingface_hub` | `pip install "minari[hf,hdf5]"` |
| provenance says wrong dataset | delete `results/probe.npz`, rerun `run_pilot.py` |
| verdict INCONCLUSIVE | fewer than 2 seeds per objective — add to `experiment.extra_seeds` |
| want analysis only (models already trained) | `.venv/bin/python analyze.py` |
| rerun one arm | `.venv/bin/python train_cql.py --seed 3` then `extract.py --encoder ... --out ...` then `analyze.py` |
| CUDA OOM (unlikely, models are tiny) | set `train.device: cpu` or lower `train.batch_size` |
| results look off after a config change | delete `results/` and rerun — stale features/probe mix runs |

Before trusting any result: log line `source=minari` + `SUMMARY.md` line 2.
