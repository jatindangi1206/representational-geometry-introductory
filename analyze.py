"""Geometry battery over extracted features + the pre-registered verdict.
Reads results/features/*.npz, writes report.json, SUMMARY.md, figures/."""
from __future__ import annotations
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

from common import load_config, dump_json, get_logger
import metrics as M
import nulls as N

log = get_logger("analyze")

OBJECTIVES = ["bc", "cql", "jepa", "reward"]


# Final encoders — the only ones that enter the pre-registered test.
LABEL_RE = re.compile(r"^(bc|cql|jepa|reward|null_random)_seed\d+$|^null_gaussian$")
# Mid-training checkpoints — used only by the convergence control.
CKPT_RE = re.compile(r"^(bc|cql|jepa|reward)_seed\d+_ckpt\d+$")


def load_feature_files(feat_dir: str, pattern: re.Pattern = LABEL_RE):
    """label -> (features, spans) for files matching `pattern`. Anything else
    is skipped: a stray .npz would silently enter the permutation test as an
    extra (possibly duplicate) encoder."""
    out = {}
    for path in sorted(glob.glob(os.path.join(feat_dir, "*.npz"))):
        label = os.path.splitext(os.path.basename(path))[0]
        if not pattern.match(label):
            if not (LABEL_RE.match(label) or CKPT_RE.match(label)):
                log.warning(f"skipping {path}: name doesn't match the "
                            f"'{{objective}}_seed{{n}}[_ckpt{{f}}].npz' convention")
            continue
        d = np.load(path)
        out[label] = (d["features"], d["spans"])
    return out


def _obj_of(label: str) -> str:
    m = re.match(r"(bc|cql|jepa|reward|null_random|null_gaussian)", label)
    return m.group(1) if m else label


def _seed_of(label: str) -> int:
    m = re.search(r"seed(\d+)", label)
    return int(m.group(1)) if m else 0


def analyze(cfg: dict, feats: dict, ckpts: dict | None = None) -> dict:
    mcfg = cfg["metrics"]
    acfg = cfg["analyze"]
    outlier_metric = acfg["outlier_metric"]
    k = mcfg["mutual_knn"]["k"]

    labels = list(feats.keys())
    # kNN index per representation, computed once (see metrics.knn_indices).
    knn = {l: M.knn_indices(feats[l][0], k) for l in labels}

    # ---- pairwise similarity matrices (CKA + mutual kNN) ----
    pair = {"cka": {}, "mutual_knn": {}}
    for a in labels:
        for b in labels:
            if a >= b:
                continue
            pair["cka"][f"{a}|{b}"] = M.cka(feats[a][0], feats[b][0],
                                            mcfg["cka"]["kernel"])
            pair["mutual_knn"][f"{a}|{b}"] = M.knn_overlap(knn[a], knn[b])

    # ---- single-representation battery (PR, jPCA) ----
    single = {}
    for a in labels:
        single[a] = M.single_battery(feats[a][0], [tuple(s) for s in feats[a][1]], mcfg)

    # Within-objective seed variation: descriptive context, not tested.
    within = defaultdict(list)
    for obj in OBJECTIVES:
        seeds = sorted([l for l in labels if _obj_of(l) == obj], key=_seed_of)
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                within[obj].append(M.knn_overlap(knn[seeds[i]], knn[seeds[j]]))
    pooled_within = [v for vs in within.values() for v in vs]
    null_seed = N.null_distribution(pooled_within) if pooled_within else \
        {"mean": 0.0, "std": 0.0, "n": 0, "values": []}

    # ---- the outlier test (falsifiable, pre-registered) ----
    verdict = _outlier_test(labels, pair, acfg, cfg["experiment"]["seed"])

    return {
        "labels": labels,
        "pairwise": pair,
        "single": single,
        "within_objective_seed_variation": null_seed,
        "verdict": verdict,
        "convergence": convergence_check(feats, ckpts or {}, k),
        "config_success_rule": {
            "outlier_metric": outlier_metric,
            "n_permutations": acfg["n_permutations"],
            "p_threshold": acfg["p_threshold"],
        },
    }


def convergence_check(feats: dict, ckpts: dict, k: int) -> dict:
    """Convergence control: per-checkpoint similarity to the final encoder,
    and the CQL-vs-pack gap per training fraction. Descriptive only — a gap
    that only appears at one fraction is a training-speed artifact."""
    if not ckpts:
        return {}
    knn = {l: M.knn_indices(X, k) for l, (X, _) in ckpts.items()}
    for l, (X, _) in feats.items():
        if _obj_of(l) in OBJECTIVES:
            knn[l] = M.knn_indices(X, k)

    out = {"knn_to_final": {}, "gap_by_frac": {}}
    by_frac = defaultdict(list)
    for l in ckpts:
        base, frac = l.rsplit("_ckpt", 1)
        if base in feats:
            out["knn_to_final"][l] = M.knn_overlap(knn[l], knn[base])
        by_frac[int(frac)].append(l)

    for frac, ls in sorted(by_frac.items()):
        grp = {o: [l for l in ls if _obj_of(l) == o] for o in ("bc", "jepa", "cql")}
        if all(grp.values()):
            sim = lambda a, b: M.knn_overlap(knn[a], knn[b])
            pack = [sim(b, j) for b in grp["bc"] for j in grp["jepa"]]
            cqlx = [sim(c, x) for c in grp["cql"]
                    for x in grp["bc"] + grp["jepa"]]
            out["gap_by_frac"][f"{frac}%"] = float(np.mean(pack) - np.mean(cqlx))
    return out


def permutation_outlier_test(sim: dict, pack_a: list, pack_b: list,
                             outlier: list, n_perm: int,
                             rng: np.random.Generator) -> dict:
    """One-sided permutation test: is the `outlier` group separated from the
    pack beyond what arbitrary relabeling of encoders produces?

    sim : {"a|b" (sorted labels): similarity} covering every pair of encoders.

    Statistic:  gap = mean sim(pack_a, pack_b) - mean sim(outlier, pack)
                (both terms are cross-group means, so the statistic is
                exchangeable under H0: objective labels don't shape geometry).
    Null: objective labels permuted across all encoders, group sizes kept.
    p uses the standard +1 correction (Phipson & Smyth 2010).
    """
    key = lambda a, b: "|".join(sorted((str(a), str(b))))

    def gap(pa, pb, out):
        pack = [sim[key(x, y)] for x in pa for y in pb]
        outx = [sim[key(o, x)] for o in out for x in list(pa) + list(pb)]
        return float(np.mean(pack) - np.mean(outx)), \
            float(np.mean(pack)), float(np.mean(outx))

    observed, s_pack, s_out = gap(pack_a, pack_b, outlier)
    all_enc = np.array(list(pack_a) + list(pack_b) + list(outlier))
    na, nb = len(pack_a), len(pack_b)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(all_enc)
        g, _, _ = gap(p[:na], p[na:na + nb], p[na + nb:])
        if g >= observed:
            hits += 1
    return {"gap": observed, "s_pack": s_pack, "s_outlier_to_pack": s_out,
            "p_value": float((hits + 1) / (n_perm + 1)),
            "n_permutations": int(n_perm)}


def _outlier_test(labels, pair, acfg, seed) -> dict:
    """Pre-registered rule: CQL separates from the BC/JEPA pack (one-sided),
    tested by permuting objective labels across ALL trained encoders.

    Secondary reward-access control (interpretive, does not move the
    decision): the same test with the reward-prediction arm as the outlier.
    If reward ALSO separates, a CQL separation may reflect reward access
    rather than decision structure."""
    groups = {obj: sorted([l for l in labels if _obj_of(l) == obj], key=_seed_of)
              for obj in OBJECTIVES}
    counts = {o: len(g) for o, g in groups.items()}
    if min(counts[o] for o in ("bc", "cql", "jepa")) < 2:
        return {"decision": "inconclusive", "group_sizes": counts,
                "reason": (f"permutation test needs >=2 seeds per objective, "
                           f"have {counts}; train more seeds")}

    sim = pair[acfg["outlier_metric"]]
    thr = acfg["p_threshold"]
    rng = np.random.default_rng(seed)
    res = permutation_outlier_test(sim, groups["bc"], groups["jepa"],
                                   groups["cql"], acfg["n_permutations"], rng)
    supports = res["gap"] > 0 and res["p_value"] <= thr
    if supports:
        decision = "supports"
        reason = (f"CQL separates from the BC/JEPA pack by {res['gap']:.3f} in "
                  f"{acfg['outlier_metric']} (one-sided permutation "
                  f"p={res['p_value']:.4f} <= {thr}). Decision geometry is a "
                  f"substrate-specific outlier.")
    else:
        decision = "weakens"
        reason = (f"CQL is not separated from the pack (gap {res['gap']:.3f}, "
                  f"permutation p={res['p_value']:.4f} > {thr}); objectives "
                  f"cluster once architecture & dim are matched.")
    out = {"decision": decision, "reason": reason, "group_sizes": counts,
           "p_threshold": thr, **res}

    if counts.get("reward", 0) >= 2:
        rc = permutation_outlier_test(sim, groups["bc"], groups["jepa"],
                                      groups["reward"],
                                      acfg["n_permutations"], rng)
        reward_separates = rc["gap"] > 0 and rc["p_value"] <= thr
        rc["interpretation"] = (
            "reward arm ALSO separates from the pack — a CQL separation may "
            "reflect reward access, not decision structure"
            if reward_separates else
            "reward arm stays with the pack — reward access alone does not "
            "reshape geometry; a CQL separation reflects decision structure")
        out["reward_control"] = rc
        if supports and reward_separates:
            out["reason"] += (" CAVEAT: the reward-prediction control also "
                              "separates; reward access is a live confound.")
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def attach_provenance(report: dict, out_dir: str) -> None:
    """Stamp the probe's data provenance into the report so no result is
    ever ambiguous about what it measured."""
    p = os.path.join(out_dir, "probe.npz")
    if not os.path.exists(p):
        report["data"] = {"source": "UNKNOWN", "warning": "no probe.npz found"}
        return
    d = np.load(p)
    report["data"] = {"source": str(d["source"]),
                      "dataset_id": str(d["dataset_id"]),
                      "n_probe_states": int(d["states"].shape[0])}


def write_summary(report: dict, path: str):
    v = report["verdict"]
    data = report.get("data", {})
    lines = ["# Round-one result\n",
             f"*data: {data.get('source', 'UNKNOWN')} "
             f"({data.get('dataset_id', '?')}, "
             f"{data.get('n_probe_states', '?')} probe states)*\n\n",
             f"**Verdict: {v['decision'].upper()}**\n",
             f"{v.get('reason','')}\n",
             "\n## Outlier test (pre-registered)\n"]
    for k in ("s_pack", "s_outlier_to_pack", "gap", "p_value",
              "p_threshold", "n_permutations", "group_sizes"):
        if k in v:
            val = v[k]
            txt = f"{val:.4f}" if isinstance(val, float) else str(val)
            lines.append(f"- {k}: {txt}\n")
    rc = v.get("reward_control")
    if rc:
        lines.append("\n### Reward-access control (secondary)\n")
        lines.append(f"- gap: {rc['gap']:.4f}, p: {rc['p_value']:.4f}\n")
        lines.append(f"- {rc['interpretation']}\n")
    conv = report.get("convergence") or {}
    if conv.get("gap_by_frac"):
        lines.append("\n## Convergence control (geometry stability)\n")
        gbf = {f: round(g, 4) for f, g in conv["gap_by_frac"].items()}
        lines.append(f"- CQL-vs-pack gap by training fraction "
                     f"(pre-registered p only at 100%): {gbf}\n")
        for l, s in sorted(conv["knn_to_final"].items()):
            lines.append(f"- {l} vs its final encoder: {s:.3f}\n")
    lines.append("\n## Effective dimensionality & rotational structure\n")
    for label, m in report["single"].items():
        lines.append(f"- **{label}**: PR={m['participation_ratio']:.2f}, "
                     f"rot_r2={m['rotational_r2']:.3f}, "
                     f"rot_frac={m['rot_frac']:.3f}, freq={m['top_freq']:.3f}\n")
    lines.append("\n## Pairwise mutual-kNN (local structure)\n")
    for key, val in report["pairwise"]["mutual_knn"].items():
        lines.append(f"- {key}: {val:.3f}\n")
    lines.append("\n## Pairwise CKA (global structure)\n")
    for key, val in report["pairwise"]["cka"].items():
        lines.append(f"- {key}: {val:.3f}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def make_figures(report: dict, feats: dict, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log.warning(f"matplotlib unavailable ({e}); skipping figures.")
        return
    os.makedirs(out_dir, exist_ok=True)
    labels = report["labels"]
    for metric in ("mutual_knn", "cka"):
        n = len(labels)
        Mtx = np.eye(n)
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j:
                    continue
                key = "|".join(sorted([a, b]))
                Mtx[i, j] = report["pairwise"][metric].get(key, np.nan)
        fig, ax = plt.subplots(figsize=(1.2 + 0.6 * n, 1.2 + 0.6 * n))
        im = ax.imshow(Mtx, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{metric} similarity")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"heatmap_{metric}.png"), dpi=150)
        plt.close(fig)
    log.info(f"figures -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = cfg["analyze"]["out_dir"]
    feat_dir = os.path.join(out_dir, "features")
    feats = load_feature_files(feat_dir)
    if not feats:
        raise SystemExit(f"No feature files in {feat_dir}. Run run_pilot.py first.")
    ckpts = load_feature_files(feat_dir, CKPT_RE)
    report = analyze(cfg, feats, ckpts)
    attach_provenance(report, out_dir)
    dump_json(os.path.join(out_dir, "report.json"), report)
    write_summary(report, os.path.join(out_dir, "SUMMARY.md"))
    make_figures(report, feats, os.path.join(out_dir, "figures"))
    log.info(f"verdict: {report['verdict']['decision'].upper()}")
    log.info(report["verdict"].get("reason", ""))


if __name__ == "__main__":
    main()
