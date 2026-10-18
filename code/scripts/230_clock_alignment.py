"""Round 2, WP4 — clock alignment: one internal clock time across our tissues/cell types and all external datasets.

Units: our cell types (31_rhythm_genes limma fits; ZT) and tissue pools; external dataset × tissue × condition
(71_external, r2_external; ZT or CT — CT is treated as ZT-equivalent and any systematic difference is absorbed by the
offset). Genes: canonical core-clock genes (lib/phase.CANONICAL_CLOCK). Per unit and gene: peak φ = atan2(b, a),
signal-to-noise s = A / SE(A) with SE(A)² = (a² se_a² + b² se_b²)/A²; weight w = min(s, 10)² for genes with s >= 1.
(1) Canonical-referenced offset δ = weighted circular mean of (φ − canonical); 95 % CI by resampling genes (2000);
    resultant length R (agreement among genes); determinate if CI half-width <= 3 h and >= 4 genes.
(2) Joint alignment: gene reference phases θ_g and unit offsets δ_u estimated by alternating weighted circular means
    (θ_g from all units; δ_u from all genes), re-anchored so that the weighted mean of θ_g − canonical_g is 0.
Internal time of any phase in a unit = phase − δ (tissue-pooled δ for cell types whose own offset is not determinate).
Outputs (results/r2_phase/): clock_alignment_offsets.csv, clock_gene_reference_phases.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r2_phase"
W = 2 * np.pi / 24
CAN = phase.CANONICAL_CLOCK
rng = np.random.default_rng(3)


def gene_table(d, unit_cols, peak_col):
    d = d[d["gene"].isin(CAN)].copy()
    A = np.hypot(d["a"], d["b"])
    se = np.sqrt((d["a"] ** 2 * d["se_a"] ** 2 + d["b"] ** 2 * d["se_b"] ** 2) / np.maximum(A ** 2, 1e-300))
    d["snr"] = A / se
    d["phi"] = d[peak_col] % 24
    d = d[d["snr"] >= 1]
    d["w"] = np.minimum(d["snr"], 10) ** 2
    d["unit"] = d[unit_cols].astype(str).agg(" | ".join, axis=1)
    return d[unit_cols + ["unit", "gene", "phi", "snr", "w"]]


parts = []
for tissue in ["bm", "cochlea"]:
    d = pd.read_csv(RESULTS / f"31_rhythm_genes/rhythm_limma_cell_type_{tissue}.csv")
    d["dataset"] = "this_atlas"; d["tissue"] = tissue; d["condition"] = "WT"; d["time_type"] = "ZT"
    d = d.rename(columns={"group": "cell_type"})
    parts.append(gene_table(d, ["dataset", "tissue", "cell_type", "condition", "time_type"], "peak_zt"))
for f in sorted(list((RESULTS / "71_external").glob("external_rhythm_*.csv")) +
                list((RESULTS / "r2_external").glob("external_rhythm_*.csv"))):
    d = pd.read_csv(f)
    if f.parent.name == "71_external" and "GSE198770" in f.name:
        continue                                            # superseded by the round-2 GSE198770 preparation
    d["cell_type"] = "bulk"
    parts.append(gene_table(d, ["dataset", "tissue", "cell_type", "condition", "time_type"], "peak_h"))
G = pd.concat(parts, ignore_index=True)
G["canon"] = G["gene"].map(CAN)
# tissue pools of our atlas (all cell types of a tissue as one unit)
pool = G[G["dataset"] == "this_atlas"].copy()
pool["cell_type"] = "tissue_pool"
pool["unit"] = pool[["dataset", "tissue", "cell_type", "condition", "time_type"]].astype(str).agg(" | ".join, axis=1)
G = pd.concat([G, pool], ignore_index=True)


def offset(g, ref):
    dev = phase.circ_diff(g["phi"].to_numpy(), ref)
    return phase.circ_mean(dev % 24, g["w"].to_numpy())


rows = []
for u, g in G.groupby("unit"):
    if len(g) < 2:
        continue
    m, R = offset(g, g["canon"].to_numpy())
    dev = phase.circ_diff(g["phi"].to_numpy(), g["canon"].to_numpy()) % 24
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(g), len(g))
        bs.append(phase.circ_mean(dev[i], g["w"].to_numpy()[i])[0])
    lo, hi, half = phase.phase_ci_draws(np.array(bs), m)
    r0 = g.iloc[0]
    rows.append({"dataset": r0["dataset"], "tissue": r0["tissue"], "unit": r0["cell_type"], "condition": r0["condition"],
                 "time_type": r0["time_type"], "n_genes": len(g), "genes": ";".join(sorted(g["gene"])),
                 "offset_canonical_h": float(phase.circ_diff(m, 0)), "ci_lo": float(phase.circ_diff(lo, 0)),
                 "ci_hi": float(phase.circ_diff(hi, 0)), "ci_halfwidth_h": float(half), "resultant": R,
                 "determinate": bool(half <= 3 and len(g) >= 4)})
off = pd.DataFrame(rows)

# joint alignment (alternating circular means)
delta = {u: 0.0 for u in G["unit"].unique()}
theta = dict(CAN)
for _ in range(50):
    th = {}
    for gname, g in G.groupby("gene"):
        th[gname] = phase.circ_mean((g["phi"].to_numpy() - g["unit"].map(delta).to_numpy()) % 24, g["w"].to_numpy())[0]
    dl = {}
    for u, g in G.groupby("unit"):
        dl[u] = phase.circ_mean((g["phi"].to_numpy() - g["gene"].map(th).to_numpy()) % 24, g["w"].to_numpy())[0]
    shift = phase.circ_mean(phase.circ_diff(np.array([th[k] for k in th]), np.array([CAN[k] for k in th])) % 24)[0]
    th = {k: (v - shift) % 24 for k, v in th.items()}
    dl = {k: (v + shift) % 24 for k, v in dl.items()}
    change = max(abs(float(phase.circ_diff(dl[k], delta[k]))) for k in dl)
    delta, theta = dl, th
    if change < 1e-4:
        break
off["offset_joint_h"] = [float(phase.circ_diff(delta[" | ".join([r.dataset, r.tissue, r.unit, r.condition, r.time_type])], 0))
                         for r in off.itertuples()]
write_csv(off.sort_values(["dataset", "tissue", "unit", "condition"]), MOD, "clock_alignment_offsets.csv",
          "Clock offset of each unit vs canonical core-clock phases (gene bootstrap CI) and from the joint alignment",
          required=["dataset", "tissue", "unit", "condition", "offset_canonical_h", "ci_halfwidth_h", "offset_joint_h"])
ref = pd.DataFrame({"gene": list(theta), "canonical_peak_zt": [CAN[g] for g in theta],
                    "joint_reference_peak": [theta[g] for g in theta],
                    "n_units": [int((G["gene"] == g).sum()) for g in theta]})
write_csv(ref, MOD, "clock_gene_reference_phases.csv", "Core-clock reference phases: canonical vs joint alignment",
          required=["gene", "canonical_peak_zt", "joint_reference_peak"])
pools = off[off["unit"].isin(["tissue_pool", "bulk"])]
log("tissue/dataset offsets: " + "; ".join(f"{r.dataset}:{r.tissue}:{r.condition} {r.offset_canonical_h:+.1f}h"
                                           for r in pools.itertuples()))
