"""Round 2, WP2 — downstream summaries of the calibrated relaxed cochlear tier (Tier C, 211); exploratory.

(a) Catalogue: every Tier C call with peak ZT / clock time, peak-to-trough FC, supports (edgeR / external cochlea /
    BM), tissue-pattern-like flag, number of cochlear cell types in which the gene is called, cell-type restriction
    (called here, p > 0.2 in every other cochlear type), lineage-foreign marker flag (gene is a specific marker of
    another cell type: 00_audit/label_markers.csv, log2FC >= 2.5, adj. p < 1e-10, >= 40 % in-type, <= 10 % elsewhere),
    and the cell type's empirical FDR under the selected rule.
(b) Per cell type: number of calls, without pattern-like genes, phase distribution (circular mean, resultant,
    Rayleigh p) and the share in each 6-h bin.
(c) Hearing / chronopharmacology: Tier C calls among curated hearing-relevant genes (resources/curated_sets.yaml)
    with ±3 h peak / trough windows in ZT and clock time.
Outputs (results/r2_cochlea/): tierC_catalogue.csv, tierC_celltype_summary.csv, tierC_hearing_windows.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r2_cochlea"
t = pd.read_csv(RESULTS / "r2_cochlea/cochlea_relaxed_tier.csv")
t["tierC"] = t["tierC"].astype(str) == "True"
tc = t[t["tierC"]].copy()
n_types = tc.groupby("gene")["cell_type"].nunique().rename("n_cochlear_types_called")
tc = tc.join(n_types, on="gene")
pmap = t.groupby("gene").apply(lambda g: g.set_index("cell_type")["p"].to_dict())
tc["celltype_restricted"] = [all(p > 0.2 for c, p in pmap.get(g, {}).items() if c != ct) and len(pmap.get(g, {})) > 1
                             for g, ct in zip(tc["gene"], tc["cell_type"])]
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 2.5) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.4) & (mk["pct_nz_reference"] <= 0.1)]
marker_of = mk.groupby("names")["cell_type"].apply(lambda s: ";".join(sorted(set(s))))
tc["marker_of"] = tc["gene"].map(marker_of).fillna("")
tc["lineage_foreign_marker"] = [bool(m) and ct not in m.split(";") for m, ct in zip(tc["marker_of"], tc["cell_type"])]
tc["peak_clock"] = phase.zt_to_clock(tc["peak_zt"])
keep = ["cell_type", "gene", "peak_zt", "peak_clock", "ptt_fc", "p", "edgeR_p", "loto_sd", "s_edgeR", "s_external", "s_bm",
        "ext_p", "bm_p", "tissue_pattern_like", "pattern_r", "n_cochlear_types_called", "celltype_restricted",
        "marker_of", "lineage_foreign_marker", "rule_selected", "celltype_empirical_fdr", "tier", "rain_p", "jtk_p", "lr_p"]
cat = tc[[c for c in keep if c in tc.columns]].rename(columns={"tier": "primary_tier"}).sort_values(["cell_type", "p"])
write_csv(cat, MOD, "tierC_catalogue.csv", "Catalogue of calibrated Tier C calls (cochlea) with supports, flags and FDR",
          required=["cell_type", "gene", "peak_zt", "peak_clock", "ptt_fc", "p"])

rows = []
for ct, g in tc.groupby("cell_type"):
    m, R = phase.circ_mean(g["peak_zt"])
    bins = pd.cut(g["peak_zt"] % 24, [0, 6, 12, 18, 24], right=False, labels=["ZT0-6", "ZT6-12", "ZT12-18", "ZT18-24"])
    rows.append({"cell_type": ct, "n_calls": len(g), "n_not_pattern_like": int((~g["tissue_pattern_like"].astype(bool)).sum()),
                 "n_external_or_bm_supported": int((g["s_external"].astype(bool) | g["s_bm"].astype(bool)).sum()),
                 "celltype_empirical_fdr": float(g["celltype_empirical_fdr"].iloc[0]),
                 "circ_mean_peak_zt": m, "resultant": R, "rayleigh_p": phase.rayleigh(g["peak_zt"].to_numpy())[1],
                 **{f"share_{b}": float((bins == b).mean()) for b in ["ZT0-6", "ZT6-12", "ZT12-18", "ZT18-24"]},
                 "n_restricted": int(g["celltype_restricted"].sum()), "n_foreign_markers": int(g["lineage_foreign_marker"].sum())})
write_csv(pd.DataFrame(rows).sort_values("n_calls", ascending=False), MOD, "tierC_celltype_summary.csv",
          "Tier C per cochlear cell type: calls, supports, empirical FDR and phase distribution",
          required=["cell_type", "n_calls", "celltype_empirical_fdr", "circ_mean_peak_zt"])

SC = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))["hearing"]
membership = {}
for s, genes in SC.items():
    for g in genes:
        membership.setdefault(g, []).append(s)
h = tc[tc["gene"].isin(membership)].copy()
h["hearing_sets"] = h["gene"].map(lambda g: ";".join(membership[g]))
h["trough_zt"] = (h["peak_zt"] + 12) % 24
for k in ["peak", "trough"]:
    h[f"{k}_window_zt"] = h[f"{k}_zt"].map(lambda z: f"ZT{(z - 3) % 24:04.1f}–ZT{(z + 3) % 24:04.1f}")
    h[f"{k}_window_clock"] = h[f"{k}_zt"].map(lambda z: f"{float(phase.zt_to_clock(z - 3)):04.1f}–{float(phase.zt_to_clock(z + 3)):04.1f} h")
cols = ["cell_type", "gene", "hearing_sets", "peak_zt", "peak_clock", "trough_zt", "peak_window_zt", "peak_window_clock",
        "trough_window_zt", "trough_window_clock", "ptt_fc", "p", "tissue_pattern_like", "s_external", "s_bm",
        "celltype_empirical_fdr"]
write_csv(h[cols].sort_values(["gene", "cell_type"]), MOD, "tierC_hearing_windows.csv",
          "Tier C calls among hearing-relevant genes with peak/trough windows (ZT and clock time)",
          required=["cell_type", "gene", "hearing_sets", "peak_zt", "trough_zt"])
log(f"Tier C: {len(tc)} calls in {tc['cell_type'].nunique()} cell types; hearing genes {len(h)}")
