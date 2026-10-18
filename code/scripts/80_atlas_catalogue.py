"""Module 6.1 — circadian cell-type atlas resource.

From the tiered gene-rhythm results (cell_type level):
  rhythmic_catalogue.csv     every Tier-1/Tier-2 gene × tissue × cell type with phase/amplitude/CI, plus how many
                             cell types (per tissue) it is rhythmic in and whether it is cell-type-restricted
                             (Tier 1 in one type, p > 0.2 in all other types where it is tested)
  celltype_chronotype.csv    per tissue × cell type: number of Tier-1/2 genes, circular mean peak ZT, mean resultant
                             length, Rayleigh p, fraction peaking in light (ZT0–12) vs dark phase, hourly histogram
  rhythmic_marker_genes.csv  cell-type-restricted rhythmic genes (top by amplitude)
  Lineage-foreign marker flag (annotation only, no covariate): a rhythmic gene that is a highly specific marker of
  another gold-standard cell type (00_audit/label_markers: log2FC >= 3, adj. p < 1e-10, detected in >= 30 % of the
  marker type and <= 10 % of other cells) is listed in foreign_marker_of; foreign_marker_phase_match = its peak lies
  within 3 h of that cell type's abundance peak (50_abundance frame A, p < 0.1) — such rhythms most likely track the
  sample's content of that cell type (composition-linked transcripts) rather than regulation in the host cell type.
  foreign_marker_summary.csv counts flagged genes per tissue × cell type.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "80_atlas"
W = 2 * np.pi / 24
dfs = [pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv") for t in ["cochlea", "bm"]]
allg = pd.concat(dfs, ignore_index=True)
rh = allg[allg["tier"].isin(["tier1", "tier2"])].copy()
n_types = rh.groupby(["tissue", "gene"])["group"].nunique().rename("n_celltypes_rhythmic_in_tissue")
rh = rh.join(n_types, on=["tissue", "gene"])
other_p = allg.groupby(["tissue", "gene"]).apply(lambda g: g.set_index("group")["p"].to_dict())


def restricted(row):
    if row["tier"] != "tier1":
        return False
    ps = other_p.get((row["tissue"], row["gene"]), {})
    others = [p for grp, p in ps.items() if grp != row["group"]]
    return len(others) > 0 and min(others) > 0.2


rh["celltype_restricted"] = rh.apply(restricted, axis=1)

mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 3) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.3) & (mk["pct_nz_reference"] <= 0.1)]
marker_of = mk.groupby("names")["cell_type"].agg(lambda x: sorted(set(x))).to_dict()
ab = pd.read_csv(RESULTS / "50_abundance/abundance_rhythm_frames.csv")
ab = ab[(ab["frame"] == "A_clr_all") & (ab["p"] < 0.1)].set_index(["tissue", "feature"])["peak_zt"].to_dict()


def foreign(row):
    src = [c for c in marker_of.get(row["gene"], []) if c != row["group"]]
    if not src or row["group"] in marker_of.get(row["gene"], []):
        return "", False
    match = any((row["tissue"], c) in ab and abs((row["peak_zt"] - ab[(row["tissue"], c)] + 12) % 24 - 12) < 3 for c in src)
    return ";".join(src), match


fm = rh.apply(foreign, axis=1, result_type="expand")
rh["foreign_marker_of"], rh["foreign_marker_phase_match"] = fm[0], fm[1].astype(bool)
cols = ["tissue", "group", "gene", "tier", "mean_logCPM", "AveExpr", "ptt_fc", "amp_log2", "peak_zt", "peak_ci_low",
        "peak_ci_high", "amp_ci_low", "amp_ci_high", "p", "q_BH", "q_perm", "edgeR_p", "rain_p", "jtk_p",
        "loto_phase_sd_h", "n_celltypes_rhythmic_in_tissue", "celltype_restricted", "foreign_marker_of",
        "foreign_marker_phase_match"]
cat = rh[[c for c in cols if c in rh.columns]].rename(columns={"group": "cell_type"})
write_csv(cat.sort_values(["tissue", "cell_type", "tier", "p"]), MOD, "rhythmic_catalogue.csv",
          "Tier-1/2 rhythmic genes per tissue × cell type with phase, amplitude, CIs and cell-type restriction",
          required=["tissue", "cell_type", "gene", "tier", "ptt_fc", "peak_zt"])
fs = cat.assign(flag=cat["foreign_marker_of"] != "").groupby(["tissue", "cell_type"]).agg(
    n_rhythmic=("gene", "size"), n_foreign_marker=("flag", "sum"),
    n_foreign_marker_phase_match=("foreign_marker_phase_match", "sum")).reset_index()
fs["frac_foreign_marker"] = fs["n_foreign_marker"] / fs["n_rhythmic"]
write_csv(fs, MOD, "foreign_marker_summary.csv", "Rhythmic genes that are specific markers of another cell type",
          required=["tissue", "cell_type", "n_rhythmic", "n_foreign_marker", "n_foreign_marker_phase_match"])
markers = cat[cat["celltype_restricted"] & (cat["foreign_marker_of"] == "")].sort_values("ptt_fc", ascending=False).groupby(["tissue", "cell_type"]).head(25)
write_csv(markers, MOD, "rhythmic_marker_genes.csv", "Cell-type-restricted rhythmic genes (top 25 by amplitude)",
          required=["tissue", "cell_type", "gene", "peak_zt"])

rows = []
for (t, c), g in cat.groupby(["tissue", "cell_type"]):
    ang = W * g["peak_zt"].to_numpy()
    n = len(ang)
    R = np.abs(np.exp(1j * ang).mean())
    z = n * R ** 2
    p_ray = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)) if n > 1 else np.nan
    hist = np.histogram(g["peak_zt"], bins=np.arange(0, 25, 1))[0]
    rows.append({"tissue": t, "cell_type": c, "n_rhythmic": n, "n_tier1": int((g["tier"] == "tier1").sum()),
                 "circ_mean_peak_zt": (np.angle(np.exp(1j * ang).mean()) / W) % 24, "mean_resultant_length": R,
                 "rayleigh_z": z, "rayleigh_p": p_ray, "frac_peak_light_zt0_12": float((g["peak_zt"] < 12).mean()),
                 **{f"n_peak_zt{h:02d}": int(hist[h]) for h in range(24)}})
write_csv(pd.DataFrame(rows), MOD, "celltype_chronotype.csv",
          "Peak-phase distribution (chronotype) of rhythmic genes per tissue × cell type",
          required=["tissue", "cell_type", "n_rhythmic", "circ_mean_peak_zt", "rayleigh_p"])
log("done")
