"""Module 3c/3d — cell-type-specific (within-sample-centred) pathway rhythms, pathway phase map, TF activity–mRNA lag.

(1) Within-sample centring: a tissue-wide per-sample shift (technical, e.g. depth-related transformation bias, or a
    systemic signal — with one pooled sample per ZT the two cannot be separated) moves a pathway's score in every cell
    type of that sample. Subtracting, for each tissue × resource × pathway × sample, the median score over the
    tissue's primary cell types leaves cell-type-specific deviations, which are re-tested (cosinor, 60-class
    permutation, LOTO; BH within tissue × cell type × resource; robust = q < 0.1 & p_perm <= 0.05 & LOTO phase
    SD < 3 h, as in 41). The removed tissue-wide component (per-sample median) is tested as pseudo cell type
    'tissue_median'. Run on the primary and, if present, the depth-matched scores of 41.
(2) Phase map: robust pathways of every version with their peak ZT; per version × tissue × resource × pathway the
    circular mean and resultant length of peak phases over cell types with p < 0.05 and a Rayleigh test — a common
    phase across cell types indicates a systemic (or sample-level) driver; per version × tissue × resource the share of
    robust pathways in the modal 3-h phase bin (high values flag sample-level effects).
(3) TF activity vs mRNA: CollecTRI ULM activity rhythm (41, primary) vs the TF's own mRNA rhythm (33) in the same
    cell type; lag = activity peak − mRNA peak in (−12, 12] h for pairs with both p < 0.05; circular mean lag and
    Rayleigh test per tissue (core-clock TFs separately).
Outputs (results/43_pathway_phase/): pathway_rhythm_centred.csv, pathway_rhythm_centred_depth_matched.csv,
pathway_robust_counts.csv, pathway_phase_map.csv, pathway_phase_consistency.csv, tf_activity_mrna_lag.csv,
tf_lag_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "43_pathway_phase"
W = 2 * np.pi / 24
CLOCK_TF = {"Clock", "Npas2", "Arntl", "Nr1d1", "Nr1d2", "Rora", "Rorc", "Dbp", "Tef", "Hlf", "Nfil3", "Bhlhe40",
            "Bhlhe41", "Cry1", "Cry2", "Per1", "Per2"}
incl = pd.read_csv(RESULTS / "30_pseudobulk/groups_included.csv")
primary = incl[(incl["level"] == "cell_type") & (incl["status"] == "primary")]


def rayleigh(h):
    n = len(h)
    if n < 3:
        return np.nan, np.nan, np.nan
    z = np.exp(1j * W * np.asarray(h)).mean()
    R = abs(z)
    Rn = n * R
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))
    return (np.angle(z) / W) % 24, R, float(min(max(p, 0.0), 1.0))


def rhythm_table(Y, zt, cols, meta):
    o = np.argsort(zt)
    Y, zt = Y[o], zt[o]
    f = cosinor.fit(Y, zt)
    _, _, pp, _ = cosinor.permutation_test(Y, zt)
    lo = cosinor.loto(Y, zt)
    out = pd.DataFrame({"resource": [c[0] for c in cols], "pathway": [c[1] for c in cols], "amplitude": f["amp"],
                        "peak_zt": f["peak"], "r2": f["r2"], "F": f["F"], "p": f["p"], "p_perm": pp,
                        "loto_phase_sd_h": lo["loto_phase_sd_h"], "loto_min_amp_ratio": lo["loto_min_amp_ratio"]})
    for k, v in meta.items():
        out.insert(0, k, v)
    return out


def centred(scores, version):
    rows = []
    for tissue, s in scores.groupby("tissue"):
        prim = set(primary.loc[primary["tissue"] == tissue, "group"])
        med = (s[s["cell_type"].isin(prim)].groupby(["resource", "pathway", "sample_id"])["score"].median()
               .rename("tissue_median").reset_index())
        s = s.merge(med, on=["resource", "pathway", "sample_id"])
        s["centred"] = s["score"] - s["tissue_median"]
        series = [(ct, g, "centred") for ct, g in s.groupby("cell_type")]
        series.append(("tissue_median", s.drop_duplicates(["resource", "pathway", "sample_id"]), "tissue_median"))
        for ct, g, col in series:
            w = g.pivot_table(index=["sample_id", "zt"], columns=["resource", "pathway"], values=col).dropna(axis=1)
            if len(w) < 6 or w.shape[1] == 0:
                continue
            rows.append(rhythm_table(w.to_numpy(), w.index.get_level_values("zt").to_numpy(float), list(w.columns),
                                     {"version": version, "tissue": tissue, "cell_type": ct}))
        log(f"{version} {tissue}: centred rhythm tests done")
    r = pd.concat(rows, ignore_index=True)
    r["q_BH"] = r.groupby(["tissue", "cell_type", "resource"])["p"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    r["robust"] = (r["q_BH"] < 0.1) & (r["p_perm"] <= 0.05) & (r["loto_phase_sd_h"] < 3)
    return r


versions = {"primary": pd.read_csv(RESULTS / "41_pathway/pathway_rhythm.csv").assign(version="primary")}
dm_rh = RESULTS / "41_pathway/pathway_rhythm_depth_matched.csv"
if dm_rh.exists():
    versions["depth_matched"] = pd.read_csv(dm_rh).assign(version="depth_matched")
for name, sfx in [("centred", ""), ("centred_depth_matched", "_depth_matched")]:
    f = RESULTS / f"41_pathway/pathway_scores{sfx}.csv"
    if not f.exists():
        continue
    r = centred(pd.read_csv(f), name)
    write_csv(r, MOD, f"pathway_rhythm_{name}.csv",
              f"Cell-type-specific pathway rhythms after removing the per-sample tissue median ({name})",
              required=["version", "tissue", "cell_type", "resource", "pathway", "peak_zt", "p", "q_BH", "robust"])
    versions[name] = r

cnt_rows, map_rows, cons_rows = [], [], []
for v, r in versions.items():
    for (tissue, res), g in r.groupby(["tissue", "resource"]):
        rb = g[g["robust"] & (g["cell_type"] != "tissue_median")]
        bins = np.floor(rb["peak_zt"] % 24 / 3).astype(int)
        cnt_rows.append({"version": v, "tissue": tissue, "resource": res,
                         "n_tested": int((g["cell_type"] != "tissue_median").sum()), "n_robust": len(rb),
                         "n_celltypes_with_robust": rb["cell_type"].nunique(),
                         "modal_phase_bin_zt": f"ZT{3 * bins.mode().iloc[0]}-{3 * bins.mode().iloc[0] + 3}" if len(rb) else "",
                         "frac_robust_in_modal_bin": float(bins.value_counts(normalize=True).iloc[0]) if len(rb) else np.nan,
                         "tissue_median_n_robust": int(g.loc[g["cell_type"] == "tissue_median", "robust"].sum())})
        map_rows.append(rb[["tissue", "cell_type", "resource", "pathway", "amplitude", "peak_zt", "p", "q_BH"]]
                        .assign(version=v))
        sig = g[(g["p"] < 0.05) & (g["cell_type"] != "tissue_median")]
        robust_pw = set(rb["pathway"])
        for pw, h in sig[sig["pathway"].isin(robust_pw)].groupby("pathway"):
            m, R, p = rayleigh(h["peak_zt"].to_numpy())
            cons_rows.append({"version": v, "tissue": tissue, "resource": res, "pathway": pw,
                              "n_celltypes_p05": len(h), "n_celltypes_robust": int(h["robust"].sum()),
                              "circ_mean_peak_zt": m, "resultant_length": R, "rayleigh_p": p,
                              "cell_types": ";".join(h["cell_type"])})
write_csv(pd.DataFrame(cnt_rows), MOD, "pathway_robust_counts.csv",
          "Robust pathway rhythms per version × tissue × resource, with the share in the modal 3-h phase bin",
          required=["version", "tissue", "resource", "n_robust", "frac_robust_in_modal_bin"])
write_csv(pd.concat(map_rows, ignore_index=True), MOD, "pathway_phase_map.csv",
          "Peak ZT of robust pathway rhythms per cell type (all versions)",
          required=["version", "tissue", "cell_type", "resource", "pathway", "peak_zt"])
cons = pd.DataFrame(cons_rows)
if len(cons):
    cons["rayleigh_q"] = cons.groupby(["version", "tissue", "resource"])["rayleigh_p"].transform(
        lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
write_csv(cons, MOD, "pathway_phase_consistency.csv",
          "Phase agreement of robust pathways across cell types (circular mean, resultant, Rayleigh)",
          required=["version", "tissue", "resource", "pathway", "circ_mean_peak_zt", "resultant_length"])

# ---- (3) TF activity vs TF mRNA
act = versions["primary"]
act = act[act["resource"] == "collectri"][["tissue", "cell_type", "pathway", "peak_zt", "p", "robust"]]
lag_rows = []
for tissue in ["cochlea", "bm"]:
    mrna = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{tissue}.csv",
                       usecols=["group", "gene", "peak_zt", "p", "tier"])
    m = act[act["tissue"] == tissue].merge(mrna, left_on=["cell_type", "pathway"], right_on=["group", "gene"],
                                           suffixes=("_activity", "_mrna"))
    m = m[(m["p_activity"] < 0.05) & (m["p_mrna"] < 0.05)]
    for r in m.itertuples():
        lag_rows.append({"tissue": tissue, "cell_type": r.cell_type, "tf": r.pathway,
                         "clock_tf": r.pathway in CLOCK_TF, "activity_peak_zt": r.peak_zt_activity,
                         "activity_p": r.p_activity, "activity_robust": r.robust, "mrna_peak_zt": r.peak_zt_mrna,
                         "mrna_p": r.p_mrna, "mrna_tier": r.tier,
                         "lag_h": float(cosinor.circ_diff(r.peak_zt_activity, r.peak_zt_mrna))})
lag = pd.DataFrame(lag_rows)
write_csv(lag, MOD, "tf_activity_mrna_lag.csv",
          "TF activity (CollecTRI ULM) vs TF mRNA peak phases per cell type (both p < 0.05)",
          required=["tissue", "cell_type", "tf", "activity_peak_zt", "mrna_peak_zt", "lag_h"])
summ = []
if len(lag):
    for (tissue, clk), g in lag.groupby(["tissue", "clock_tf"]):
        m, R, p = rayleigh(g["lag_h"].to_numpy() % 24)
        summ.append({"tissue": tissue, "clock_tf": clk, "n_pairs": len(g), "circ_mean_lag_h": ((m + 12) % 24) - 12,
                     "resultant_length": R, "rayleigh_p": p, "frac_activity_after_mrna": float((g["lag_h"] > 0).mean())})
write_csv(pd.DataFrame(summ), MOD, "tf_lag_summary.csv", "Circular summary of TF activity–mRNA lags",
          required=["tissue", "clock_tf", "n_pairs", "circ_mean_lag_h"])
log("done")
