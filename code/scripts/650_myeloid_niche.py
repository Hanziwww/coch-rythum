"""Round 6, WP-M5 — retention / release axis (M6) and niche cues.

Primary M6 (per tissue; tissues combined by Fisher in 670): Fisher combination of two exact permutation p values
  (a) the CXCL12 → CXCR4 interaction to neutrophils — the source × neutrophil-target pair with the largest mean LIANA
      magnitude (64 ccc_rhythm, lr_means per sample);
  (b) the per-sample mean neutrophil Cxcr4 / Cxcr2 log-ratio (66: log2((Cxcr4 + 1) / (Cxcr2 + 1)) on CP10k) in pooled
      neutrophils (four labels). The plan named the median; in the cochlea 23–45 % of cells have neither transcript, so every
      per-sample median is exactly 0 (degenerate). The mean is used in both tissues (D47; BM mean ≈ median); medians
      are reported alongside.
  Direction: the Cxcr4 / Cxcr2 peak lies within ± 4 h of the GSE86619 blood-timer peak (M2 score, pooled neutrophils;
  m_axes_rhythm) in the same tissue.
Agreeing methods (per tissue):
  (1) Cxcl12 mRNA rhythm in the representative sender (pseudobulk log2 CPM; exact p <= 0.05);
  (2) NicheNet: CXCL12 activity percentile >= 0.9 for a day or night programme of a neutrophil receiver (651);
  (3) cue phase: glucocorticoid-response programme (82) rhythmic (exact p <= 0.05) in >= 1 neutrophil label with the
      circular-mean peak of the rhythmic labels within ± 4 h of the Cxcr4 / Cxcr2 peak (glucocorticoids induce Cxcr4);
      the adrenergic-cAMP programme of the sender vs its Cxcl12 peak is reported descriptively;
  (4) GSE198770 whole bone marrow (controls, 6 ZT × 3): Cxcl12 cosinor p < 0.05 (BM only).
Robustness: LOTO, Wasserstein (OT) test of the per-cell ratio, high-confidence cells (00_audit margin > 0), r with the
sample pattern (210 PC1); paired cochlea − BM contrast of the ratio. Exploratory: CXCR2 ligands (Cxcl1, Cxcl2, Cxcl5,
Ppbp) → Cxcr2 and CXCL12 → CXCR4 to monocytes (64); Sell (CD62L) per sample (66); GSE198770 Cxcr4 / Cxcr2 / Sell and the
neutrophil-depleted arm.
Outputs (results/r6_myeloid/): m_niche_interactions.csv, m_niche_ratio_rhythm.csv, m_niche_agreeing.csv,
m_niche_external.csv, m_primary_m6.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.abundance import loto_sd, ols, perm_p  # noqa: E402
from lib.disttest import ot_rhythm_test  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r6_myeloid"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
CXCR2_LIG = ["Cxcl1", "Cxcl2", "Cxcl5", "Ppbp"]
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
axr = pd.read_csv(RESULTS / f"{MOD}/m_axes_rhythm.csv")
qc = pd.read_csv(RESULTS / "00_audit/label_support_qc.csv", usecols=["cell_id", "margin"]).set_index("cell_id")["margin"]
cells = pd.read_csv(RESULTS / "66_state_continua/state_scores_by_cell.csv", usecols=["cell_id", "tissue_code", "sample_id", "zt", "cell_type", "cxcr4_cxcr2", "sell"])
cells = cells[cells["cell_type"].isin(NEU)]
cells["hc"] = cells["cell_id"].map(qc).fillna(-1).to_numpy() > 0
cue = pd.read_csv(RESULTS / "82_systemic/systemic_cue_rhythm.csv")
nn_path = RESULTS / f"{MOD}/m_nichenet_ligand_activity.csv"
nn = pd.read_csv(nn_path) if nn_path.exists() else pd.DataFrame(columns=["tissue", "receiver", "gene_set", "ligand", "percentile", "rank"])


def rrow(y, zt, pc1):
    F, p, pk, amp = ols(y, zt)
    return {"peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p, "p_perm": perm_p(y, zt),
            "loto_phase_sd_h": loto_sd(y, zt), "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1])}


inter, rr, ag, prim, ratio_med = [], [], [], [], {}
for t in ["bm", "cochlea"]:
    s0 = design[(design["tissue"] == t) & (design["level"] == "cell_type")].drop_duplicates("sample_id").sort_values("zt")
    samples, zt = s0["sample_id"].tolist(), s0["zt"].to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(samples).to_numpy(float)
    # ---------------- (a) CXCL12 → CXCR4 and related interactions (64)
    cr = pd.read_csv(RESULTS / f"64_ccc_rhythm/ccc_rhythm_{t}.csv")
    fam = cr[((cr["ligand_complex"] == "Cxcl12") & cr["receptor_complex"].isin(["Cxcr4"]) & cr["target"].isin(NEU + MONO))
             | (cr["ligand_complex"].isin(CXCR2_LIG) & (cr["receptor_complex"] == "Cxcr2") & cr["target"].isin(NEU))].copy()
    fam["axis"] = np.where(fam["ligand_complex"] == "Cxcl12", np.where(fam["target"].isin(NEU), "CXCL12→CXCR4 (neutrophils)", "CXCL12→CXCR4 (monocytes / macrophages)"),
                           "CXCR2 ligands→CXCR2 (neutrophils)")
    fam["magnitude_rank_in_axis"] = fam.groupby("axis")["mean_lr_means"].rank(ascending=False, method="first").astype(int)
    fam.insert(0, "tissue_", t)
    inter.append(fam.drop(columns=["tissue"]).rename(columns={"tissue_": "tissue"}))
    rep = fam[(fam["axis"] == "CXCL12→CXCR4 (neutrophils)")].sort_values("mean_lr_means", ascending=False).iloc[0]
    # ---------------- (b) neutrophil Cxcr4 / Cxcr2 log-ratio
    ct = cells[cells["tissue_code"] == t]
    out = {}
    for subset, c_ in [("all cells", ct), ("high-confidence cells", ct[ct["hc"]])]:
        for score in ["cxcr4_cxcr2", "sell"]:
            vals = [c_.loc[c_["sample_id"] == s, score].dropna().to_numpy() for s in samples]
            if min(len(v) for v in vals) < 10:
                continue
            y = np.array([np.mean(v) for v in vals])
            med = np.array([np.median(v) for v in vals])
            r = {"tissue": t, "score": score, "group": "pooled neutrophils", "subset": subset, "summary": "per-sample mean",
                 "n_cells": int(sum(len(v) for v in vals)), **rrow(y, zt, pc1), **ot_rhythm_test(vals),
                 "median_by_zt": "; ".join(f"{a:.3f}" for a in med), "p_perm_median": perm_p(med, zt)}
            rr.append(r)
            out[(subset, score)] = r
            if subset == "all cells" and score == "cxcr4_cxcr2":
                ratio_med[t] = pd.Series(y, index=zt)
    rb = out[("all cells", "cxcr4_cxcr2")]
    p_fisher = float(stats.chi2.sf(-2 * (np.log(rep["p_perm"]) + np.log(rb["p_perm"])), 4))
    m2 = axr[(axr["tissue"] == t) & (axr["axis"] == "blood timer ZT13−ZT5 (GSE86619)") & (axr["group"] == "pooled neutrophils")
             & (axr["subset"] == "all cells")].iloc[0]
    lag = float(phase.circ_diff(rb["peak_zt"], m2["peak_zt"]))
    # ---------------- agreeing methods
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{t}__cell_type.parquet")
    lib_ = pb.sum(0)
    snd = []
    for ty in sorted(set(c.split("|")[0] for c in pb.columns)):
        cols = [f"{ty}|{s}" for s in samples]
        if not all(c in pb.columns for c in cols) or "Cxcl12" not in pb.index:
            continue
        cpm = pb.loc["Cxcl12", cols].to_numpy(float) / lib_[cols].to_numpy(float) * 1e6
        if cpm.mean() < 10:
            continue
        snd.append({"tissue": t, "method": "Cxcl12 mRNA in senders (pseudobulk log2 CPM)", "unit": ty, "mean_cpm": float(cpm.mean()),
                    "representative_sender": ty == rep["source"], **rrow(np.log2(cpm + 1), zt, pc1)})
    snd = pd.DataFrame(snd)
    rs = snd[snd["representative_sender"]]
    a1 = bool(len(rs) and rs["p_perm"].iloc[0] <= 0.05)
    ag += snd.to_dict("records")
    nt = nn[(nn["tissue"] == t) & nn["receiver"].isin(NEU) & (nn["ligand"] == "Cxcl12")]
    a2 = bool(len(nt) and (nt["percentile"] >= 0.9).any())
    for r in nt.itertuples():
        ag.append({"tissue": t, "method": "NicheNet CXCL12 activity (neutrophil receivers)", "unit": f"{r.receiver} | {r.gene_set}",
                   "percentile": r.percentile, "rank": r.rank})
    gc = cue[(cue["tissue"] == t) & (cue["set"] == "glucocorticoid_response") & cue["cell_type"].isin(NEU)]
    gr = gc[gc["p_perm"] <= 0.05]
    gc_peak = phase.circ_mean(gr["peak_zt"].to_numpy())[0] if len(gr) else np.nan
    a3 = bool(len(gr) and abs(phase.circ_diff(gc_peak, rb["peak_zt"])) <= 4)
    for r in gc.itertuples():
        ag.append({"tissue": t, "method": "glucocorticoid-response programme (82) in neutrophils", "unit": r.cell_type, "peak_zt": r.peak_zt,
                   "p_perm": r.p_perm, "lag_vs_cxcr4_cxcr2_h": float(phase.circ_diff(r.peak_zt, rb["peak_zt"]))})
    adr = cue[(cue["tissue"] == t) & (cue["set"] == "adrenergic_camp") & (cue["cell_type"] == rep["source"])]
    for r in adr.itertuples():
        ag.append({"tissue": t, "method": "adrenergic-cAMP programme (82) in the representative sender (descriptive)", "unit": r.cell_type,
                   "peak_zt": r.peak_zt, "p_perm": r.p_perm,
                   "lag_vs_sender_cxcl12_h": float(phase.circ_diff(r.peak_zt, rs["peak_zt"].iloc[0])) if len(rs) else np.nan})
    prim.append({"hypothesis": "M6 retention / release axis", "tissue": t, "primary_statistic": "Fisher of exact permutation p: CXCL12→CXCR4 to "
                 "neutrophils (representative by magnitude) and the per-sample mean neutrophil Cxcr4/Cxcr2 log-ratio",
                 "primary_p": p_fisher, "p_interaction": rep["p_perm"], "p_ratio": rb["p_perm"],
                 "representative_interaction": f"{rep['source']} → {rep['target']}", "interaction_peak_zt": rep["peak_zt"],
                 "ratio_peak_zt": rb["peak_zt"], "m2_timer_peak_zt": float(m2["peak_zt"]), "lag_ratio_minus_timer_h": lag,
                 "direction_ok": abs(lag) <= 4, "agree_sender_cxcl12": a1, "agree_nichenet_cxcl12": a2, "agree_gc_cue": a3,
                 "gc_rhythmic_labels": len(gr), "gc_peak_zt": gc_peak, "in_window": True})
    log(f"{t}: interaction p {rep['p_perm']:.3f}, ratio p {rb['p_perm']:.3f} (peak ZT{rb['peak_zt']:.1f}), Fisher {p_fisher:.3f}; lag vs timer {lag:+.1f} h")

# ---------------- paired contrast of the ratio
d_ = (ratio_med["cochlea"] - ratio_med["bm"])
zts = d_.index.to_numpy(float)
F, p, pk, amp = ols(d_.to_numpy(float), zts)
rr.append({"tissue": "paired (cochlea − BM)", "score": "cxcr4_cxcr2", "group": "pooled neutrophils", "subset": "all cells", "peak_zt": pk,
           "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p": p, "p_perm": perm_p(d_.to_numpy(float), zts),
           "loto_phase_sd_h": loto_sd(d_.to_numpy(float), zts), "mean_difference": float(d_.mean()), "n_zt_cochlea_higher": int((d_ > 0).sum())})

# ---------------- GSE198770 whole bone marrow
meta = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__meta.csv")
ex = pd.read_csv(EXTERNAL / "prepared_r2/GSE198770_bone_marrow__expr.csv", index_col=0)
ext = []
for cond in ["control", "neutrophil_depleted"]:
    m = meta[meta["condition"] == cond]
    L = np.log2(ex[m["sample"]].astype(float) + 1)
    for g in ["Cxcl12", "Cxcr4", "Cxcr2", "Sell", "Kitl"]:
        if g not in L.index:
            continue
        f = cosinor.fit(L.loc[g].to_numpy(float), m["time_h"].to_numpy(float))
        ext.append({"dataset": "GSE198770 bone marrow", "condition": cond, "gene": g, "n_samples": len(m), "peak_zt": float(f["peak"][0]),
                    "peak_clock": float(phase.zt_to_clock(f["peak"][0])), "amplitude_log2": float(f["amp"][0]), "p": float(f["p"][0])})
    r4 = (L.loc["Cxcr4"] - L.loc["Cxcr2"]).to_numpy(float)
    f = cosinor.fit(r4, m["time_h"].to_numpy(float))
    ext.append({"dataset": "GSE198770 bone marrow", "condition": cond, "gene": "Cxcr4 − Cxcr2 (log2)", "n_samples": len(m), "peak_zt": float(f["peak"][0]),
                "peak_clock": float(phase.zt_to_clock(f["peak"][0])), "amplitude_log2": float(f["amp"][0]), "p": float(f["p"][0])})
ext = pd.DataFrame(ext)
c12 = ext[(ext["condition"] == "control") & (ext["gene"] == "Cxcl12")].iloc[0]
for r in prim:
    r["agree_gse198770_cxcl12"] = bool(c12["p"] < 0.05) if r["tissue"] == "bm" else np.nan
    r["n_agreeing_methods"] = int(sum(bool(r[k]) for k in ["agree_sender_cxcl12", "agree_nichenet_cxcl12", "agree_gc_cue"])
                                  + (bool(r["agree_gse198770_cxcl12"]) if r["tissue"] == "bm" else 0))

write_csv(pd.concat(inter, ignore_index=True), MOD, "m_niche_interactions.csv", "CXCL12→CXCR4 (neutrophils, monocytes / macrophages) and CXCR2-ligand → CXCR2 "
          "interaction rhythms (64) with magnitude ranks", required=["tissue", "axis", "source", "target", "peak_zt", "p_perm"])
write_csv(pd.DataFrame(rr), MOD, "m_niche_ratio_rhythm.csv", "Neutrophil Cxcr4/Cxcr2 log-ratio and Sell per-sample means (medians alongside): rhythm (exact permutation, LOTO, OT; "
          "high-confidence cells; pattern r) and the paired cochlea − BM contrast", required=["tissue", "score", "subset", "peak_zt", "p_perm"])
write_csv(pd.DataFrame(ag), MOD, "m_niche_agreeing.csv", "M6 agreeing-method components: sender Cxcl12 rhythms, NicheNet CXCL12 percentiles, glucocorticoid / "
          "adrenergic cue phases", required=["tissue", "method", "unit"])
write_csv(ext, MOD, "m_niche_external.csv", "GSE198770 whole bone marrow: Cxcl12 / Cxcr4 / Cxcr2 / Sell / Kitl cosinor (control and neutrophil-depleted)",
          required=["dataset", "condition", "gene", "peak_zt", "p"])
write_csv(pd.DataFrame(prim), MOD, "m_primary_m6.csv", "Primary hypothesis M6: per-tissue Fisher statistic, direction and agreeing methods",
          required=["hypothesis", "tissue", "primary_p", "direction_ok", "n_agreeing_methods"])
log("\n" + pd.DataFrame(rr)[["tissue", "score", "subset", "peak_zt", "p_perm", "ot_p_perm", "r_with_sample_pattern_pc1"]].round(3).to_string())
log("\n" + ext.round(3).to_string())
log("\n" + pd.DataFrame(prim).T.to_string())
