"""Round 6, WP-M2 — myeloid composition rhythms, BM and cochlea (paired; D39). Agreeing evidence for M3 and M5.

Frames per tissue (gold-standard labels):
  share_of_tissue      each myeloid type / all cells of the sample
  within_myeloid       each myeloid type / all myeloid cells (GMP, four neutrophil states, two monocyte types,
                       macrophages, mast cells)
  within_neutrophil    band, mature, aged, activated / all neutrophils
  within_mono_mac      classical, non-classical monocytes, macrophages / their sum
  ilr_balance          sequential binary partitions along the lineages:
                       neutrophil  GMP | neutrophils; band | mature + aged + activated; mature | aged + activated; aged | activated
                       mono / mac  classical | non-classical + macrophage; non-classical | macrophage
Models (scripts/lib/abundance.py, shared with 510): OLS on logit share (or balance) with exact 60-class permutation p
and LOTO; quasi-binomial; beta-binomial; CLR within the frame (within_* frames) with permutation p. Verdict: robust /
supported / fragile / not rhythmic (core = permutation, quasi-binomial, beta-binomial, CLR).
Columns: high-confidence-only rerun (00_audit margin > 0), r with the tissue-wide sample pattern (210 PC1), cochlear
marrow-adjusted residual peak (52; share_of_tissue), multinomial-bootstrap peak CI (1,000).
Phase lags (within frames, bootstrap CI): band → mature → aged → activated; classical → non-classical → macrophage.
Paired contrast: per feature, cochlea − BM peak difference.
Outputs (results/r6_myeloid/): m_composition_long.csv, m_composition_summary.csv, m_phase_lags.csv, m_composition_paired.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.abundance import balance, betabin, logit, loto_sd, ols, perm_p, quasibin, verdict  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r6_myeloid"
rng = np.random.default_rng(62)
NBOOT = 1000
GMP = "Granulocyte-Monocyte Progenitor"
NEU = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"]
MONO = ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"]
MYEL = [GMP] + NEU + MONO + ["Mast Cell"]
BAL = {"GMP | neutrophils": ([GMP], NEU), "band | mature + aged + activated": (NEU[:1], NEU[1:]),
       "mature | aged + activated": (NEU[1:2], NEU[2:]), "aged | activated": (NEU[2:3], NEU[3:]),
       "classical | non-classical + macrophage": (MONO[:1], MONO[1:]), "non-classical | macrophage": (MONO[1:2], MONO[2:])}
FRAMES = {"within_myeloid": MYEL, "within_neutrophil": NEU, "within_mono_mac": MONO}

cnt = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
qc = pd.read_csv(RESULTS / "00_audit/label_support_qc.csv", usecols=["sample_id", "cell_type", "margin"])
hc = qc[qc["margin"] > 0].groupby(["sample_id", "cell_type"]).size().rename("n_hc").reset_index()
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
md = pd.read_csv(RESULTS / "52_marrow_decomposition/tissue_residual_rhythm.csv")
md = md[md["model"] == "marrow_adjusted"].set_index("cell_type")["resid_peak_zt"]

long, summ, lags = [], [], []
for t in ["bm", "cochlea"]:
    c = cnt[cnt["tissue"] == t]
    smp = c.drop_duplicates("sample_id").sort_values("zt")[["sample_id", "zt"]]
    zt = smp["zt"].to_numpy(float)
    N = c.groupby("sample_id")["n_cells"].sum().reindex(smp["sample_id"]).to_numpy(float)
    M = c.pivot_table(index="sample_id", columns="cell_type", values="n_cells", fill_value=0).reindex(smp["sample_id"]).reindex(columns=MYEL).fillna(0)
    H = hc[hc["sample_id"].isin(smp["sample_id"])].pivot_table(index="sample_id", columns="cell_type", values="n_hc", fill_value=0) \
        .reindex(smp["sample_id"]).reindex(columns=MYEL).fillna(0)
    Hn = qc[qc["margin"] > 0].groupby("sample_id").size().reindex(smp["sample_id"]).to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(smp["sample_id"]).to_numpy(float)
    feats = [("share_of_tissue", ty, M[ty].to_numpy(float), N, H[ty].to_numpy(float), Hn, None) for ty in MYEL]
    for fr, members in FRAMES.items():
        D = M[members].sum(1).to_numpy(float)
        Dh = H[members].sum(1).to_numpy(float)
        Lc = np.log(M[members].to_numpy(float) + 0.5)
        clr = Lc - Lc.mean(1, keepdims=True)
        for j, ty in enumerate(members):
            feats.append((fr, ty, M[ty].to_numpy(float), D, H[ty].to_numpy(float), Dh, clr[:, j]))
    allparts = M.to_numpy(float)
    allh = H.to_numpy(float)
    for name, (a, b) in BAL.items():
        ia, ib = [MYEL.index(x) for x in a], [MYEL.index(x) for x in b]
        feats.append(("ilr_balance", name, balance(allparts, ia, ib), None, balance(allh, ia, ib), None, (ia, ib)))
    boots = {}
    for frame, feat, n, d, nh, dh, extra in feats:
        y, yh = (logit(n, d), logit(nh, dh)) if d is not None else (n, nh)
        F, p, pk, amp = ols(y, zt)
        pp, lsd = perm_p(y, zt), loto_sd(y, zt)
        rec = {"tissue": t, "frame": frame, "feature": feat}
        long.append({**rec, "model": "ols_logit" if d is not None else "ols_balance", "p": p, "p_perm": pp, "peak_zt": pk, "stat": F,
                     "note": f"LOTO SD {lsd:.2f} h"})
        core, peaks = {"ols_perm": pp <= 0.05}, [pk]
        if d is not None:
            Fq, pq, pkq = quasibin(n, d, zt)
            lr, pb, pkb = betabin(n, d, zt)
            long += [{**rec, "model": "quasibinomial", "p": pq, "p_perm": np.nan, "peak_zt": pkq, "stat": Fq, "note": ""},
                     {**rec, "model": "betabinomial", "p": pb, "p_perm": np.nan, "peak_zt": pkb, "stat": lr, "note": ""}]
            core.update({"quasibinomial": pq < 0.05, "betabinomial": pb < 0.05})
            peaks += [pkq, pkb]
            if extra is not None and frame != "share_of_tissue":
                Fc, pc_, pkc, _ = ols(extra, zt)
                ppc = perm_p(extra, zt)
                long.append({**rec, "model": "clr_within_frame", "p": pc_, "p_perm": ppc, "peak_zt": pkc, "stat": Fc, "note": ""})
                core["clr_within_frame"] = ppc <= 0.05
                peaks.append(pkc)
        # bootstrap (binomial resampling within each sample; balances: multinomial over the myeloid parts)
        bp = []
        for _ in range(NBOOT):
            if frame == "ilr_balance":
                mb = np.vstack([rng.multinomial(int(allparts[i].sum()), allparts[i] / allparts[i].sum()) for i in range(6)])
                bp.append(ols(balance(mb, *extra), zt)[2])
            else:
                nb = np.array([rng.binomial(int(d[i]), n[i] / d[i]) if d[i] > 0 else 0 for i in range(6)])
                bp.append(ols(logit(nb, d), zt)[2])
        bp = np.array(bp)
        boots[(frame, feat)] = bp
        dev = np.abs(phase.circ_diff(bp, pk))
        nc = int(sum(core.values()))
        v, csd = verdict(nc, lsd, peaks, p)
        Fh, ph, pkh, _ = ols(yh, zt)
        summ.append({**rec, "peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p_ols": p, "p_perm": pp,
                     "loto_phase_sd_h": lsd, "n_core_supporting": nc, "core_models": ";".join(k for k, x in core.items() if x),
                     "core_peak_circ_sd_h": csd, "verdict": v, "boot_peak_ci_halfwidth_h": float(np.quantile(dev, 0.95)),
                     "hc_only_p_ols": ph, "hc_only_peak_shift_h": float(phase.circ_diff(pkh, pk)),
                     "r_with_sample_pattern_pc1": float(np.corrcoef(y, pc1)[0, 1]),
                     "marrow_adjusted_resid_peak_zt": float(md.get(feat, np.nan)) if (t == "cochlea" and frame == "share_of_tissue") else np.nan,
                     "mean_value": float(np.mean(n / d)) if d is not None else float(np.mean(y))})
    for fr, chain in [("within_neutrophil", NEU), ("within_mono_mac", MONO)]:
        pk = {s_: [r for r in summ if r["tissue"] == t and r["frame"] == fr and r["feature"] == s_][0]["peak_zt"] for s_ in chain}
        for a, b in zip(chain[:-1], chain[1:]):
            lb = (boots[(fr, b)] - boots[(fr, a)]) % 24
            lo = (pk[b] - pk[a]) % 24
            lags.append({"tissue": t, "frame": fr, "from_type": a, "to_type": b, "peak_from_zt": pk[a], "peak_to_zt": pk[b], "forward_lag_h": lo,
                         "boot_ci_halfwidth_h": float(np.quantile(np.abs(phase.circ_diff(lb, lo)), 0.95))})
    log(f"{t}: composition done")

lg = pd.DataFrame(long)
lg["peak_clock"] = phase.zt_to_clock(lg["peak_zt"])
sm_ = pd.DataFrame(summ)
pr = []
for (fr, ft), g in sm_.groupby(["frame", "feature"]):
    g = g.set_index("tissue")
    if {"bm", "cochlea"} <= set(g.index):
        pr.append({"frame": fr, "feature": ft, "peak_bm_zt": g.loc["bm", "peak_zt"], "peak_cochlea_zt": g.loc["cochlea", "peak_zt"],
                   "diff_cochlea_minus_bm_h": float(phase.circ_diff(g.loc["cochlea", "peak_zt"], g.loc["bm", "peak_zt"])),
                   "verdict_bm": g.loc["bm", "verdict"], "verdict_cochlea": g.loc["cochlea", "verdict"],
                   "p_perm_bm": g.loc["bm", "p_perm"], "p_perm_cochlea": g.loc["cochlea", "p_perm"]})
write_csv(lg, MOD, "m_composition_long.csv", "Myeloid composition rhythms: every frame × feature × model (p, exact permutation p, peak ZT / clock)",
          required=["tissue", "frame", "feature", "model", "p", "peak_zt", "peak_clock"])
write_csv(sm_, MOD, "m_composition_summary.csv", "Myeloid composition verdicts per tissue × frame × type / balance with LOTO, bootstrap CI, high-confidence and sample-pattern columns",
          required=["tissue", "frame", "feature", "peak_zt", "peak_clock", "p_perm", "verdict"])
write_csv(pd.DataFrame(lags), MOD, "m_phase_lags.csv", "Consecutive myeloid-state phase lags (within frames) with bootstrap CIs",
          required=["tissue", "frame", "from_type", "to_type", "forward_lag_h"])
write_csv(pd.DataFrame(pr), MOD, "m_composition_paired.csv", "Paired BM–cochlea contrast of myeloid composition rhythms (peak difference, verdicts)",
          required=["frame", "feature", "peak_bm_zt", "peak_cochlea_zt", "diff_cochlea_minus_bm_h"])
log("\n" + sm_[sm_["frame"] != "share_of_tissue"][["tissue", "frame", "feature", "peak_zt", "p_perm", "n_core_supporting", "verdict", "r_with_sample_pattern_pc1"]].round(3).to_string())
