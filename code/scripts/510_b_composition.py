"""Round 5, WP-B1 — B-lineage composition and developmental timing (H1, H2), BM and cochlea with equal weight.

Stages (gold-standard labels, developmental order): Pro-B Cell (Cycling) → Pre-B Cell → B Cell (Transitional) →
B Cell (Naïve); B Cell (Mature) reported alongside (recirculating / mature pool, not part of the order test).
Frames per tissue:
  share_of_tissue  n_stage / all cells of the sample
  within_B         n_stage / all B-lineage cells of the sample — cancels the per-sample marrow-content axis, because
                   every cochlear B cell is marrow-derived (D16 / D17)
  ilr_balance      sequential binary partition along the developmental order (pro | rest, pre | later,
                   transitional | naïve + mature, naïve | mature): b = sqrt(r·s/(r+s)) · ln(gm(first) / gm(rest)),
                   counts + 0.5
Models per feature (24-h cosinor on the ZT axis):
  ols_logit     OLS on logit((n + 0.5)/(D + 1)) (balances: on b) — F test, exact permutation p over the 60 dihedral
                relabeling classes, LOTO phase SD
  quasibinomial GLM on proportions with var_weights = D, dispersion from Pearson X², F test of cos/sin (2 df)
  betabinomial  maximum-likelihood beta-binomial (logit mean, common precision), likelihood-ratio test (2 df)
  clr_within_B  ln(n + 0.5) − mean over the B stages (within_B only), OLS cosinor with exact permutation p
Verdict per feature (core = perm p ≤ 0.05 of ols_logit, quasibinomial p < 0.05, betabinomial p < 0.05, clr p ≤ 0.05):
  robust ≥ 3 core with LOTO phase SD < 3 h and core peaks within SD < 3 h; supported 2; fragile 1 or ols p < 0.05;
  else not rhythmic.
Sensitivity columns: high-confidence cells only (00_audit/label_support_qc margin > 0) — ols_logit p and peak shift;
r with the tissue-wide sample pattern (210 PC1); cochlea: marrow-adjusted residual of 52 (share_of_tissue).
Bootstrap (2,000 multinomial resamplings of cells within each sample): peak CIs and consecutive-stage phase lags.
Primary tests (pre-specified, D33):
  H1  naïve-B share within the B lineage: exact permutation p of ols_logit; peak inside ZT16–24
  H2  forward-lag statistic T = Σ_k ((φ_{k+1} − φ_k) mod 24) over pro → pre → transitional → naïve within-B peaks
      (T < 24 = one monotone sweep within a day); exact p over the 24 stage-label permutations (P(T_perm ≤ T_obs));
      bootstrap share of replicates with T < 24
Outputs (results/r5_bcell/): b_composition_long.csv, b_composition_summary.csv, b_phase_lags.csv, b_primary_composition.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.abundance import betabin, logit, loto_sd, ols, perm_p, quasibin  # noqa: E402  (shared models, round 6)
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r5_bcell"
ORDER = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)"]
BSTAGES = ORDER + ["B Cell (Mature)"]
rng = np.random.default_rng(5)
NBOOT = 2000

cnt = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
qc = pd.read_csv(RESULTS / "00_audit/label_support_qc.csv", usecols=["sample_id", "cell_type", "margin"])
hc = qc[qc["margin"] > 0].groupby(["sample_id", "cell_type"]).size().rename("n_hc").reset_index()
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
md = pd.read_csv(RESULTS / "52_marrow_decomposition/tissue_residual_rhythm.csv")
md = md[md["model"] == "marrow_adjusted"] if "model" in md else md


def ilr_balances(m):
    """m: samples × stages (BSTAGES order) counts → dict of balances (sequential binary partition)."""
    L = np.log(m + 0.5)
    out = {}
    for k, name in enumerate(["pro | rest", "pre | later", "transitional | naïve+mature", "naïve | mature"]):
        first, rest = L[:, k:k + 1], L[:, k + 1:]
        r, s = 1, rest.shape[1]
        out[f"balance: {name}"] = np.sqrt(r * s / (r + s)) * (first.mean(1) - rest.mean(1))
    return out


def order_T(peaks):
    return float(sum((peaks[k + 1] - peaks[k]) % 24 for k in range(len(peaks) - 1)))


long, summ, lags, prim = [], [], [], []
for t in ["bm", "cochlea"]:
    c = cnt[cnt["tissue"] == t]
    smp = c.drop_duplicates("sample_id").sort_values("zt")[["sample_id", "zt"]]
    zt = smp["zt"].to_numpy(float)
    N = c.groupby("sample_id")["n_cells"].sum().reindex(smp["sample_id"]).to_numpy(float)
    M = c.pivot_table(index="sample_id", columns="cell_type", values="n_cells", fill_value=0).reindex(smp["sample_id"])
    mB = M[BSTAGES].to_numpy(float)
    NB = mB.sum(1)
    h = hc[hc["sample_id"].isin(smp["sample_id"])].pivot_table(index="sample_id", columns="cell_type", values="n_hc", fill_value=0).reindex(smp["sample_id"]).fillna(0)
    hN, hB = h.sum(1).to_numpy(float), h.reindex(columns=BSTAGES).fillna(0).to_numpy(float)
    pc1 = pat[(pat["tissue"] == t) & (pat["pc"] == 1)].set_index("sample_id")["score"].reindex(smp["sample_id"]).to_numpy(float)
    clr = np.log(mB + 0.5) - np.log(mB + 0.5).mean(1, keepdims=True)
    feats = []
    for j, st in enumerate(BSTAGES):
        feats.append(("share_of_tissue", st, mB[:, j], N, hB[:, j], hN))
        feats.append(("within_B", st, mB[:, j], NB, hB[:, j], hB.sum(1)))
    for name, b in ilr_balances(mB).items():
        feats.append(("ilr_balance", name, b, None, ilr_balances(hB)[name], None))
    boot_peaks = {}
    for frame, feat, n, d, nh, dh in feats:
        if d is not None:
            y, yh = logit(n, d), logit(nh, dh)
        else:
            y, yh = n, nh
        F, p, pk, amp = ols(y, zt)
        pp = perm_p(y, zt)
        lsd = loto_sd(y, zt)
        rec = {"tissue": t, "frame": frame, "feature": feat}
        long.append({**rec, "model": "ols_logit" if d is not None else "ols_balance", "p": p, "p_perm": pp, "peak_zt": pk, "stat": F,
                     "note": f"LOTO SD {lsd:.2f} h"})
        core = {"ols_perm": pp <= 0.05}
        peaks = [pk]
        if d is not None:
            Fq, pq, pkq = quasibin(n, d, zt)
            long.append({**rec, "model": "quasibinomial", "p": pq, "p_perm": np.nan, "peak_zt": pkq, "stat": Fq, "note": ""})
            lr, pb, pkb = betabin(n, d, zt)
            long.append({**rec, "model": "betabinomial", "p": pb, "p_perm": np.nan, "peak_zt": pkb, "stat": lr, "note": ""})
            core.update({"quasibinomial": pq < 0.05, "betabinomial": pb < 0.05})
            peaks += [pkq, pkb]
            if frame == "within_B":
                j = BSTAGES.index(feat)
                Fc, pc_, pkc, _ = ols(clr[:, j], zt)
                ppc = perm_p(clr[:, j], zt)
                long.append({**rec, "model": "clr_within_B", "p": pc_, "p_perm": ppc, "peak_zt": pkc, "stat": Fc, "note": ""})
                core["clr_within_B"] = ppc <= 0.05
                peaks.append(pkc)
        # bootstrap: multinomial resampling of cells within each sample
        bp = []
        for _ in range(NBOOT):
            if frame == "ilr_balance":
                mb_ = np.vstack([rng.multinomial(int(NB[i]), mB[i] / NB[i]) for i in range(6)])
                yb = ilr_balances(mb_)[feat]
            else:
                j = BSTAGES.index(feat)
                if frame == "within_B":
                    nb_ = np.array([rng.binomial(int(NB[i]), mB[i, j] / NB[i]) for i in range(6)])
                    yb = logit(nb_, NB)
                else:
                    nb_ = np.array([rng.binomial(int(N[i]), mB[i, j] / N[i]) for i in range(6)])
                    yb = logit(nb_, N)
            bp.append(ols(yb, zt)[2])
        bp = np.array(bp)
        boot_peaks[(frame, feat)] = bp
        mu_b, _ = phase.circ_mean(bp)
        dev = np.abs(phase.circ_diff(bp, pk))
        Fh, ph, pkh, _ = ols(yh, zt)
        nc = int(sum(core.values()))
        csd = phase.circ_sd(np.array(peaks)) if len(peaks) >= 2 else np.nan
        verdict = ("robust" if nc >= 3 and lsd < 3 and (np.isnan(csd) or csd < 3) else "supported" if nc >= 2 else
                   "fragile" if nc >= 1 or p < 0.05 else "not rhythmic")
        r_pat = float(np.corrcoef(y, pc1)[0, 1])
        madj = np.nan
        if t == "cochlea" and frame == "share_of_tissue" and len(md):
            mm = md[md["cell_type"] == feat]
            madj = float(mm["resid_peak_zt"].iloc[0]) if len(mm) else np.nan
        summ.append({**rec, "peak_zt": pk, "peak_clock": float(phase.zt_to_clock(pk)), "amplitude": amp, "p_ols": p, "p_perm": pp,
                     "loto_phase_sd_h": lsd, "n_core_supporting": nc, "core_models": ";".join(k for k, v in core.items() if v),
                     "core_peak_circ_sd_h": csd, "verdict": verdict,
                     "boot_peak_ci_halfwidth_h": float(np.quantile(dev, 0.95)), "boot_frac_within_3h": float((dev <= 3).mean()),
                     "hc_only_p_ols": ph, "hc_only_peak_shift_h": float(phase.circ_diff(pkh, pk)),
                     "r_with_sample_pattern_pc1": r_pat, "marrow_adjusted_resid_peak_zt": madj,
                     "mean_value": float(np.mean(n / d)) if d is not None else float(np.mean(y))})
    # ---- consecutive-stage lags (within_B) and H2 order test
    pks = {st: [s for s in summ if s["tissue"] == t and s["frame"] == "within_B" and s["feature"] == st][0]["peak_zt"] for st in BSTAGES}
    for a, b in list(zip(ORDER[:-1], ORDER[1:])) + [("B Cell (Naïve)", "B Cell (Mature)")]:
        lb = (boot_peaks[("within_B", b)] - boot_peaks[("within_B", a)]) % 24
        lo = (pks[b] - pks[a]) % 24
        dev = np.abs(phase.circ_diff(lb, lo))
        lags.append({"tissue": t, "from_stage": a, "to_stage": b, "peak_from_zt": pks[a], "peak_to_zt": pks[b], "forward_lag_h": lo,
                     "boot_ci_halfwidth_h": float(np.quantile(dev, 0.95)), "boot_median_lag_h": float(np.median(lb))})
    obs = [pks[s] for s in ORDER]
    T = order_T(obs)
    perms = [order_T([obs[i] for i in pm]) for pm in itertools.permutations(range(4))]
    p_T = sum(x <= T + 1e-9 for x in perms) / len(perms)
    bootT = np.array([order_T([boot_peaks[("within_B", s)][k] for s in ORDER]) for k in range(NBOOT)])
    nv = [s for s in summ if s["tissue"] == t and s["frame"] == "within_B" and s["feature"] == "B Cell (Naïve)"][0]
    prim.append({"hypothesis": "H1 naïve-B timing", "tissue": t, "primary_statistic": "exact permutation p of the within-B naïve share cosinor (ols_logit)",
                 "primary_p": nv["p_perm"], "peak_zt": nv["peak_zt"], "peak_clock": nv["peak_clock"], "pre_specified_window": "ZT16–24",
                 "in_window": bool(16 <= nv["peak_zt"] < 24), "composition_verdict": nv["verdict"],
                 "detail": f"core models {nv['core_models']}; LOTO SD {nv['loto_phase_sd_h']:.2f} h; boot CI ±{nv['boot_peak_ci_halfwidth_h']:.1f} h"})
    prim.append({"hypothesis": "H2 developmental order", "tissue": t, "primary_statistic": "forward-lag statistic T over pro→pre→transitional→naïve within-B peaks; exact p over 24 label permutations",
                 "primary_p": p_T, "peak_zt": np.nan, "peak_clock": np.nan, "pre_specified_window": "T < 24 h (one monotone sweep)",
                 "in_window": bool(T < 24), "composition_verdict": "",
                 "detail": f"T {T:.1f} h; peaks " + ", ".join(f"{s.split(' (')[0]} ZT{pks[s]:.1f}" for s in ORDER) +
                           f"; bootstrap share T < 24: {(bootT < 24).mean():.2f}"})
    log(f"{t}: H1 p {nv['p_perm']:.3f} peak ZT{nv['peak_zt']:.1f}; H2 T {T:.1f} h p {p_T:.3f}")

lg = pd.DataFrame(long)
lg["peak_clock"] = phase.zt_to_clock(lg["peak_zt"])
write_csv(lg, MOD, "b_composition_long.csv", "B-lineage composition rhythms: every frame × feature × model (p, exact permutation p, peak ZT / clock)",
          required=["tissue", "frame", "feature", "model", "p", "peak_zt", "peak_clock"])
sm_ = pd.DataFrame(summ)
write_csv(sm_, MOD, "b_composition_summary.csv",
          "B-lineage composition: verdict per tissue × frame × stage / balance with LOTO, bootstrap CI, high-confidence and sample-pattern sensitivity",
          required=["tissue", "frame", "feature", "peak_zt", "peak_clock", "p_perm", "verdict"])
write_csv(pd.DataFrame(lags), MOD, "b_phase_lags.csv", "Consecutive B-stage phase lags (within-B frame) with multinomial-bootstrap CIs",
          required=["tissue", "from_stage", "to_stage", "forward_lag_h", "boot_ci_halfwidth_h"])
write_csv(pd.DataFrame(prim), MOD, "b_primary_composition.csv", "Primary hypotheses H1 (naïve-B timing) and H2 (developmental order): primary statistic per tissue",
          required=["hypothesis", "tissue", "primary_p", "in_window"])
log("\n" + sm_[sm_["frame"] != "share_of_tissue"][["tissue", "frame", "feature", "peak_zt", "p_ols", "p_perm", "n_core_supporting", "verdict",
                                                    "r_with_sample_pattern_pc1"]].round(3).to_string())
