"""Round 4, WP-A — how reliable is a one-sample-per-time-point, 6-time-point design? Benchmark of our rhythm pipeline on
dense public time series with an independent (held-out) truth, with and without a sample-level nuisance shaped like our
cochlear tissue-wide sample pattern.

References
  GSE54650 (Zhang 2014; 12 mouse organs, constant darkness, CT18–64 every 2 h, microarray log2). A draw takes one 4-h
    grid (A: phases 2, 6, …, 22 or B: 0, 4, …, 20) and, for each of its 6 phases, one of the two cycles → 6 samples;
    truth = OLS cosinor on the 18 held-out samples (other cycle of the same grid + the whole other grid).
  GSE88954 (cochlea sham, every 4 h, n = 3; RNA-seq counts → log2 CPM). A draw takes one replicate per time point;
    truth = the other 12 samples.
Pipeline per draw (vectorised re-implementation of 31/32/33 for log2 data): limma-style moderated cosinor F (empirical-
Bayes variance squeezing, fitFDist), BH q; the 60 dihedral relabeling classes → pooled-null empirical FDR (q_perm, as in
31); LOTO (phase SD, max p, min amplitude ratio); independent method = a JTK-like rank test (Kendall tau vs cosine
templates at the 6 lags, exact null by enumeration, Bonferroni over lags — cannot reach p < 0.05 with 6 points) or the
unmoderated OLS F test (standing in for edgeR-QL / LR, the parametric alternatives that carried the real Tier 1).
  Tier 1: q_BH < 0.05 & q_perm < 0.10 & ptt FC ≥ 1.5 & JTK p < 0.05 & LOTO phase SD < 3 h & not single-sample-driven
  Tier 2: q_BH < 0.10 & ptt FC ≥ 1.5 (not Tier 1); 'Tier 1 core' = Tier 1 without q_perm; nominal: p < 0.05.
Estimated FDR of a rule = mean number of calls under the 59 non-identity relabelings / observed calls (as in 211);
realised FDP = share of calls not rhythmic in the truth (lenient: truth q ≥ 0.05; strict: truth p > 0.2).
Sensitivity = share of truth-rhythmic genes (truth q < 0.05, truth ptt FC ≥ 1.5) called. Phase error = |peak − truth|.
Nuisance ('sample pattern' conditions): for 40 % of genes, c_g · s(t) is added, s = our cochlear PC1 sample scores
(210; ordered by ZT and mapped to the draw's time order), |c_g| scaled so the induced cosinor amplitude has median 0.38
log2 (pattern-like genes of 210/211; 'half' = 0.19), random sign, log-normal spread. Pattern flag (as in 211): r > 0.8
between the gene's draw profile and s.
Outputs (results/r4_benchmark/): benchmark_draws.csv, benchmark_summary.csv, benchmark_sensitivity.csv,
benchmark_clock_recovery.csv, benchmark_pattern_flags.csv
"""
import itertools
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_benchmark"
W = 2 * np.pi / 24
CLASSES = cosinor.dihedral_classes(6)
N_DRAWS = int(os.environ.get("N_DRAWS", 40))
CLOCK = ["Arntl", "Npas2", "Clock", "Nr1d1", "Nr1d2", "Dbp", "Tef", "Hlf", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Nfil3", "Ciart", "Bhlhe41"]
pat = pd.read_csv(RESULTS / "r2_cochlea/sample_pattern_scores.csv")
S_PAT = pat[(pat["tissue"] == "cochlea") & (pat["pc"] == 1)].sort_values("zt")["score"].to_numpy()
S_PAT = (S_PAT - S_PAT.mean()) / S_PAT.std()
_A_PAT = float(cosinor.fit(S_PAT[:, None], np.arange(6) * 4.0)["amp"][0])


# ---------------- statistics
def trigamma_inv(x):
    y = 0.5 + 1 / x
    for _ in range(50):
        d = special.polygamma(1, y) * (1 - special.polygamma(1, y) / x) / special.polygamma(2, y)
        y = y + d
        if np.all(np.abs(d / y) < 1e-8):
            break
    return y


def fit_fdist(s2, df):
    s2 = np.maximum(s2, 1e-12)
    z = np.log(s2)
    e = z - special.digamma(df / 2) + np.log(df / 2)
    emean, evar = e.mean(), e.var(ddof=1) - special.polygamma(1, df / 2)
    if evar > 0:
        d0 = 2 * trigamma_inv(evar)
        s02 = np.exp(emean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s02 = np.inf, np.exp(emean)
    return d0, s02


def mod_cosinor(Y, t):
    """Y: samples × genes (log2). Moderated F (2, df + d0) cosinor; returns a, b, amp, peak, F, p."""
    X = cosinor.design(t)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    df = Y.shape[0] - 3
    rss = (res ** 2).sum(0)
    s2 = rss / df
    d0, s02 = fit_fdist(s2, df)
    s2p = (d0 * s02 + df * s2) / (d0 + df) if np.isfinite(d0) else np.full_like(s2, s02)
    ssr = ((Y - Y.mean(0)) ** 2).sum(0) - rss
    F = (ssr / 2) / s2p
    dft = df + d0 if np.isfinite(d0) else 1e6
    p = stats.f.sf(F, 2, dft)
    amp, peak = cosinor.amp_phase(beta[1], beta[2])
    return {"a": beta[1], "b": beta[2], "amp": amp, "peak": peak, "F": F, "p": p}


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


PAIRS = list(itertools.combinations(range(6), 2))


def _jtk_tables():
    tmpl, dists = [], []
    for lag in range(6):
        T = np.cos(W * (np.arange(6) * 4.0 - lag * 4.0))
        T = np.round(T, 9)
        sT = np.array([np.sign(T[i] - T[j]) for i, j in PAIRS])
        tmpl.append(sT)
        Ss = []
        for perm in itertools.permutations(range(6)):
            r = np.array(perm, float)
            Ss.append(float(np.sum(np.sign([r[i] - r[j] for i, j in PAIRS]) * sT)))
        dists.append(np.array(Ss))
    return np.vstack(tmpl), dists


JTK_T, JTK_D = _jtk_tables()


def jtk_p(Y):
    """Y: 6 × genes in time order. Min over lags of the exact two-sided p of Kendall S, Bonferroni × 6."""
    sY = np.stack([np.sign(Y[i] - Y[j]) for i, j in PAIRS], 1)  # genes × 15
    S = sY @ JTK_T.T                                            # genes × 6
    ps = np.empty_like(S)
    for k in range(6):
        d = np.sort(np.abs(JTK_D[k]))
        ps[:, k] = 1 - np.searchsorted(d, np.abs(S[:, k]) - 1e-9, side="left") / len(d)
    return np.minimum(ps.min(1) * 6, 1)


def loto(Y, t):
    full = mod_cosinor(Y, t)
    pk, am, pv = [], [], []
    for i in range(6):
        k = np.arange(6) != i
        r = mod_cosinor(Y[k], t[k])
        pk.append(r["peak"]); am.append(r["amp"]); pv.append(r["p"])
    pk, am, pv = np.vstack(pk), np.vstack(am), np.vstack(pv)
    R = np.hypot(np.cos(W * pk).mean(0), np.sin(W * pk).mean(0))
    sd = np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1))) / W
    with np.errstate(divide="ignore", invalid="ignore"):
        return sd, am.min(0) / full["amp"], pv.max(0)


def rules(Y, t, q_perm=None):
    f = mod_cosinor(Y, t)
    q = bh(f["p"])
    fc = 2 ** (2 * f["amp"])
    jp = jtk_p(Y)
    ols_p = cosinor.fit(Y, t)["p"]                   # unmoderated OLS F: stands in for edgeR-QL / LR (parametric alternatives)
    indep = (jp < 0.05) | (ols_p < 0.05)
    sd, mar, mxp = loto(Y, t)
    single = (f["p"] < 0.05) & ((mxp > 0.25) | (mar < 0.5))
    core = (q < 0.05) & (fc >= 1.5) & indep & (sd < 3) & ~single
    out = {"fit": f, "q": q, "fc": fc, "nominal": f["p"] < 0.05, "tier1_core": core, "tier2_rule": (q < 0.10) & (fc >= 1.5)}
    if q_perm is not None:
        out["tier1"] = core & (q_perm < 0.10)
        out["tier2"] = out["tier2_rule"] & ~out["tier1"]
    return out


def run_draw(Y, t, truth, genes, expr_dec, nuisance_genes=None):
    """Y: 6 × genes ordered by t ascending (equally spaced)."""
    obs = rules(Y, t)
    nullF, nulls = [], {k: [] for k in ["nominal", "tier1_core", "tier2_rule"]}
    for perm in CLASSES[1:]:
        r = rules(Y, t[list(perm)])
        nullF.append(r["fit"]["F"])
        for k in nulls:
            nulls[k].append(int(r[k].sum()))
    pooled = np.sort(np.concatenate(nullF))
    F = obs["fit"]["F"]
    n_ge = len(pooled) - np.searchsorted(pooled, F, side="left")
    o = np.argsort(-F)
    fdr = np.minimum(1, (n_ge[o] / (len(CLASSES) - 1)) / np.arange(1, len(o) + 1))
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    qp = np.empty_like(fdr)
    qp[o] = fdr
    full = rules(Y, t, q_perm=qp)
    true_rh = (truth["q"] < 0.05) & (truth["fc"] >= 1.5)
    rows = []
    for rule in ["tier1", "tier2", "tier1_core", "tier2_rule", "nominal"]:
        c = full[rule]
        n = int(c.sum())
        est = (min(1.0, np.mean(nulls[rule]) / n) if n else np.nan) if rule in nulls else np.nan
        pe = np.abs(cosinor.circ_diff(full["fit"]["peak"][c & (truth["q"] < 0.05)], truth["peak"][c & (truth["q"] < 0.05)]))
        rows.append({"rule": rule, "n_calls": n, "est_fdr_perm": est,
                     "fdp_lenient": float((truth["q"][c] >= 0.05).mean()) if n else np.nan,
                     "fdp_strict": float((truth["p"][c] > 0.2).mean()) if n else np.nan,
                     "sensitivity": float(c[true_rh].mean()) if true_rh.sum() else np.nan, "n_true_rhythmic": int(true_rh.sum()),
                     "median_phase_error_h": float(np.median(pe)) if len(pe) else np.nan,
                     "n_calls_nuisance_genes": int(c[nuisance_genes].sum()) if nuisance_genes is not None else 0,
                     "fdp_lenient_nuisance_genes": float((truth["q"][c & nuisance_genes] >= 0.05).mean())
                     if nuisance_genes is not None and (c & nuisance_genes).sum() else np.nan})
    sens = []
    amp_bin = pd.cut(truth["fc"], [1.5, 2, 3, np.inf], right=False, labels=["1.5-2", "2-3", ">=3"])
    for rule in ["tier1", "tier2"]:
        c = full[rule] | (full["tier1"] if rule == "tier2" else False)
        for (ab, ed), idx in pd.DataFrame({"ab": amp_bin, "ed": expr_dec}).assign(tr=truth["q"] < 0.05).query("tr").groupby(["ab", "ed"], observed=True).groups.items():
            idx = np.asarray(list(idx))
            sens.append({"rule": "tier1" if rule == "tier1" else "tier1_or_tier2", "truth_fc_bin": ab, "expr_decile": int(ed), "n": len(idx),
                         "called": int(c[idx].sum())})
    clock = {g: (bool(full["tier1"][i]), bool(full["tier1"][i] | full["tier2"][i]), bool(truth["q"][i] < 0.05)) for g, i in genes.items() if g in CLOCK}
    pf = None
    if nuisance_genes is not None:
        Yc = Y - Y.mean(0)
        s = S_PAT - S_PAT.mean()
        r = (Yc * s[:, None]).sum(0) / np.sqrt((Yc ** 2).sum(0) * (s ** 2).sum())
        flag = r > 0.8
        pf = []
        for rule in ["tier1", "tier2", "tier1_core"]:
            c = full[rule]
            pf.append({"rule": rule, "n_calls": int(c.sum()), "n_flagged": int((c & flag).sum()),
                       "fdp_lenient_all": float((truth["q"][c] >= 0.05).mean()) if c.sum() else np.nan,
                       "fdp_lenient_unflagged": float((truth["q"][c & ~flag] >= 0.05).mean()) if (c & ~flag).sum() else np.nan,
                       "frac_nuisance_calls_flagged": float(flag[c & nuisance_genes & (truth["q"] >= 0.05)].mean())
                       if (c & nuisance_genes & (truth["q"] >= 0.05)).sum() else np.nan,
                       "sensitivity_unflagged": float((c & ~flag)[true_rh].mean()) if true_rh.sum() else np.nan})
    return rows, sens, clock, pf


def truth_fit(Yt, tt):
    f = cosinor.fit(Yt, tt)
    return {"q": bh(f["p"]), "p": f["p"], "fc": 2 ** (2 * f["amp"]), "peak": f["peak"]}


def add_nuisance(Y, rng, scale):
    G = Y.shape[1]
    sel = rng.random(G) < 0.4
    c = (scale / _A_PAT) * np.exp(rng.normal(0, 0.5, G)) * rng.choice([-1, 1], G)
    Y2 = Y + np.outer(S_PAT, c * sel)
    return Y2, sel


def work(task):
    name, unit, Y_all, t_all, rep_all, genes, draws, seed = task
    rng = np.random.default_rng(seed)
    expr = Y_all.mean(1)
    expr_dec = pd.qcut(pd.Series(expr).rank(method="first"), 10, labels=False).to_numpy()
    gidx = {g: i for i, g in enumerate(genes)}
    out_rows, out_sens, out_clock, out_pf = [], [], [], []
    for d_i, (sel, held) in enumerate(draws):
        t = t_all[sel] % 24
        o = np.argsort(t)
        Y = Y_all[:, sel][:, o].T
        tt = t[o]
        truth = truth_fit(Y_all[:, held].T, t_all[held] % 24)
        for cond, scale in [("clean", 0.0), ("sample pattern (half)", 0.19), ("sample pattern (full)", 0.38)]:
            Yd, nsel = (Y, None) if scale == 0 else add_nuisance(Y, rng, scale)
            rows, sens, clock, pf = run_draw(Yd, tt, truth, gidx, expr_dec, nsel)
            base = {"dataset": name, "unit": unit, "draw": d_i, "condition": cond}
            out_rows += [{**base, **r} for r in rows]
            out_sens += [{**base, **r} for r in sens]
            out_clock += [{**base, "gene": g, "tier1": v[0], "tier1_or_2": v[1], "truth_rhythmic": v[2]} for g, v in clock.items()]
            if pf:
                out_pf += [{**base, **r} for r in pf]
    return out_rows, out_sens, out_clock, out_pf


def build_tasks():
    tasks = []
    rng = np.random.default_rng(1)
    # GSE54650
    ex = pd.read_csv(EXTERNAL / "prepared_r2/GSE54650__expr.csv", index_col=0)
    me = pd.read_csv(EXTERNAL / "prepared_r2/GSE54650__meta.csv")
    for org, m in me.groupby("tissue"):
        m = m.sort_values("time_h")
        Y = ex[m["sample"]].to_numpy(float)
        keep = Y.mean(1) >= np.quantile(Y.mean(1), 0.3)                  # drop the lowest-expressed 30 % (array background)
        Y, genes = Y[keep], ex.index[keep].to_numpy()
        th = m["time_h"].to_numpy(float)
        draws = []                         # prepared meta: time_h = CT mod 24, replicate = cycle (1: CT18–40, 2: CT42–64)
        for _ in range(N_DRAWS):
            grid = 2 * int(rng.integers(2))
            sel = [int(rng.choice(np.where(th == ph)[0])) for ph in np.unique(th) if int(ph) % 4 == grid]
            held = np.setdiff1d(np.arange(len(th)), sel)
            draws.append((np.array(sel), held))
        tasks.append(("GSE54650 (DD, microarray)", org, Y, th, None, genes, draws, int(rng.integers(1e9))))
    # GSE88954 cochlea sham
    ex = pd.read_csv(EXTERNAL / "prepared/GSE88954__expr.csv", index_col=0)
    me = pd.read_csv(EXTERNAL / "prepared/GSE88954__meta.csv")
    m = me[me["condition"] == "sham"].sort_values(["time_h", "replicate"])
    C = ex[m["sample"]].to_numpy(float)
    cpm = C / C.sum(0) * 1e6
    keep = cpm.mean(1) >= 2
    Y, genes = np.log2(cpm[keep] + 1), ex.index[keep].to_numpy()
    th = m["time_h"].to_numpy(float)
    draws = []
    for _ in range(N_DRAWS):
        sel = np.array([rng.choice(np.where(th == tv)[0]) for tv in np.unique(th)])
        draws.append((sel, np.setdiff1d(np.arange(len(th)), sel)))
    tasks.append(("GSE88954 (cochlea sham, RNA-seq)", "cochlea", Y, th, None, genes, draws, int(rng.integers(1e9))))
    return tasks


if __name__ == "__main__":
    tasks = build_tasks()
    log(f"{len(tasks)} units × {N_DRAWS} draws × 3 conditions")
    with Pool(min(len(tasks), int(os.environ.get("WORKERS", 13)))) as pool:
        res = pool.map(work, tasks)
    rows = pd.DataFrame([r for x in res for r in x[0]])
    sens = pd.DataFrame([r for x in res for r in x[1]])
    clk = pd.DataFrame([r for x in res for r in x[2]])
    pf = pd.DataFrame([r for x in res for r in x[3]])
    write_csv(rows, MOD, "benchmark_draws.csv",
              "Per draw × condition × rule: calls, permutation-estimated FDR, realised FDP vs held-out truth, sensitivity, phase error",
              required=["dataset", "unit", "draw", "condition", "rule", "n_calls", "est_fdr_perm", "fdp_lenient", "sensitivity"])
    summ = rows.groupby(["dataset", "unit", "condition", "rule"]).agg(
        n_draws=("draw", "nunique"), mean_calls=("n_calls", "mean"), mean_est_fdr_perm=("est_fdr_perm", "mean"),
        mean_fdp_lenient=("fdp_lenient", "mean"), mean_fdp_strict=("fdp_strict", "mean"), q90_fdp_lenient=("fdp_lenient", lambda x: x.quantile(0.9)),
        mean_sensitivity=("sensitivity", "mean"), median_phase_error_h=("median_phase_error_h", "median"),
        n_true_rhythmic=("n_true_rhythmic", "mean")).reset_index()
    allu = rows.groupby(["dataset", "condition", "rule"]).agg(
        n_draws=("draw", "size"), mean_calls=("n_calls", "mean"), mean_est_fdr_perm=("est_fdr_perm", "mean"),
        mean_fdp_lenient=("fdp_lenient", "mean"), mean_fdp_strict=("fdp_strict", "mean"), q90_fdp_lenient=("fdp_lenient", lambda x: x.quantile(0.9)),
        mean_sensitivity=("sensitivity", "mean"), median_phase_error_h=("median_phase_error_h", "median"),
        n_true_rhythmic=("n_true_rhythmic", "mean")).reset_index().assign(unit="ALL")
    summ = pd.concat([allu, summ], ignore_index=True)
    write_csv(summ, MOD, "benchmark_summary.csv",
              "Benchmark summary per dataset × unit × condition × rule (means over draws): estimated vs realised FDR, sensitivity, phase error",
              required=["dataset", "unit", "condition", "rule", "mean_est_fdr_perm", "mean_fdp_lenient", "mean_sensitivity"])
    ss = sens.groupby(["dataset", "condition", "rule", "truth_fc_bin", "expr_decile"], observed=True)[["n", "called"]].sum().reset_index()
    ss["sensitivity"] = ss["called"] / ss["n"]
    write_csv(ss, MOD, "benchmark_sensitivity.csv", "Sensitivity by truth amplitude bin × expression decile (pooled over draws and units)",
              required=["dataset", "condition", "rule", "truth_fc_bin", "expr_decile", "sensitivity"])
    cr = clk.groupby(["dataset", "condition", "gene"]).agg(n=("draw", "size"), frac_tier1=("tier1", "mean"), frac_tier1_or_2=("tier1_or_2", "mean"),
                                                           frac_units_truth_rhythmic=("truth_rhythmic", "mean")).reset_index()
    cr_t = clk[clk["truth_rhythmic"]].groupby(["dataset", "condition", "gene"]).agg(frac_tier1_or_2_when_truth=("tier1_or_2", "mean")).reset_index()
    write_csv(cr.merge(cr_t, how="left"), MOD, "benchmark_clock_recovery.csv",
              "Core-clock gene recovery (share of draws called Tier 1 / Tier 1–2), overall and where the held-out truth is rhythmic",
              required=["dataset", "condition", "gene", "frac_tier1", "frac_tier1_or_2"])
    if len(pf):
        pfs = pf.groupby(["dataset", "condition", "rule"]).mean(numeric_only=True).reset_index().drop(columns=["draw"])
        write_csv(pfs, MOD, "benchmark_pattern_flags.csv",
                  "Sample-pattern stress test: FDP with and without pattern-flagged calls (r > 0.8 with the nuisance profile), share of nuisance false calls flagged",
                  required=["dataset", "condition", "rule", "fdp_lenient_all", "fdp_lenient_unflagged"])
    log("\n" + summ[summ["unit"] == "ALL"].round(3).to_string())
    if len(pf):
        log("\n" + pfs.round(3).to_string())
