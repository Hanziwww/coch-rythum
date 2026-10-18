"""Round 2, WP0 — time-axis audit of every result CSV (ZT = clock − 8; lights on 08:00).

(1) Column scan: every column whose name suggests a phase or time (peak, phase, acrophase, trough, zt, clock, time_h,
    lag, offset) in every results CSV: n non-missing, min, max, and a range check ([0, 24) for phases and times,
    (−12, 12] for signed differences/lags/offsets).
(2) Recomputation checks:
    a. design: the six sample ZTs are {3, 7, 11, 15, 19, 23} per tissue and ZT = clock − 8 for every sample;
    b. abundance frames: peaks re-fitted (OLS cosinor) from composition_frames.csv vs the stored peak_zt;
    c. gene phases: core-clock genes re-fitted (OLS on log2 CPM of the pseudobulk counts) vs the primary limma peak_zt;
    d. canonical clock order: circular mean offset of observed − canonical clock-gene peaks per tissue (expected
       ≈ +1.6–1.8 h); a clock-labelled axis would show ≈ +8 h.
Outputs (results/r2_audit/): time_axis_columns.csv, time_axis_checks.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import CACHE, LIGHTS_ON, RESULTS, log, write_csv  # noqa: E402

MOD = "r2_audit"
PAT = re.compile(r"peak|phase|acrophase|trough|(^|_)zt($|_)|clock|time_h|lag|offset", re.I)
SIGNED = re.compile(r"diff|delta|lag|offset|shift", re.I)
CLOCK = ["Npas2", "Clock", "Nr1d1", "Nr1d2", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Dbp", "Tef", "Hlf", "Nfil3",
         "Bhlhe40", "Bhlhe41", "Ciart"]

cols = []
for p in sorted(RESULTS.rglob("*.csv")):
    rel = str(p.relative_to(RESULTS))
    if rel.startswith("r2_audit") or "/counts/" in rel.replace("\\", "/"):
        continue
    try:
        head = pd.read_csv(p, nrows=0).columns
    except Exception:
        continue
    hit = [c for c in head if PAT.search(c) and not re.search(r"_p$|_q$|^p_|pvalue|fdr|n_|frac|resultant|ci_half|sd_h|"
                                                                r"window|clock_tf|cue|set$|^n$", c, re.I)]
    if not hit:
        continue
    d = pd.read_csv(p, usecols=hit, low_memory=False)
    for c in hit:
        v = pd.to_numeric(d[c], errors="coerce").dropna()
        if v.empty:
            continue
        signed = bool(SIGNED.search(c))
        in_phase = bool(((v >= 0) & (v < 24.0001)).all())
        in_signed = bool(((v > -12) & (v <= 12)).all())
        if signed and not in_signed and in_phase:
            ok, note = True, "lag defined modulo 24 (phase-type)"
        elif signed and not in_signed and re.search(r"_(lo|hi|low|high)$", c):
            ok, note = True, "CI bound unwrapped around its point estimate"
        else:
            ok, note = (in_signed if signed else in_phase), ""
        cols.append({"file": rel, "column": c, "n": int(v.size), "min": float(v.min()), "max": float(v.max()),
                     "expected_range": "(-12, 12]" if signed else "[0, 24)", "range_ok": ok, "note": note})
write_csv(pd.DataFrame(cols), MOD, "time_axis_columns.csv",
          "Phase/time columns of every result CSV with range checks", required=["file", "column", "range_ok"])

checks = []
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
s = design.drop_duplicates("sample_id")
checks.append({"check": "design: six ZTs per tissue {3,7,11,15,19,23}",
               "value": ";".join(f"{t}:{sorted(g['zt'].astype(int).tolist())}" for t, g in s.groupby("tissue")),
               "pass": all(sorted(g["zt"].astype(int)) == [3, 7, 11, 15, 19, 23] for _, g in s.groupby("tissue"))})
mism = int((((s["clock_time"] - LIGHTS_ON) % 24 - s["zt"]).abs() > 1e-6).sum())
checks.append({"check": "design: ZT = (clock − 8) mod 24 for every sample", "value": f"{mism} mismatches", "pass": mism == 0})

fr = pd.read_csv(RESULTS / "50_abundance/composition_frames.csv")
st = pd.read_csv(RESULTS / "50_abundance/abundance_rhythm_frames.csv")
dev = []
for (f, t, feat), g in fr.groupby(["frame", "tissue", "feature"]):
    g = g.sort_values("zt")
    if len(g) != 6:
        continue
    pk = float(cosinor.fit(g[["value_log2"]].to_numpy(), g["zt"].to_numpy())["peak"][0])
    ref = st[(st["frame"] == f) & (st["tissue"] == t) & (st["feature"] == feat)]["peak_zt"]
    if len(ref):
        dev.append(abs(float(phase.circ_diff(pk, ref.iloc[0]))))
checks.append({"check": "abundance frames: re-fitted vs stored peak_zt (max |Δ| h)", "value": f"{max(dev):.4f} over {len(dev)} series",
               "pass": max(dev) < 0.01})

dev = []
for tissue in ["bm", "cochlea"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    lim = pd.read_csv(RESULTS / f"31_rhythm_genes/rhythm_limma_cell_type_{tissue}.csv")
    lim = lim[lim["gene"].isin(CLOCK) & (lim["p"] < 0.01)]
    for r in lim.itertuples():
        d = design[(design["tissue"] == tissue) & (design["level"] == "cell_type") & (design["group"] == r.group)].sort_values("zt")
        cnt = pb[[f"{r.group}|{x}" for x in d["sample_id"]]]
        lcpm = np.log2(cnt.loc[r.gene] / cnt.sum(0) * 1e6 + 1).to_numpy()[:, None]
        pk = float(cosinor.fit(lcpm, d["zt"].to_numpy())["peak"][0])
        dev.append({"tissue": tissue, "cell_type": r.group, "gene": r.gene,
                    "abs_diff_h": abs(float(phase.circ_diff(pk, r.peak_zt)))})
dv = pd.DataFrame(dev)
checks.append({"check": "core-clock genes (limma p < 0.01): OLS log2CPM re-fit vs limma peak_zt (median / 90th pct |Δ| h)",
               "value": f"{dv['abs_diff_h'].median():.2f} / {dv['abs_diff_h'].quantile(0.9):.2f} (n={len(dv)})",
               "pass": dv["abs_diff_h"].median() < 1.0})

cg = pd.read_csv(RESULTS / "02_clock_controls/clock_gene_rhythm.csv")
cg = cg[cg["expressed"].astype(str) == "True"]
cg = cg[cg["p"] < 0.05].dropna(subset=["peak_zt", "canonical_peak_zt"])
for t, g in cg.groupby("tissue"):
    m, R = phase.circ_mean(phase.circ_diff(g["peak_zt"], g["canonical_peak_zt"]) % 24)
    off = float(phase.circ_diff(m, 0))
    checks.append({"check": f"{t}: circular mean offset observed − canonical clock-gene peak (h)",
                   "value": f"{off:+.2f} (resultant {R:.2f}, n={len(g)})", "pass": abs(off) < 4})
write_csv(pd.DataFrame(checks), MOD, "time_axis_checks.csv", "Recomputation checks of the ZT axis",
          required=["check", "value", "pass"])
log(f"{len(cols)} phase/time columns scanned ({sum(not c['range_ok'] for c in cols)} out of range); "
    f"checks passed {sum(c['pass'] for c in checks)}/{len(checks)}")
