"""Module 0.4a — core-clock positive controls and time-label checks (data-internal part).

For every included tissue × cell type: pseudobulk log2CPM profiles of core-clock genes, cosinor fits with
60-class permutation p and LOTO, and a clock phase offset = amplitude-weighted circular mean of
(observed peak − canonical peak) across clock genes, with phase coherence and a circular correlation between
observed and canonical peaks. Arntl is absent from the matrix; Npas2 (E-box partner, Arntl phase) stands in.
Cochlea and BM are independent cohorts, so agreement of their offsets independently validates the ZT labels.

Canonical peaks (mouse peripheral tissues, LD 12:12, ZT h) are literature priors; Module 0.4b re-derives them
from external time series (CircaDB, GSE88954, GSE198770). Sample times are ZT = clock hour − 8 (lights on 08:00);
the first run on raw clock hours showed a uniform +8–10 h offset in both cohorts, which led to this correction.

Outputs (results/02_clock_controls/): clock_gene_profiles.csv, clock_gene_rhythm.csv,
clock_phase_order.csv, clock_crosstissue_concordance.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "02_clock_controls"
CANONICAL = {"Npas2": 23.5, "Nfil3": 23.0, "Cry1": 20.0, "Rorc": 19.5, "Nr1d1": 7.0, "Nr1d2": 9.0, "Dbp": 10.0,
             "Tef": 12.0, "Hlf": 12.0, "Per3": 10.0, "Per1": 11.5, "Per2": 14.0, "Cry2": 11.0, "Ciart": 12.5}
EXTRA = ["Clock", "Bhlhe40", "Bhlhe41", "Rora", "Cirbp", "Fkbp5", "Zbtb16", "Sgk1", "Klf9", "Dusp1"]
W = 2 * np.pi / 24

incl = pd.read_csv(RESULTS / "30_pseudobulk/groups_included.csv")
incl = incl[(incl["level"] == "cell_type") & (incl["status"] != "excluded")]
design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")

prof_rows, fit_rows, order_rows = [], [], []
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    lib = pb.sum(0)
    genes = [g for g in list(CANONICAL) + EXTRA if g in pb.index]
    logcpm = np.log2(pb.loc[genes] / lib * 1e6 + 1)
    for _, row in incl[incl["tissue"] == tissue].iterrows():
        group = row["group"]
        d = design[(design["tissue"] == tissue) & (design["level"] == "cell_type") & (design["group"] == group)]
        d = d.sort_values("zt")
        cols = [f"{group}|{s}" for s in d["sample_id"]]
        zt = d["zt"].to_numpy()
        Y = logcpm[cols].T.to_numpy()                       # samples × genes
        counts = pb.loc[genes, cols].T.to_numpy()
        for j, s in enumerate(d["sample_id"]):
            for k, g in enumerate(genes):
                prof_rows.append({"tissue": tissue, "cell_type": group, "status": row["status"], "sample_id": s,
                                  "zt": zt[j], "gene": g, "count": counts[j, k], "log2cpm": Y[j, k]})
        f = cosinor.fit(Y, zt)
        _, _, p_perm, _ = cosinor.permutation_test(Y, zt)
        lo = cosinor.loto(Y, zt)
        expressed = (counts >= 5).sum(0) >= 4
        for k, g in enumerate(genes):
            fit_rows.append({"tissue": tissue, "cell_type": group, "status": row["status"], "gene": g,
                             "expressed": bool(expressed[k]), "mean_log2cpm": Y[:, k].mean(),
                             "mesor": f["mesor"][k], "amp_log2": f["amp"][k], "ptt_fc": 2 ** (2 * f["amp"][k]),
                             "peak_zt": f["peak"][k], "canonical_peak_zt": CANONICAL.get(g, np.nan),
                             "F": f["F"][k], "p": f["p"][k], "p_perm": p_perm[k],
                             "loto_phase_sd_h": lo["loto_phase_sd_h"][k], "loto_max_p": lo["loto_max_p"][k]})
        # clock phase offset across canonical genes
        sel = [k for k, g in enumerate(genes) if g in CANONICAL and expressed[k]]
        if len(sel) >= 4:
            obs_peak = f["peak"][sel]
            can = np.array([CANONICAL[genes[k]] for k in sel])
            amp = f["amp"][sel]
            z = amp * np.exp(1j * W * (obs_peak - can))
            m = z.sum() / amp.sum()
            offset = (np.angle(m) / W) % 24
            offset = offset - 24 if offset > 12 else offset
            # circular correlation (Fisher & Lee) between observed and canonical peaks
            a1, a2 = W * obs_peak, W * can
            num = np.sum(np.sin(a1 - a1.mean()) * np.sin(a2 - a2.mean()))
            # use circular means for centring
            m1, m2 = np.angle(np.exp(1j * a1).mean()), np.angle(np.exp(1j * a2).mean())
            num = np.sum(np.sin(a1 - m1) * np.sin(a2 - m2))
            den = np.sqrt(np.sum(np.sin(a1 - m1) ** 2) * np.sum(np.sin(a2 - m2) ** 2))
            rho = num / den if den > 0 else np.nan
            rng = np.random.default_rng(0)
            null = []
            for _ in range(2000):
                a2p = rng.permutation(a2)
                m2p = np.angle(np.exp(1j * a2p).mean())
                nn = np.sum(np.sin(a1 - m1) * np.sin(a2p - m2p))
                dd = np.sqrt(np.sum(np.sin(a1 - m1) ** 2) * np.sum(np.sin(a2p - m2p) ** 2))
                null.append(nn / dd if dd > 0 else 0)
            p_rho = (1 + np.sum(np.array(null) >= rho)) / (len(null) + 1)
            n_sig = int(np.sum(f["p"][sel] < 0.05))
            order_rows.append({"tissue": tissue, "cell_type": group, "status": row["status"],
                               "n_clock_genes_expressed": len(sel), "n_clock_genes_p05": n_sig,
                               "clock_phase_offset_h": offset, "phase_coherence": float(np.abs(m)),
                               "median_clock_amp_log2": float(np.median(amp)),
                               "circ_corr_observed_vs_canonical": rho, "p_circ_corr": p_rho})

prof = pd.DataFrame(prof_rows)
fits = pd.DataFrame(fit_rows)
order = pd.DataFrame(order_rows)
write_csv(prof, MOD, "clock_gene_profiles.csv", "Pseudobulk counts and log2CPM of clock/control genes per sample",
          required=["tissue", "cell_type", "sample_id", "zt", "gene", "log2cpm"])
write_csv(fits, MOD, "clock_gene_rhythm.csv", "Cosinor fits of clock/control genes per tissue × cell type",
          required=["tissue", "cell_type", "gene", "peak_zt", "p", "p_perm"])
write_csv(order, MOD, "clock_phase_order.csv",
          "Clock phase offset vs canonical ZT peaks, coherence and circular correlation per tissue × cell type",
          required=["tissue", "cell_type", "clock_phase_offset_h", "phase_coherence"])

# cross-tissue concordance (independent cohorts)
cc = order.pivot(index="cell_type", columns="tissue", values=["clock_phase_offset_h", "phase_coherence"])
cc.columns = [f"{a}_{b}" for a, b in cc.columns]
cc = cc.dropna().reset_index()
if len(cc):
    cc["offset_diff_cochlea_minus_bm_h"] = cosinor.circ_diff(cc["clock_phase_offset_h_cochlea"],
                                                             cc["clock_phase_offset_h_bm"])
write_csv(cc, MOD, "clock_crosstissue_concordance.csv",
          "Clock phase offsets of shared cell types in the independent cochlea and BM cohorts",
          required=["cell_type"])
for t, g in order.groupby("tissue"):
    z = np.exp(1j * W * g["clock_phase_offset_h"].to_numpy()) * g["phase_coherence"].to_numpy()
    off = (np.angle(z.mean()) / W)
    log(f"{t}: {len(g)} cell types; pooled clock offset {off:+.2f} h; median coherence "
        f"{g['phase_coherence'].median():.2f}; median circ corr {g['circ_corr_observed_vs_canonical'].median():.2f}")
log("done")
