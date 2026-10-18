"""Module 5.4d — phasor decomposition of tissue-level (bulk-like) rhythms into cell-type expression vs abundance terms.

For tissue T and sample s, the tissue pseudobulk CPM of gene g is B_gs = Σ_c w_cs·μ_gcs, with w_cs the UMI share of
cell type c and μ_gcs its CPM. Around the time means: B = Σ w̄μ̄ + Σ w̄·δμ (expression) + Σ μ̄·δw (abundance)
+ Σ δw·δμ (interaction). Each term (relative to B̄_g) is summarised by its first-harmonic phasor
(cosinor on the 6 ZTs, linear scale); per gene we report the total phasor, the projections of the expression,
abundance and interaction phasors on it (fractions of the total rhythm), and the top contributing cell types.
Validation: predicted tissue rhythms vs measured external bulk rhythms of the same tissue (E1 cochlea sham,
E2 BM control): peak concordance (circular), amplitude correlation, for genes rhythmic in the external data.

Outputs (results/72_phasor/): phasor_decomposition_<tissue>.csv, phasor_celltype_contributions_<tissue>.csv,
                             phasor_validation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "72_phasor"
W = 2 * np.pi / 24
EXTERNAL_MATCH = {"cochlea": ("cochlea", "sham"), "bm": ("bone_marrow", "control")}


def phasor(Y, zt):
    """Y: samples × features (relative deviations). Complex amplitude c = a − i b and the fit."""
    f = cosinor.fit(Y, zt)
    return f["a"] - 1j * f["b"], f


val_rows = []
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
    d = design[(design["tissue"] == tissue) & (design["level"] == "cell_type")]
    samples = d.drop_duplicates("sample_id").sort_values("zt")
    zt = samples["zt"].to_numpy()
    groups = sorted(d["group"].unique())
    C = np.stack([pb[[f"{g}|{s}" for s in samples["sample_id"]]].to_numpy().T for g in groups], axis=1)  # S×K×G
    lib_cs = C.sum(2)                                   # S × K
    L = lib_cs.sum(1, keepdims=True)
    w = lib_cs / L                                      # UMI share
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(lib_cs[:, :, None] > 0, C / lib_cs[:, :, None] * 1e6, 0.0)
    B = (C.sum(1) / L) * 1e6                            # S × G tissue CPM
    keep = B.mean(0) >= 5
    genes = pb.index[keep]
    mu, B = mu[:, :, keep], B[:, keep]
    wbar, mubar = w.mean(0), mu.mean(0)                 # K ; K × G
    Bbar = B.mean(0)
    E = wbar[None, :, None] * (mu - mubar[None]) / Bbar[None, None, :]          # S × K × G
    A = (w - wbar)[:, :, None] * mubar[None] / Bbar[None, None, :]
    I = ((w - wbar)[:, :, None] * (mu - mubar[None])).sum(1) / Bbar[None, :]
    c_tot, f_tot = phasor(B / Bbar - 1, zt)
    S, K, G = E.shape
    cE, _ = phasor(E.reshape(S, K * G), zt)
    cA, _ = phasor(A.reshape(S, K * G), zt)
    cE, cA = cE.reshape(K, G), cA.reshape(K, G)
    cI, _ = phasor(I, zt)
    denom = np.abs(c_tot) ** 2 + 1e-12
    proj = lambda c: np.real(c * np.conj(c_tot)) / denom
    frac_E, frac_A, frac_I = proj(cE.sum(0)), proj(cA.sum(0)), proj(cI)
    projE, projA = np.real(cE * np.conj(c_tot)[None]) / denom, np.real(cA * np.conj(c_tot)[None]) / denom
    amp = np.abs(c_tot)
    out = pd.DataFrame({"tissue": tissue, "gene": genes, "mean_cpm": Bbar, "rel_amplitude": amp,
                        "ptt_ratio": (1 + amp) / np.clip(1 - amp, 1e-3, None),
                        "peak_zt": (np.angle(np.conj(c_tot)) % (2 * np.pi)) / W, "p": f_tot["p"],
                        "frac_expression": frac_E, "frac_abundance": frac_A, "frac_interaction": frac_I,
                        "top_expression_celltype": np.array(groups)[projE.argmax(0)],
                        "top_expression_proj": projE.max(0),
                        "top_abundance_celltype": np.array(groups)[projA.argmax(0)],
                        "top_abundance_proj": projA.max(0)})
    write_csv(out, MOD, f"phasor_decomposition_{tissue}.csv",
              f"Tissue-level rhythm per gene split into expression/abundance/interaction contributions ({tissue})",
              required=["tissue", "gene", "rel_amplitude", "peak_zt", "frac_expression", "frac_abundance"])
    sel = out["p"] < 0.05
    rows = []
    for k, gname in enumerate(groups):
        for term, cc in [("expression", cE[k]), ("abundance", cA[k])]:
            a_ = np.abs(cc[sel.values])
            rows.append(pd.DataFrame({"tissue": tissue, "gene": genes[sel.values], "cell_type": gname, "term": term,
                                      "rel_amplitude": a_, "peak_zt": (np.angle(np.conj(cc[sel.values])) % (2 * np.pi)) / W,
                                      "projection_on_total": (projE if term == "expression" else projA)[k, sel.values]}))
    contrib = pd.concat(rows, ignore_index=True)
    contrib = contrib[np.abs(contrib["projection_on_total"]) >= 0.05]
    write_csv(contrib, MOD, f"phasor_celltype_contributions_{tissue}.csv",
              f"Per cell type expression/abundance phasor contributions for nominally rhythmic tissue genes ({tissue})",
              required=["tissue", "gene", "cell_type", "term", "projection_on_total"])
    ext_t, ext_c = EXTERNAL_MATCH[tissue]
    for f in (RESULTS / "71_external").glob("external_rhythm_*.csv"):
        ext = pd.read_csv(f)
        ext = ext[(ext["tissue"] == ext_t) & (ext["condition"] == ext_c) & (ext["q_BH"] < 0.05)]
        if not len(ext):
            continue
        m = out.merge(ext[["gene", "peak_h", "amp_log2", "dataset", "time_type"]], on="gene")
        if len(m) < 10:
            continue
        dphi = cosinor.circ_diff(m["peak_zt"], m["peak_h"])
        ang1, ang2 = W * m["peak_zt"].to_numpy(), W * m["peak_h"].to_numpy()
        m1, m2 = np.angle(np.exp(1j * ang1).mean()), np.angle(np.exp(1j * ang2).mean())
        rho = np.sum(np.sin(ang1 - m1) * np.sin(ang2 - m2)) / np.sqrt(np.sum(np.sin(ang1 - m1) ** 2) *
                                                                      np.sum(np.sin(ang2 - m2) ** 2))
        val_rows.append({"tissue": tissue, "external_dataset": m["dataset"].iloc[0],
                         "external_time_type": m["time_type"].iloc[0], "n_genes": len(m),
                         "circ_corr_peaks": rho, "median_abs_peak_diff_h": float(np.median(np.abs(dphi))),
                         "circ_mean_peak_diff_h": float(np.angle(np.exp(1j * W * dphi).mean()) / W),
                         "frac_within_3h": float(np.mean(np.abs(dphi) <= 3)),
                         "spearman_amplitude": spearmanr(m["rel_amplitude"], m["amp_log2"]).correlation})
    log(f"{tissue}: {len(out)} genes; median frac_expression {np.median(frac_E[sel.values]):.2f}, "
        f"frac_abundance {np.median(frac_A[sel.values]):.2f} (p<0.05 genes)")
if val_rows:
    write_csv(pd.DataFrame(val_rows), MOD, "phasor_validation.csv",
              "Predicted tissue-level rhythms vs measured external bulk rhythms (peak concordance, amplitude)",
              required=["tissue", "external_dataset", "n_genes", "circ_corr_peaks", "median_abs_peak_diff_h"])
log("done")
