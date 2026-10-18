"""Round 3, WP4 — supplementary analyses on the atlas cell-level counts (labels unchanged).

(a) Label-free immune sub-states: within each immune cell type (cell_class Myeloid / Lymphoid / Progenitor) with
    >= 500 cells in a tissue — CP10k-log1p, 2,000 HVGs, PCA 30, kNN 15, Leiden (resolution 0.4) → sub-states with
    >= 2 % of the type's cells; share of each sub-state within its type per sample → OLS cosinor (ZT) with the exact
    60-class dihedral permutation p and BH within tissue; sample-specific sub-states flagged (> 50 % of cells from one
    sample); top Wilcoxon markers per sub-state. Answers whether type-level rhythms sit in particular states.
(b) Cell-level clock phase (light alternative to Tempo): per cell, z-scored log CP10k of the core-clock genes within its
    cell type, projected on the canonical peak phases, φ_cell = arg Σ_g z_g e^{iωφ_g} (cells with >= 3 clock UMIs);
    per sample × cell type the circular mean φ (weighted by the projection length) and its resultant R (within-sample
    synchrony); per cell type the consistency of the offset (sample mean φ − ZT) across the 6 samples (resultant R;
    a cell-level clock that tracks time gives a constant offset), exact p over the 60 dihedral relabelings of ZT
    (a circular correlation is undefined here because six equally spaced ZTs have no mean direction).
Outputs (results/r3_audit/): immune_substates.csv, immune_substate_shares.csv, cell_clock_phase_by_sample.csv,
cell_clock_phase_summary.csv
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import H5AD, log, sample_table, write_csv  # noqa: E402

MOD = "r3_audit"
TCODE = {"bone marrow": "bm", "cochlea": "cochlea"}
IMMUNE_CLASSES = {"Myeloid", "Lymphoid", "Progenitor"}
a = ad.read_h5ad(H5AD)
smp = sample_table(a.obs).set_index("sample_id")
a.obs["tcode"] = a.obs["tissue"].astype(str).map(TCODE)
assert a.obs["tcode"].notna().all()
a.obs["sample_id"] = a.obs["batch_id"].astype(str)
a.obs["zt"] = a.obs["sample_id"].map(smp["zt"]).astype(float)
classes = cosinor.dihedral_classes(6)


def perm_p(y, zt_sorted):
    """Exact 60-class dihedral permutation p of the OLS cosinor F (y in ZT order)."""
    _, _, pp, _ = cosinor.permutation_test(y[:, None], zt_sorted)
    return float(pp[0])


# ---- (a) immune sub-states
sub_rows, share_rows = [], []
for (t, ct), idx in a.obs.groupby(["tcode", "cell_type"], observed=True).indices.items():
    cls = str(a.obs["cell_class"].iloc[idx[0]])
    if cls not in IMMUNE_CLASSES or len(idx) < 500:
        continue
    s = a[idx].copy()
    sc.pp.normalize_total(s, target_sum=1e4)
    sc.pp.log1p(s)
    sc.pp.highly_variable_genes(s, n_top_genes=2000)
    s = s[:, s.var["highly_variable"]].copy()
    sc.pp.scale(s, max_value=10)
    sc.pp.pca(s, n_comps=30, random_state=0)
    sc.pp.neighbors(s, n_neighbors=15, random_state=0)
    sc.tl.leiden(s, resolution=0.4, random_state=0, flavor="igraph", n_iterations=2, directed=False)
    lab = s.obs["leiden"].astype(str)
    keep = lab.map(lab.value_counts(normalize=True)) >= 0.02
    if lab[keep].nunique() < 2:
        continue
    s2 = s[keep.to_numpy()].copy()
    sc.tl.rank_genes_groups(s2, "leiden", method="wilcoxon", n_genes=8)
    rg = s2.uns["rank_genes_groups"]["names"]
    tab = pd.crosstab(s2.obs["sample_id"], s2.obs["leiden"])
    order = sorted(tab.index, key=lambda x: smp.loc[x, "zt"])
    tab = tab.loc[order]
    zt = np.array([smp.loc[x, "zt"] for x in order], float)
    pct = 100 * tab.div(tab.sum(1), axis=0)
    for k in tab.columns:
        y = pct[k].to_numpy(float)
        f = cosinor.fit(y[:, None], zt)
        sub_rows.append({"tissue": t, "cell_type": ct, "substate": f"{ct}|s{k}", "n_cells": int(tab[k].sum()),
                         "frac_of_type": float(tab[k].sum() / tab.values.sum()),
                         "max_single_sample_share": float((tab[k] / tab[k].sum()).max()),
                         "markers": ";".join(list(rg[k])), "mean_pct_within_type": float(y.mean()),
                         "peak_zt": float(f["peak"][0]), "peak_clock": float(phase.zt_to_clock(f["peak"][0])),
                         "rel_amplitude": float(f["amp"][0] / max(y.mean(), 1e-9)), "p": float(f["p"][0]),
                         "p_perm": perm_p(y, zt)})
        for sid, v, n in zip(order, y, tab[k]):
            share_rows.append({"tissue": t, "cell_type": ct, "substate": f"{ct}|s{k}", "sample_id": sid,
                               "zt": smp.loc[sid, "zt"], "n_cells": int(n), "pct_within_type": float(v)})
    log(f"{t} {ct}: {tab.shape[1]} sub-states")
ss = pd.DataFrame(sub_rows)
ss["sample_specific"] = ss["max_single_sample_share"] > 0.5
ss["q_BH"] = ss.groupby("tissue")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
write_csv(ss, MOD, "immune_substates.csv",
          "Label-free sub-states within immune cell types: markers, sample specificity, within-type share rhythm (OLS, permutation)",
          required=["tissue", "cell_type", "substate", "markers", "peak_zt", "p", "p_perm", "q_BH"])
write_csv(pd.DataFrame(share_rows), MOD, "immune_substate_shares.csv", "Sub-state share within its type per sample (figure-ready)",
          required=["tissue", "cell_type", "substate", "sample_id", "zt", "pct_within_type"])

# ---- (b) cell-level clock phase
clock = [g for g in phase.CANONICAL_CLOCK if g in a.var_names]
gi = [a.var_names.get_loc(g) for g in clock]
ph_g = np.array([phase.CANONICAL_CLOCK[g] for g in clock])
E = np.exp(1j * phase.W * ph_g)
X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
lib = np.asarray(X.sum(1)).ravel()
C = X[:, gi].toarray()
L = np.log1p(C / lib[:, None] * 1e4)
by_rows, sum_rows = [], []
for (t, ct), idx in a.obs.groupby(["tcode", "cell_type"], observed=True).indices.items():
    ok = idx[C[idx].sum(1) >= 3]
    if len(ok) < 300:
        continue
    z = L[ok]
    z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1)
    v = z @ E
    phi = np.mod(np.angle(v) / phase.W, 24)
    w = np.abs(v)
    d = pd.DataFrame({"sample_id": a.obs["sample_id"].to_numpy()[ok], "phi": phi, "w": w})
    rows = []
    for sid, g in d.groupby("sample_id"):
        m, R = phase.circ_mean(g["phi"], g["w"])
        rows.append({"tissue": t, "cell_type": ct, "sample_id": sid, "zt": smp.loc[sid, "zt"], "n_cells": len(g),
                     "mean_cell_phase_h": m, "within_sample_resultant": R})
    r = pd.DataFrame(rows).sort_values("zt")
    by_rows += r.to_dict("records")
    if len(r) < 6:
        continue

    mph, zts = r["mean_cell_phase_h"].to_numpy(), r["zt"].to_numpy()
    off_, Roff = phase.circ_mean(phase.circ_diff(mph, zts) % 24)
    null = [phase.circ_mean(phase.circ_diff(mph, zts[np.array(k)]) % 24)[1] for k in classes]
    sum_rows.append({"tissue": t, "cell_type": ct, "n_cells_used": int(r["n_cells"].sum()), "n_clock_genes": len(clock),
                     "offset_resultant": Roff, "p_perm": float(np.mean(np.array(null) >= Roff - 1e-12)),
                     "mean_offset_cell_phase_minus_zt_h": float(phase.circ_diff(off_, 0)),
                     "median_within_sample_resultant": float(r["within_sample_resultant"].median())})
write_csv(pd.DataFrame(by_rows), MOD, "cell_clock_phase_by_sample.csv",
          "Cell-level clock phase (canonical-phase projection): per sample × cell type mean phase and within-sample synchrony",
          required=["tissue", "cell_type", "sample_id", "zt", "mean_cell_phase_h", "within_sample_resultant"])
cs = pd.DataFrame(sum_rows)
cs["q_BH"] = multipletests(cs["p_perm"], method="fdr_bh")[1] if len(cs) else []
write_csv(cs, MOD, "cell_clock_phase_summary.csv",
          "Does the per-sample mean cell-level clock phase track ZT? Offset consistency (resultant) with exact permutation p",
          required=["tissue", "cell_type", "offset_resultant", "p_perm"])
log("\n" + cs.sort_values("p_perm").head(20).round(3).to_string())
log("done")
