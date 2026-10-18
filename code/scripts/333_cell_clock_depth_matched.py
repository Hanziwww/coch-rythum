"""Round 10 (D113) — R1: is the absent single-cell clock tracking of cochlear immune cells a matter of depth?

332 (b) projects each cell's z-scored log CP10k core-clock profile on the canonical peak phases (cells with >= 3 clock
UMIs; cell types with >= 300 such cells) and asks whether the per-sample mean cell phase keeps a constant offset from
ZT across the six samples (offset resultant; exact p over the 60 dihedral relabelings). In the cochlea 10 of 12
structural types tracked ZT and none of 10 haematopoietic types did, but haematopoietic cells carry far fewer clock
UMIs.

Primary test (rule fixed before running; plan round 10): every cell of a cochlear structural type is thinned to the
clock-UMI depth of the cochlear haematopoietic cells. Its target clock-UMI total is the haematopoietic quantile at the
cell's own quantile within its type (ties broken at random; never above its own total), and all its counts are thinned
by the same fraction: binomially for the clock genes, by expectation for the library size. 332 (b) then runs unchanged.
20 replicates. A structural type 'tracks at haematopoietic depth' if exact p <= 0.05 in >= 50 % of replicates, counted
over all 20 replicates: a replicate in which fewer than 300 cells keep >= 3 clock UMIs cannot be tested and counts as
not tracking (the share among testable replicates is reported alongside; the first run counted only testable
replicates in the decision, corrected to the rule as written).
Decision: if at least half of the structural types track at haematopoietic depth (the haematopoietic types do not
track at their own depth, 332), the single-cell contrast stands; otherwise it is called power-limited.
Secondary (reported, not decisive): (i) depth and cell number matched: after thinning, each structural type keeps at
most the median number of cells used by the cochlear haematopoietic types (random subsample of the passing cells);
(ii) natural depth stratum: every cochlear type restricted to its cells with 3-5 clock UMIs, no thinning (>= 100 cells).
Baseline check: without thinning, the re-implementation reproduces results/r3_audit/cell_clock_phase_summary.csv
(offset_resultant, p_perm) for every cell type of both tissues.
Outputs (results/r10_reanalysis/): cell_clock_depth_matched.csv (per cochlear cell type and variant),
cell_clock_depth_replicates.csv (every replicate), cell_clock_depth_reference.csv (clock-UMI distributions),
cell_clock_depth_decision.csv (the pre-specified rule applied).
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import H5AD, RESULTS, log, sample_table, write_csv  # noqa: E402

MOD = "r10_reanalysis"
TCODE = {"bone marrow": "bm", "cochlea": "cochlea"}
HEME = {"Myeloid", "Lymphoid", "Progenitor", "Erythroid", "Cycling"}
N_REP, MIN_CELLS, MIN_STRATUM = 20, 300, 100
rng = np.random.default_rng(10)

a = ad.read_h5ad(H5AD)
smp = sample_table(a.obs).set_index("sample_id")
tcode = a.obs["tissue"].astype(str).map(TCODE).to_numpy()
assert pd.notna(tcode).all()
sid = a.obs["batch_id"].astype(str).to_numpy()
ctype = a.obs["cell_type"].astype(str).to_numpy()
cclass = a.obs["cell_class"].astype(str).to_numpy()
zt_of = smp["zt"].astype(float)
classes = cosinor.dihedral_classes(6)

# exactly as 332 (b)
clock = [g for g in phase.CANONICAL_CLOCK if g in a.var_names]
gi = [a.var_names.get_loc(g) for g in clock]
ph_g = np.array([phase.CANONICAL_CLOCK[g] for g in clock])
E = np.exp(1j * phase.W * ph_g)
X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
lib = np.asarray(X.sum(1)).ravel()
C = X[:, gi].toarray()
del X, a
T = C.sum(1)                                                  # clock UMIs per cell


def track(Cm, libm, ids, min_cells=MIN_CELLS, n_max=None, lo=3, hi=None):
    """332 (b) for one cell type. Cm: clock counts (cells x genes); libm: library sizes; ids: sample ids."""
    tot = Cm.sum(1)
    keep = np.where((tot >= lo) & ((tot <= hi) if hi is not None else True))[0]
    if n_max is not None and len(keep) > n_max:
        keep = np.sort(rng.choice(keep, n_max, replace=False))
    if len(keep) < min_cells:
        return None
    z = np.log1p(Cm / libm[:, None] * 1e4)[keep]
    z = (z - z.mean(0)) / np.where(z.std(0) > 0, z.std(0), 1)
    v = z @ E
    d = pd.DataFrame({"sample_id": ids[keep], "phi": np.mod(np.angle(v) / phase.W, 24), "w": np.abs(v)})
    rows = []
    for s, g in d.groupby("sample_id"):
        m, R = phase.circ_mean(g["phi"], g["w"])
        rows.append({"zt": zt_of[s], "n": len(g), "mph": m, "R": R})
    r = pd.DataFrame(rows).sort_values("zt")
    if len(r) < 6:
        return None
    mph, zts = r["mph"].to_numpy(), r["zt"].to_numpy()
    _, Roff = phase.circ_mean(phase.circ_diff(mph, zts) % 24)
    null = [phase.circ_mean(phase.circ_diff(mph, zts[np.array(k)]) % 24)[1] for k in classes]
    return {"n_cells_used": int(r["n"].sum()), "offset_resultant": float(Roff),
            "p_perm": float(np.mean(np.array(null) >= Roff - 1e-12)),
            "median_within_sample_resultant": float(r["R"].median()),
            "median_clock_umis_used": float(np.median(tot[keep]))}


# ---- baseline: reproduce 332 (b) for every cell type of both tissues
groups = pd.DataFrame({"t": tcode, "ct": ctype}).groupby(["t", "ct"]).indices
base = {}
for (t, ct), idx in groups.items():
    res = track(C[idx], lib[idx], sid[idx])
    if res is not None:
        base[(t, ct)] = res
ref = pd.read_csv(RESULTS / "r3_audit" / "cell_clock_phase_summary.csv").set_index(["tissue", "cell_type"])
assert set(base) == set(ref.index), (set(base) ^ set(ref.index))
for k, res in base.items():
    assert np.isclose(res["offset_resultant"], ref.loc[k, "offset_resultant"], rtol=0, atol=1e-9), k
    assert np.isclose(res["p_perm"], ref.loc[k, "p_perm"], rtol=0, atol=1e-12), k
    assert res["n_cells_used"] == ref.loc[k, "n_cells_used"], k
log(f"baseline reproduced for {len(base)} cell types (offset resultant, exact p, cells used)")

# ---- groups in the cochlea
coch = {ct: idx for (t, ct), idx in groups.items() if t == "cochlea" and (t, ct) in base}
cls_of = {ct: cclass[idx[0]] for ct, idx in coch.items()}
heme = sorted(ct for ct in coch if cls_of[ct] in HEME)
struct = sorted(ct for ct in coch if cls_of[ct] not in HEME)
ref_T = np.sort(np.concatenate([T[coch[ct]] for ct in heme]))
n_heme_median = int(np.median([base[("cochlea", ct)]["n_cells_used"] for ct in heme]))
log(f"cochlea: {len(struct)} structural, {len(heme)} haematopoietic types; haematopoietic clock UMIs median "
    f"{np.median(ref_T):.0f}; median cells used {n_heme_median}")


def thin(idx):
    """Thin the cells of one structural type to the haematopoietic clock-UMI distribution (quantile matching)."""
    t0 = T[idx]
    u = (np.argsort(np.argsort(t0 + rng.random(len(t0)))) + 0.5) / len(t0)
    target = np.minimum(np.quantile(ref_T, u, method="inverted_cdf"), t0)
    frac = np.where(t0 > 0, target / np.maximum(t0, 1), 1.0)
    return rng.binomial(C[idx].astype(np.int64), frac[:, None]).astype(C.dtype), (lib[idx] * frac).astype(lib.dtype)


rep_rows = []
for rep in range(1, N_REP + 1):
    for ct in struct:
        idx = coch[ct]
        Ct, lt = thin(idx)
        for variant, n_max in (("depth matched", None), ("depth and cell number matched", n_heme_median)):
            res = track(Ct, lt, sid[idx], n_max=n_max)
            rep_rows.append({"cell_type": ct, "group": "structural", "variant": variant, "replicate": rep,
                             **(res or {"n_cells_used": 0, "offset_resultant": np.nan, "p_perm": np.nan,
                                        "median_within_sample_resultant": np.nan, "median_clock_umis_used": np.nan})})
    log(f"replicate {rep} done")
reps = pd.DataFrame(rep_rows)

# ---- summary per cochlear cell type and variant
rows = []
for ct in struct + heme:
    b = base[("cochlea", ct)]
    rows.append({"cell_type": ct, "group": "structural" if ct in struct else "haematopoietic", "variant": "own depth (332)",
                 "n_replicates": 1, "n_testable": 1, "frac_tracking": float(b["p_perm"] <= 0.05),
                 "frac_tracking_testable": float(b["p_perm"] <= 0.05), "median_p_perm": b["p_perm"],
                 "median_offset_resultant": b["offset_resultant"], "median_cells_used": b["n_cells_used"],
                 "median_within_sample_resultant": b["median_within_sample_resultant"],
                 "median_clock_umis_used": b["median_clock_umis_used"]})
for (ct, variant), g in reps.groupby(["cell_type", "variant"]):
    ok = g["p_perm"].notna()
    rows.append({"cell_type": ct, "group": "structural", "variant": variant, "n_replicates": len(g),
                 "n_testable": int(ok.sum()), "frac_tracking": float((g["p_perm"] <= 0.05).sum() / len(g)),
                 "frac_tracking_testable": float((g.loc[ok, "p_perm"] <= 0.05).mean()) if ok.any() else np.nan,
                 "median_p_perm": float(g.loc[ok, "p_perm"].median()) if ok.any() else np.nan,
                 "median_offset_resultant": float(g.loc[ok, "offset_resultant"].median()) if ok.any() else np.nan,
                 "median_cells_used": float(g["n_cells_used"].median()),
                 "median_within_sample_resultant": float(g.loc[ok, "median_within_sample_resultant"].median()) if ok.any() else np.nan,
                 "median_clock_umis_used": float(g.loc[ok, "median_clock_umis_used"].median()) if ok.any() else np.nan})
for ct in struct + heme:                                      # natural depth stratum, no thinning
    idx = coch[ct]
    res = track(C[idx], lib[idx], sid[idx], min_cells=MIN_STRATUM, lo=3, hi=5)
    rows.append({"cell_type": ct, "group": "structural" if ct in struct else "haematopoietic",
                 "variant": "natural stratum 3-5 clock UMIs", "n_replicates": 1, "n_testable": 1 if res else 0,
                 "frac_tracking": float(res["p_perm"] <= 0.05) if res else 0.0,
                 "frac_tracking_testable": float(res["p_perm"] <= 0.05) if res else np.nan,
                 "median_p_perm": res["p_perm"] if res else np.nan,
                 "median_offset_resultant": res["offset_resultant"] if res else np.nan,
                 "median_cells_used": res["n_cells_used"] if res else 0,
                 "median_within_sample_resultant": res["median_within_sample_resultant"] if res else np.nan,
                 "median_clock_umis_used": res["median_clock_umis_used"] if res else np.nan})
summ = pd.DataFrame(rows)

# ---- pre-specified decision (primary: depth matched)
prim = summ[(summ["variant"] == "depth matched")]
n_track = int((prim["frac_tracking"] >= 0.5).sum())
n_track_testable = int((prim["frac_tracking_testable"] >= 0.5).sum())
n_untestable = int((prim["n_testable"] == 0).sum())
heme_own = summ[(summ["group"] == "haematopoietic") & (summ["variant"] == "own depth (332)")]
strat = summ[summ["variant"] == "natural stratum 3-5 clock UMIs"]
dec = pd.DataFrame([{"n_structural_types": len(struct), "n_tracking_at_haematopoietic_depth": n_track,
                     "n_tracking_among_testable_replicates": n_track_testable,
                     "n_structural_untestable_at_haematopoietic_depth": n_untestable,
                     "threshold": int(np.ceil(len(struct) / 2)),
                     "n_haematopoietic_types": len(heme), "n_haematopoietic_tracking_own_depth": int((heme_own["frac_tracking"] > 0).sum()),
                     "stratum_structural_tracking": int((strat[strat["group"] == "structural"]["frac_tracking"] > 0).sum()),
                     "stratum_haematopoietic_tracking": int((strat[strat["group"] == "haematopoietic"]["frac_tracking"] > 0).sum()),
                     "decision": "single-cell contrast stands" if n_track >= np.ceil(len(struct) / 2) and (heme_own["frac_tracking"] == 0).all()
                     else "single-cell contrast power-limited"}])

# ---- clock-UMI reference distributions
q = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
ref_rows = []
for name, cts in (("cochlear structural (all types)", struct), ("cochlear haematopoietic (all types)", heme)):
    v = np.concatenate([T[coch[ct]] for ct in cts])
    ref_rows.append({"group": name, "cell_type": "", "n_cells": len(v), "frac_ge3": float((v >= 3).mean()),
                     **{f"q{int(x * 100)}": float(np.quantile(v, x)) for x in q}})
for ct in struct + heme:
    v = T[coch[ct]]
    ref_rows.append({"group": "structural" if ct in struct else "haematopoietic", "cell_type": ct, "n_cells": len(v),
                     "frac_ge3": float((v >= 3).mean()), **{f"q{int(x * 100)}": float(np.quantile(v, x)) for x in q}})

write_csv(summ, MOD, "cell_clock_depth_matched.csv", "R1: single-cell clock tracking of cochlear cell types at their own "
          "depth (332), at the clock-UMI depth of cochlear haematopoietic cells (20 thinning replicates; with and without "
          "cell-number matching) and in the natural 3-5 clock-UMI stratum (333)",
          required=["cell_type", "group", "variant", "frac_tracking", "median_p_perm", "median_offset_resultant"])
write_csv(reps, MOD, "cell_clock_depth_replicates.csv", "R1: every thinning replicate of the cochlear structural types (333)",
          required=["cell_type", "variant", "replicate", "p_perm", "offset_resultant"])
write_csv(pd.DataFrame(ref_rows), MOD, "cell_clock_depth_reference.csv", "R1: clock-UMI distributions of cochlear "
          "structural and haematopoietic cells (333)", required=["group", "cell_type", "n_cells", "q50"])
write_csv(dec, MOD, "cell_clock_depth_decision.csv", "R1: the pre-specified decision rule applied (333)",
          required=["n_structural_types", "n_tracking_at_haematopoietic_depth", "decision"])
pd.set_option("display.width", 250)
log("\n" + summ.round(3).to_string(index=False))
log("\n" + dec.to_string(index=False))
