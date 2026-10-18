"""Round 4, WP-E — positioning of the atlas against independent adult mouse cochlear single-cell / single-nucleus atlases
(labels transferred onto our gold standard by 300 → 301 → 302; our labels are never changed).

Datasets: GSE283534 (10-week whole cochlea, 6 replicates), GSE135913 (adult cochlea, epithelial-sorted + negative
fractions), GSE234926 (P90 cochlear-duct Fbxo2+ nuclei; author cell-type labels), GSE274279 (snRNA cochlea 3 / 12 / 24 months).
(1) atlas_concordance_summary.csv: per dataset — cells after QC, confident share, number of our cell types recovered
    (≥ 20 confident cells), marker-check pass rate (302: our markers higher in the transferred label, z ≥ 0.5).
(2) atlas_coverage.csv: our cell type × dataset — confident cells and share; marker z.
(3) author_label_concordance.csv / author_label_agreement.csv (GSE234926): contingency of author labels vs transferred
    labels, dominant transferred label per author type, agreement with the prior-knowledge correspondence (below) and
    adjusted Rand index.
(4) composition_similarity.csv: Spearman correlation of structural-cell-type shares (confident cells) between our cochlear
    samples (mean over ZTs) and each whole-cochlea external sample.
Outputs (results/r4_concordance/)
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_concordance"
KEYS = {"gse283534": "GSE283534 (10-week whole cochlea)", "gse135913": "GSE135913 (adult cochlea, sorted fractions)",
        "gse234926": "GSE234926 (P90 cochlear duct, Fbxo2+ nuclei)", "gse274279": "GSE274279 (snRNA cochlea, 3/12/24 months)"}
# prior-knowledge correspondence of GSE234926 author labels to our gold-standard labels
EXPECT = {"IHC": {"Hair Cell (Outer/Inner)"}, "OHC": {"Hair Cell (Outer/Inner)"},
          "DC/PC_1": {"Epithelial Cell (Non-sensory)"}, "DC/PC_2": {"Epithelial Cell (Non-sensory)"},
          "IPhC/IB": {"Epithelial Cell (Non-sensory)"}, "OSC_1": {"Epithelial Cell (Non-sensory)"}, "OSC_2": {"Epithelial Cell (Non-sensory)"},
          "RM": {"Epithelial Cell (Non-sensory)"}, "SpC/RC": {"Epithelial Cell (Non-sensory)"},
          "MC": {"SV Marginal Cell"}, "IC": {"SV Intermediate Cell (Functional)", "SV Intermediate Cell (Pigmented)"},
          "BC": {"Fibrocyte (Spiral Ligament)", "Fibrocyte (Cochlear)", "Epithelial Cell (Non-sensory)"},
          "FC": {"Fibrocyte (Spiral Ligament)", "Fibrocyte (Cochlear)"}, "FC_1": {"Fibrocyte (Spiral Ligament)", "Fibrocyte (Cochlear)"},
          "FC_2": {"Fibrocyte (Spiral Ligament)", "Fibrocyte (Cochlear)"}, "TBC": {"Fibrocyte (Cochlear)", "Fibrocyte (Spiral Ligament)", "Osteoprogenitor"},
          "Interdental": {"Interdental Cell"}, "neuronal": {"Spiral Ganglion Neuron"}, "Glia": {"Schwann Cell", "Oligodendrocyte"},
          "Vascular_Cd34p": {"Endothelial Cell"}}
# broad correspondence: our gold-standard 'Interdental Cell' label is marked by pan-duct epithelial genes (top markers Fbxo2, Epcam,
# Eya4, Zbtb20; 00_audit/label_markers.csv), so non-sensory duct epithelia (supporting cells, outer sulcus, Reissner's
# membrane) are also expected there. The strict (pre-specified) correspondence is reported alongside.
DUCT = ["DC/PC_1", "DC/PC_2", "IPhC/IB", "OSC_1", "OSC_2", "RM", "SpC/RC"]
EXPECT_BROAD = {k: (v | {"Interdental Cell"} if k in DUCT else v) for k, v in EXPECT.items()}
IMMUNE_PREFIX = ("B Cell", "Pre-B", "Pro-B", "Neutrophil", "Monocyte", "Macrophage", "Mast", "NK", "T Cell", "pDC", "HSC", "Granulocyte",
                 "Cycling Cell", "Erythroblast")

ours = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
ours = ours[ours["tissue"] == "cochlea"]
our_types = sorted(ours["cell_type"].unique())
summ, cov, comp = [], [], []
for key, label in KEYS.items():
    lt_p = RESULTS / f"r3_ingest/{key}_label_transfer.csv"
    if not lt_p.exists():
        log(f"{key}: not mapped yet — skipped")
        continue
    obs = ad.read_h5ad(CACHE / "r3_query" / f"{key}_mapped.h5ad", backed="r").obs[["sample", "condition", "label", "confident", "scanvi_prob", "knn_vote"]].copy()
    mk = pd.read_csv(RESULTS / f"r3_ingest/{key}_marker_check.csv").set_index("label")
    qc = pd.read_csv(RESULTS / f"r3_ingest/{key}_cell_qc.csv")
    conf = obs[obs["confident"].astype(bool)]
    n_by = conf["label"].value_counts()
    recovered = sorted(n_by[n_by >= 20].index)
    mk_pass = (mk["marker_z_in_label"] >= 0.5) & (mk["marker_z_in_label"] > mk["marker_z_in_others"])
    summ.append({"dataset": label, "key": key, "n_samples": obs["sample"].nunique(), "n_cells_qc": len(obs),
                 "frac_confident": float(obs["confident"].astype(bool).mean()), "n_our_types": len(our_types),
                 "n_types_recovered_20_confident": len(recovered),
                 "types_not_recovered": "; ".join(t for t in our_types if t not in recovered),
                 "marker_check_pass_rate": float(mk_pass.mean()), "n_labels_marker_checked": len(mk),
                 "n_hair_cells_confident": int(n_by.get("Hair Cell (Outer/Inner)", 0)),
                 "n_barcodes_in": int(qc.loc[qc["step"].isin(["QC thresholds", "emptyDrops"]), "n_barcodes_in"].sum())})
    for t in our_types:
        cov.append({"dataset": label, "key": key, "cell_type": t, "n_cells": int((obs["label"] == t).sum()), "n_confident": int(n_by.get(t, 0)),
                    "share_confident": float(n_by.get(t, 0) / max(len(conf), 1)),
                    "frac_confident_of_label": float(obs.loc[obs["label"] == t, "confident"].astype(bool).mean()) if (obs["label"] == t).any() else np.nan,
                    "marker_z_in_label": mk["marker_z_in_label"].get(t, np.nan), "marker_check_pass": bool(mk_pass.get(t, False))})
    # composition similarity of structural types (whole-cochlea datasets)
    if key in ("gse283534", "gse274279"):
        struct = [t for t in our_types if not t.startswith(IMMUNE_PREFIX)]
        o = ours[ours["cell_type"].isin(struct)].groupby("cell_type")["n_cells"].sum()
        o = o / o.sum()
        for s, g in conf.groupby("sample", observed=True):
            e = g["label"].value_counts().reindex(struct).fillna(0)
            if e.sum() < 200:
                continue
            e = e / e.sum()
            rho, p = stats.spearmanr(o.reindex(struct).fillna(0), e)
            comp.append({"dataset": label, "sample": s, "condition": g["condition"].iloc[0], "n_structural_cells": int(g["label"].isin(struct).sum()),
                         "spearman_rho_structural_shares": rho, "p": p})
sm = pd.DataFrame(summ)
write_csv(sm, MOD, "atlas_concordance_summary.csv",
          "External adult cochlear atlases mapped onto our gold-standard labels: confident share, types recovered, marker-check pass rate",
          required=["dataset", "n_cells_qc", "frac_confident", "n_types_recovered_20_confident", "marker_check_pass_rate"])
write_csv(pd.DataFrame(cov), MOD, "atlas_coverage.csv", "Our cochlear cell types × external datasets: confident cells, share, marker check",
          required=["dataset", "cell_type", "n_confident", "share_confident", "marker_check_pass"])
if comp:
    write_csv(pd.DataFrame(comp), MOD, "composition_similarity.csv",
              "Structural-cell-type composition of external whole-cochlea samples vs our cochlea (Spearman of shares, confident cells)",
              required=["dataset", "sample", "spearman_rho_structural_shares"])
log("\n" + sm.drop(columns=["types_not_recovered"]).round(3).to_string())

# ---------------- author labels (GSE234926)
meta_p = CACHE / "r4_concordance/GSE234926_author_meta.csv"
if meta_p.exists() and (CACHE / "r3_query/gse234926_mapped.h5ad").exists():
    am = pd.read_csv(meta_p)
    sex = am["orig.ident"].str.startswith("male").map({True: "M", False: "F"})
    am["key"] = "P90_Fbxo2_" + sex + ":" + am["cell"].str.replace(r"_\d+$", "", regex=True)
    obs = ad.read_h5ad(CACHE / "r3_query/gse234926_mapped.h5ad", backed="r").obs[["label", "confident", "scanvi_prob"]].copy()
    j = am.set_index("key").join(obs, how="inner")
    log(f"GSE234926: author cells {len(am)}, matched to mapped cells {len(j)}")
    ct = j.groupby(["celltype", "label"], observed=True).size().rename("n_cells").reset_index()
    ct["share_of_author_type"] = ct["n_cells"] / ct.groupby("celltype")["n_cells"].transform("sum")
    write_csv(ct, MOD, "author_label_concordance.csv", "GSE234926: author cell types × transferred gold-standard labels (cells, share of author type)",
              required=["celltype", "label", "n_cells", "share_of_author_type"])
    mk34 = pd.read_csv(RESULTS / "r3_ingest/gse234926_marker_check.csv").set_index("label")
    mpass = ((mk34["marker_z_in_label"] >= 0.5) & (mk34["marker_z_in_label"] > mk34["marker_z_in_others"])).to_dict()
    ag = []
    for a, g in j.groupby("celltype"):
        exp, expb = EXPECT.get(a, set()), EXPECT_BROAD.get(a, set())
        dom = g["label"].value_counts()
        ag.append({"author_celltype": a, "n_cells": len(g), "dominant_label": dom.index[0], "dominant_share": float(dom.iloc[0] / len(g)),
                   "dominant_label_passes_marker_check": mpass.get(dom.index[0], np.nan),
                   "expected_labels": "; ".join(sorted(exp)), "frac_expected": float(g["label"].isin(exp).mean()) if exp else np.nan,
                   "expected_labels_broad": "; ".join(sorted(expb)), "frac_expected_broad": float(g["label"].isin(expb).mean()) if expb else np.nan,
                   "frac_expected_confident": float(g.loc[g["confident"].astype(bool), "label"].isin(exp).mean())
                   if exp and g["confident"].astype(bool).any() else np.nan})
    ag = pd.DataFrame(ag)
    tot = {"author_celltype": "ALL (cell-weighted)", "n_cells": int(ag["n_cells"].sum()),
           "frac_expected": float(np.average(ag["frac_expected"].fillna(0), weights=ag["n_cells"])),
           "frac_expected_broad": float(np.average(ag["frac_expected_broad"].fillna(0), weights=ag["n_cells"])),
           "dominant_share": adjusted_rand_score(j["celltype"], j["label"]), "dominant_label": "ARI in dominant_share column"}
    ag = pd.concat([ag, pd.DataFrame([tot])], ignore_index=True)
    write_csv(ag, MOD, "author_label_agreement.csv",
              "GSE234926: agreement of transferred labels with the prior-knowledge correspondence to author labels (strict and broad: duct epithelia "
              "also expected in our Fbxo2+/Epcam+ 'Interdental Cell' label); dominant label's marker check; ARI (last row)",
              required=["author_celltype", "n_cells", "dominant_label", "frac_expected"])
    log("\n" + ag.round(3).to_string())
