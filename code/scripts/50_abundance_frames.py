"""Module 4 — abundance and distribution rhythms in several compositional reference frames.

Frames (log-ratios with pseudocount 0.5; natural-log values converted to log2 for amplitudes):
  A  all cells of the sample: centred log-ratio (CLR) over all cell types of the tissue
  B  cochlea: additive log-ratio vs the structural reference (hair cells, SV, interdental, epithelial,
     fibrocytes, SGN, Schwann) — cells whose number per cochlea should not oscillate; reference stability tested
  C  cochlea: additive log-ratio vs marrow content (HSC/MPP, GMP, Pro-B, Pre-B, erythroblast)
  D  within-lineage stage fractions (log-ratio of a stage vs the lineage total), both tissues
  E  cross-tissue distribution index: log(p_cochlea / p_BM) with p = share within the haematopoietic compartment
     of each tissue (constant platform capture bias cancels in the time trend)
Each value series (6 ZTs) gets a cosinor fit, 60-class permutation p, LOTO stability and BH within frame×tissue.

Outputs (results/50_abundance/): composition_frames.csv, abundance_rhythm_frames.csv, reference_frame_stability.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "50_abundance"
PC = 0.5
STRUCT = ["Hair Cell (Outer/Inner)", "SV Marginal Cell", "SV Intermediate Cell (Pigmented)",
          "SV Intermediate Cell (Functional)", "Interdental Cell", "Epithelial Cell (Non-sensory)",
          "Fibrocyte (Cochlear)", "Fibrocyte (Spiral Ligament)", "Spiral Ganglion Neuron", "Schwann Cell"]
MARROW = ["HSC / MPP", "Granulocyte-Monocyte Progenitor", "Pro-B Cell (Cycling)", "Pre-B Cell", "Erythroblast"]
HEME_CLASSES = ["Myeloid", "Lymphoid", "Progenitor", "Erythroid"]
LINEAGES = {
    "neutrophil": ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)"],
    "B": ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"],
    "monocyte": ["Monocyte (Classical)", "Monocyte (Non-classical)"],
    "mono_mac": ["Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage"],
    "myeloid_progenitor": ["Granulocyte-Monocyte Progenitor", "Neutrophil (Band)", "Monocyte (Classical)"],
}

ct = pd.read_csv(RESULTS / "00_audit/celltype_counts_by_sample.csv")
wide = ct.pivot_table(index=["tissue", "sample_id", "zt"], columns="cell_type", values="n_cells",
                      aggfunc="sum", fill_value=0)
cls = ct.drop_duplicates("cell_type").set_index("cell_type")["cell_class"]
rows = []


def add(frame, tissue, feature, s):
    for (t, sid, zt), v in s.items():
        rows.append({"frame": frame, "tissue": tissue, "feature": feature, "sample_id": sid, "zt": zt,
                     "value_ln": float(v)})


for tissue in ["cochlea", "bm"]:
    W = wide.loc[tissue].copy()
    W.index = pd.MultiIndex.from_tuples([(tissue, *i) for i in W.index], names=["tissue", "sample_id", "zt"])
    present = W.columns[W.sum(0) > 0]
    W = W[present]
    L = np.log(W + PC)
    clr = L.sub(L.mean(1), axis=0)
    for c in present:
        add("A_clr_all", tissue, c, clr[c])
    if tissue == "cochlea":
        ref = np.log(W[[c for c in STRUCT if c in W]].sum(1) + PC)
        mar = np.log(W[[c for c in MARROW if c in W]].sum(1) + PC)
        for c in present:
            add("B_alr_structural", tissue, c, L[c] - ref)
            if c not in MARROW:
                add("C_alr_marrow", tissue, c, L[c] - mar)
        add("C_alr_marrow", tissue, "__marrow_content_vs_structural", mar - ref)
        # reference stability: share of each structural type within the structural reference
        S = W[[c for c in STRUCT if c in W]]
        Ls = np.log(S + PC)
        for c in S.columns:
            add("B_reference_internal_clr", tissue, c, Ls[c] - Ls.mean(1))
    for lin, members in LINEAGES.items():
        m = [c for c in members if c in W]
        if len(m) < 2:
            continue
        tot = np.log(W[m].sum(1) + PC)
        for c in m:
            add(f"D_within_{lin}", tissue, c, L[c] - tot)

# frame E — cochlea vs BM share within the haematopoietic compartment
heme = [c for c in wide.columns if cls.get(c) in HEME_CLASSES]
shares = {}
for tissue in ["cochlea", "bm"]:
    W = wide.loc[tissue][heme]
    shares[tissue] = (W + PC).div(W.sum(1) + PC * len(heme), axis=0)
for c in heme:
    if wide.loc["cochlea"][c].sum() < 60 or wide.loc["bm"][c].sum() < 60:
        continue
    pc = shares["cochlea"][c].reset_index().sort_values("zt")
    pb = shares["bm"][c].reset_index().sort_values("zt")
    for (_, r1), (_, r2) in zip(pc.iterrows(), pb.iterrows()):
        assert r1["zt"] == r2["zt"]
        rows.append({"frame": "E_distribution_cochlea_vs_bm", "tissue": "both", "feature": c,
                     "sample_id": f"{r1['sample_id']}|{r2['sample_id']}", "zt": r1["zt"],
                     "value_ln": float(np.log(r1[c] / r2[c]))})

frames = pd.DataFrame(rows)
frames["value_log2"] = frames["value_ln"] / np.log(2)
write_csv(frames.sort_values(["frame", "tissue", "feature", "zt"]), MOD, "composition_frames.csv",
          "Cell-type abundance in compositional reference frames per sample (natural-log and log2 ratios)",
          required=["frame", "tissue", "feature", "sample_id", "zt", "value_log2"])

res = []
for (frame, tissue, feat), g in frames.groupby(["frame", "tissue", "feature"]):
    g = g.sort_values("zt")
    y = g["value_log2"].to_numpy()[:, None]
    zt = g["zt"].to_numpy()
    if np.allclose(y, y[0]):
        continue
    f = cosinor.fit(y, zt)
    _, _, p_perm, _ = cosinor.permutation_test(y, zt)
    lo = cosinor.loto(y, zt)
    res.append({"frame": frame, "tissue": tissue, "feature": feat, "mesor_log2": f["mesor"][0],
                "amp_log2": f["amp"][0], "ptt_fc": 2 ** (2 * f["amp"][0]), "peak_zt": f["peak"][0],
                "r2": f["r2"][0], "F": f["F"][0], "p": f["p"][0], "p_perm": p_perm[0],
                "loto_phase_sd_h": lo["loto_phase_sd_h"][0], "loto_min_amp_ratio": lo["loto_min_amp_ratio"][0],
                "loto_max_p": lo["loto_max_p"][0]})
res = pd.DataFrame(res)
res["q_BH"] = res.groupby(["frame", "tissue"])["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
res["robust"] = (res["p"] < 0.05) & (res["p_perm"] <= 0.05) & (res["loto_phase_sd_h"] < 3) & \
                (res["loto_min_amp_ratio"] > 0.5)
write_csv(res.sort_values(["frame", "tissue", "p"]), MOD, "abundance_rhythm_frames.csv",
          "Cosinor rhythm tests of cell-type abundance in each reference frame (log2 units)",
          required=["frame", "tissue", "feature", "amp_log2", "peak_zt", "p", "p_perm", "q_BH", "robust"])
stab = res[res["frame"] == "B_reference_internal_clr"]
write_csv(stab, MOD, "reference_frame_stability.csv",
          "Rhythmicity of each structural reference cell type within the structural reference (should be null)",
          required=["feature", "p", "p_perm"])
pd.set_option("display.width", 250)
print(res[res["robust"]].sort_values(["frame", "tissue", "p"])[
    ["frame", "tissue", "feature", "ptt_fc", "peak_zt", "p", "p_perm", "q_BH", "loto_phase_sd_h"]].round(3).to_string(index=False))
log("done")
