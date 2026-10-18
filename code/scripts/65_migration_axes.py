"""Module 5.2c — curated homing / recruitment / retention / niche axes across the cochlea–BM system.

Potential of an axis at ZT t = log2(CPM_ligand,sender(t) + 1) + log2(CPM_receptor,receiver(t) + 1) (log of the
ligand × receptor product), computed from pseudobulk of gold-standard cell types:
  within_cochlea / within_bm : sender and receiver in the same tissue
  recruitment_cochlea_to_bm  : sender = cochlear tissue/niche cell, receiver = BM immune cell (same ZT)
  niche_retention            : sender = cochlear osteo/endothelial/fibrocyte niche cells, receiver = BM HSPC/immune
Enzyme–receptor axes use the product of the synthesising enzymes (e.g. Ch25h·Cyp7b1 → Gpr183).
Each potential: cosinor, 60-class permutation p, LOTO, BH within class. Phase relations: Δφ between the cochlear
abundance peak (frame C, excess over marrow) of the receiver type and the recruitment-potential peak.

Outputs (results/65_migration_axes/): migration_axes_rhythm.csv, axes_vs_abundance_phase.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor  # noqa: E402
from lib.io import CACHE, RESULTS, log, write_csv  # noqa: E402

MOD = "65_migration_axes"
AXES = [  # (axis, ligand genes (product), receptor genes (product), category)
    ("Cxcl12-Cxcr4", ["Cxcl12"], ["Cxcr4"], "retention_homing"), ("Cxcl12-Ackr3", ["Cxcl12"], ["Ackr3"], "retention_homing"),
    ("Cxcl1-Cxcr2", ["Cxcl1"], ["Cxcr2"], "egress_recruitment"), ("Cxcl2-Cxcr2", ["Cxcl2"], ["Cxcr2"], "egress_recruitment"),
    ("Cxcl5-Cxcr2", ["Cxcl5"], ["Cxcr2"], "egress_recruitment"), ("Ccl2-Ccr2", ["Ccl2"], ["Ccr2"], "egress_recruitment"),
    ("Ccl7-Ccr2", ["Ccl7"], ["Ccr2"], "egress_recruitment"), ("Ccl12-Ccr2", ["Ccl12"], ["Ccr2"], "egress_recruitment"),
    ("Cx3cl1-Cx3cr1", ["Cx3cl1"], ["Cx3cr1"], "tissue_retention"), ("Ccl19-Ccr7", ["Ccl19"], ["Ccr7"], "lymphoid_homing"),
    ("Ccl21a-Ccr7", ["Ccl21a"], ["Ccr7"], "lymphoid_homing"), ("Cxcl13-Cxcr5", ["Cxcl13"], ["Cxcr5"], "lymphoid_homing"),
    ("Ccl25-Ccr9", ["Ccl25"], ["Ccr9"], "lymphoid_homing"), ("Cxcl9-Cxcr3", ["Cxcl9"], ["Cxcr3"], "inflammatory"),
    ("Cxcl10-Cxcr3", ["Cxcl10"], ["Cxcr3"], "inflammatory"), ("Ccl3-Ccr1", ["Ccl3"], ["Ccr1"], "inflammatory"),
    ("Ccl3-Ccr5", ["Ccl3"], ["Ccr5"], "inflammatory"), ("Ccl4-Ccr5", ["Ccl4"], ["Ccr5"], "inflammatory"),
    ("Ccl5-Ccr5", ["Ccl5"], ["Ccr5"], "inflammatory"), ("Ccl6-Ccr1", ["Ccl6"], ["Ccr1"], "inflammatory"),
    ("Ccl9-Ccr1", ["Ccl9"], ["Ccr1"], "inflammatory"), ("Cxcl16-Cxcr6", ["Cxcl16"], ["Cxcr6"], "tissue_retention"),
    ("Sele-Selplg", ["Sele"], ["Selplg"], "adhesion"), ("Selp-Selplg", ["Selp"], ["Selplg"], "adhesion"),
    ("Sele-Cd44", ["Sele"], ["Cd44"], "adhesion"), ("Icam1-Itgal/Itgb2", ["Icam1"], ["Itgal", "Itgb2"], "adhesion"),
    ("Icam1-Itgam/Itgb2", ["Icam1"], ["Itgam", "Itgb2"], "adhesion"), ("Vcam1-Itga4/Itgb1", ["Vcam1"], ["Itga4", "Itgb1"], "adhesion"),
    ("Madcam1-Itga4/Itgb7", ["Madcam1"], ["Itga4", "Itgb7"], "adhesion"), ("Kitl-Kit", ["Kitl"], ["Kit"], "niche"),
    ("Angpt1-Tek", ["Angpt1"], ["Tek"], "niche"), ("Il7-Il7r", ["Il7"], ["Il7r"], "niche"), ("Csf1-Csf1r", ["Csf1"], ["Csf1r"], "niche"),
    ("Jag1-Notch1", ["Jag1"], ["Notch1"], "niche"), ("Jag1-Notch2", ["Jag1"], ["Notch2"], "niche"),
    ("Spp1-Cd44", ["Spp1"], ["Cd44"], "niche"), ("C3-C3ar1", ["C3"], ["C3ar1"], "complement"),
    ("Ch25h*Cyp7b1-Gpr183", ["Ch25h", "Cyp7b1"], ["Gpr183"], "oxysterol"),
    ("Sphk1*Spns2-S1pr1", ["Sphk1", "Spns2"], ["S1pr1"], "s1p_egress"),
    ("Alox5*Lta4h-Ltb4r1", ["Alox5", "Lta4h"], ["Ltb4r1"], "ltb4")]
NICHE_SENDERS = ["Osteoprogenitor", "Osteoblast", "Endothelial Cell", "Fibrocyte (Cochlear)",
                 "Fibrocyte (Spiral Ligament)", "Cycling Cell (Stromal)"]
TISSUE_SENDERS = NICHE_SENDERS + ["SV Marginal Cell", "SV Intermediate Cell (Pigmented)",
                                  "SV Intermediate Cell (Functional)", "Interdental Cell", "Epithelial Cell (Non-sensory)",
                                  "Schwann Cell", "Spiral Ganglion Neuron", "Hair Cell (Outer/Inner)", "Macrophage",
                                  "Mast Cell"]
IMMUNE = ["Neutrophil (Band)", "Neutrophil (Mature)", "Neutrophil (Aged)", "Neutrophil (Activated)",
          "Monocyte (Classical)", "Monocyte (Non-classical)", "Macrophage", "B Cell (Transitional)", "B Cell (Naïve)",
          "B Cell (Mature)", "T Cell (CD3+)", "NK Cell", "pDC", "Mast Cell", "HSC / MPP", "Granulocyte-Monocyte Progenitor",
          "Pre-B Cell", "Pro-B Cell (Cycling)"]

design = pd.read_csv(RESULTS / "30_pseudobulk/pseudobulk_design.csv")
incl = pd.read_csv(RESULTS / "30_pseudobulk/groups_included.csv")
incl = incl[(incl["level"] == "cell_type") & (incl["status"] != "excluded")]
expr = {}
for tissue in ["cochlea", "bm"]:
    pb = pd.read_parquet(CACHE / "30_pseudobulk" / f"{tissue}__cell_type.parquet")
    for g in incl[incl["tissue"] == tissue]["group"]:
        d = design[(design["tissue"] == tissue) & (design["level"] == "cell_type") & (design["group"] == g)].sort_values("zt")
        cnt = pb[[f"{g}|{s}" for s in d["sample_id"]]]
        expr[(tissue, g)] = np.log2(cnt / cnt.sum(0) * 1e6 + 1)
        expr[(tissue, g)].columns = d["zt"].to_numpy()
ZT = np.array(sorted(next(iter(expr.values())).columns))


def level(tissue, group, genes):
    m = expr.get((tissue, group))
    if m is None or not all(g in m.index for g in genes):
        return None
    v = m.loc[genes, ZT].to_numpy()
    if (2 ** v - 1).mean(1).min() < 1:        # require >= 1 CPM on average for every factor
        return None
    return v.sum(0)                              # log2 of the product


combos = []
for axis, lig, rec, cat in AXES:
    for tissue in ["cochlea", "bm"]:
        for s in incl[incl["tissue"] == tissue]["group"]:
            for r in IMMUNE:
                combos.append(("within_" + tissue, axis, cat, tissue, s, tissue, r, lig, rec))
    for s in TISSUE_SENDERS:
        for r in IMMUNE:
            combos.append(("recruitment_cochlea_to_bm", axis, cat, "cochlea", s, "bm", r, lig, rec))
    if cat in ("niche", "retention_homing"):
        for s in NICHE_SENDERS:
            for r in IMMUNE:
                combos.append(("niche_retention", axis, cat, "cochlea", s, "bm", r, lig, rec))
rows, Y = [], []
for cls, axis, cat, st, s, rt, r, lig, rec in combos:
    L = level(st, s, lig)
    R = level(rt, r, rec)
    if L is None or R is None:
        continue
    rows.append({"class": cls, "axis": axis, "category": cat, "sender_tissue": st, "sender": s,
                 "receiver_tissue": rt, "receiver": r})
    Y.append(L + R)
Y = np.array(Y).T
log(f"{Y.shape[1]} axis × sender × receiver potentials")
f = cosinor.fit(Y, ZT)
_, _, pp, _ = cosinor.permutation_test(Y, ZT)
lo = cosinor.loto(Y, ZT)
out = pd.DataFrame(rows)
for k, v in {"mesor_log2": f["mesor"], "amp_log2": f["amp"], "ptt_fc": 2 ** (2 * f["amp"]), "peak_zt": f["peak"],
             "p": f["p"], "p_perm": pp, "loto_phase_sd_h": lo["loto_phase_sd_h"]}.items():
    out[k] = v
for i, zt in enumerate(ZT):
    out[f"potential_zt{int(zt):02d}"] = Y[i]
out["q_BH"] = out.groupby("class")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
out["robust"] = (out["q_BH"] < 0.1) & (out["p_perm"] <= 0.05) & (out["loto_phase_sd_h"] < 3)
write_csv(out.sort_values(["class", "p"]), MOD, "migration_axes_rhythm.csv",
          "Ligand×receptor potentials of curated migration/niche axes per sender/receiver and their 24-h rhythm",
          required=["class", "axis", "sender", "receiver", "peak_zt", "p", "p_perm", "q_BH"])

ab = pd.read_csv(RESULTS / "50_abundance/abundance_rhythm_frames.csv")
abc = ab[(ab["frame"] == "C_alr_marrow") & (ab["tissue"] == "cochlea")].set_index("feature")
abb = ab[(ab["frame"] == "A_clr_all") & (ab["tissue"] == "bm")].set_index("feature")
rec = out[out["class"] == "recruitment_cochlea_to_bm"].copy()
rec = rec[rec["receiver"].isin(abc.index)]
rec["cochlea_abund_peak_zt"] = rec["receiver"].map(abc["peak_zt"])
rec["cochlea_abund_p"] = rec["receiver"].map(abc["p"])
rec["bm_abund_peak_zt"] = rec["receiver"].map(abb["peak_zt"])
rec["lag_abundance_after_potential_h"] = (rec["cochlea_abund_peak_zt"] - rec["peak_zt"]) % 24
write_csv(rec, MOD, "axes_vs_abundance_phase.csv",
          "Recruitment-potential peaks vs cochlear (excess-over-marrow) abundance peaks of the receiver type",
          required=["axis", "sender", "receiver", "peak_zt", "cochlea_abund_peak_zt", "lag_abundance_after_potential_h"])
print(out[out["robust"]].groupby(["class", "category"]).size().to_string())
log("done")
