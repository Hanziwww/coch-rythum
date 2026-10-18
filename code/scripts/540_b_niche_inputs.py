"""Round 5, WP-B4 — niche and systemic inputs to B cells (H5), BM and cochlea.

Curated axes (LIANA mouse consensus complexes; 63 per-sample scores, 64 rhythm fits on log2 lr_means):
  IL-7    Il7 → Il7r / Il2rg          CXCL12  Cxcl12 → Cxcr4 (Ackr3)       BAFF / APRIL  Tnfsf13b / Tnfsf13 → Tnfrsf13c / 13b / 17
  (exploratory) CXCL13 → Cxcr5; VCAM1 → Itga4 / Itgb1; Cd40lg → Cd40
Senders: niche (stromal, bone, endothelial, glia) and myeloid cell types; receivers: B stages.
H5 primary (pre-specified): per axis the representative sender → B interaction = highest mean lr_means among candidates
present in ≥ 5 samples (selection by magnitude, not by rhythm); Fisher combination of the three axes' exact permutation
p (64); lead criterion = sender-ligand peak (gene level, 64) precedes the receiving stage's share peak (510,
share_of_tissue) by 0–8 h. Agreeing methods: ligand gene-level rhythm in the sender (64 ligand_rhythmic); NicheNet
ligand-activity percentile ≥ 0.9 in the receiving stage (541); MultiNicheNet prioritisation of the axis ligand towards B
receivers (67; rank among that tissue's B-receiver pairs).
Outputs (results/r5_bcell/): b_niche_axes.csv, b_primary_h5.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import phase  # noqa: E402
from lib.io import RESULTS, log, write_csv  # noqa: E402

MOD = "r5_bcell"
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
NICHE = {"Osteoprogenitor", "Osteoblast", "Fibrocyte (Cochlear)", "Fibrocyte (Spiral Ligament)", "Cycling Cell (Stromal)",
         "Endothelial Cell", "Schwann Cell", "Oligodendrocyte"}
MYELOID = {"Macrophage", "Monocyte (Classical)", "Monocyte (Non-classical)", "Neutrophil (Band)", "Neutrophil (Mature)",
           "Neutrophil (Aged)", "Neutrophil (Activated)", "Granulocyte-Monocyte Progenitor", "pDC", "Mast Cell"}
AXES = {"IL-7": (["Il7"], ["Il7r"]), "CXCL12": (["Cxcl12"], ["Cxcr4", "Ackr3"]),
        "BAFF/APRIL": (["Tnfsf13b", "Tnfsf13"], ["Tnfrsf13c", "Tnfrsf13b", "Tnfrsf17"]),
        "CXCL13 (exploratory)": (["Cxcl13"], ["Cxcr5"]), "VCAM1 (exploratory)": (["Vcam1"], ["Itga4", "Itgb1"]),
        "CD40L (exploratory)": (["Cd40lg"], ["Cd40"])}
PRIMARY = ["IL-7", "CXCL12", "BAFF/APRIL"]

comp = pd.read_csv(RESULTS / "r5_bcell/b_composition_summary.csv")
comp = comp[comp["frame"] == "share_of_tissue"].set_index(["tissue", "feature"])["peak_zt"]
nn = pd.read_csv(RESULTS / "r5_bcell/b_nichenet_ligand_activity.csv") if (RESULTS / "r5_bcell/b_nichenet_ligand_activity.csv").exists() else pd.DataFrame()
mnn = pd.read_csv(RESULTS / "67_multinichenet/multinichenet_prioritization_bm.csv")
mnn = pd.concat([mnn, pd.read_csv(RESULTS / "67_multinichenet/multinichenet_prioritization_cochlea.csv")], ignore_index=True)
rows, prim = [], []
for t in ["bm", "cochlea"]:
    rr = pd.read_csv(RESULTS / f"64_ccc_rhythm/ccc_rhythm_{t}.csv")
    rr = rr[rr["target"].isin(BST) & (rr["source"].isin(NICHE) | rr["source"].isin(MYELOID))]
    mb = mnn[(mnn["tissue"] == t) & mnn["receiver"].isin(BST)].sort_values("prioritization_score", ascending=False).reset_index(drop=True)
    reps = {}
    for ax, (ligs, recs) in AXES.items():
        c = rr[rr["ligand_complex"].isin(ligs) & rr["receptor_complex"].apply(lambda s: any(x in str(s).split("_") for x in recs))].copy()
        if not len(c):
            continue
        c["sender_class"] = np.where(c["source"].isin(NICHE), "niche", "myeloid")
        c["receiver_share_peak_zt"] = [comp.get((t, r_), np.nan) for r_ in c["target"]]
        c["lead_h"] = (c["receiver_share_peak_zt"] - c["ligand_peak_zt"]) % 24
        c["ligand_leads_0_8h"] = c["lead_h"].between(0, 8)
        c["representative"] = False
        rep = c.sort_values("mean_lr_means", ascending=False).index[0]
        c.loc[rep, "representative"] = True
        c["axis"] = ax
        c["tissue"] = t
        for r_ in c.itertuples():
            nsub = nn[(nn.get("tissue") == t) & (nn.get("receiver") == r_.target) & nn["ligand"].isin(ligs)] if len(nn) else pd.DataFrame()
            rows.append({**{k: getattr(r_, k) for k in ["tissue", "axis", "source", "sender_class", "target", "ligand_complex", "receptor_complex",
                                                         "mean_lr_means", "peak_zt", "p", "p_perm", "q_BH", "loto_phase_sd_h", "ligand_peak_zt",
                                                         "ligand_rhythmic", "receptor_peak_zt", "receptor_rhythmic", "mechanism",
                                                         "receiver_share_peak_zt", "lead_h", "ligand_leads_0_8h", "representative"]},
                         "nichenet_best_percentile": float(nsub["percentile"].max()) if len(nsub) else np.nan,
                         "nichenet_best_set": nsub.sort_values("percentile").iloc[-1]["gene_set"] if len(nsub) else ""})
        reps[ax] = c.loc[rep]
    # H5 primary
    pv = [reps[a]["p_perm"] for a in PRIMARY if a in reps]
    chi = -2 * np.sum(np.log(pv)) if pv else np.nan
    pf = float(stats.chi2.sf(chi, 2 * len(pv))) if pv else np.nan
    lead_ok = [bool(reps[a]["ligand_leads_0_8h"]) for a in PRIMARY if a in reps]
    agree_gene = [bool(reps[a]["ligand_rhythmic"]) for a in PRIMARY if a in reps]
    nn_ok = []
    for a in PRIMARY:
        if a in reps and len(nn):
            s = nn[(nn["tissue"] == t) & (nn["receiver"] == reps[a]["target"]) & nn["ligand"].isin(AXES[a][0])]
            nn_ok.append(bool(len(s) and s["percentile"].max() >= 0.9))
    mnn_hits = [a for a in PRIMARY if mb["ligand"].isin(AXES[a][0]).any()]
    prim.append({"hypothesis": "H5 niche signals lead", "tissue": t,
                 "primary_statistic": "Fisher combination of the representative IL-7 / CXCL12 / BAFF-APRIL interactions' exact permutation p (64)",
                 "primary_p": pf, "n_axes": len(pv), "axes_p_perm": "; ".join(f"{a} {reps[a]['source']}→{reps[a]['target']} {reps[a]['p_perm']:.3f}"
                                                                              for a in PRIMARY if a in reps),
                 "n_axes_ligand_leads_0_8h": int(sum(lead_ok)), "n_axes_ligand_gene_rhythmic": int(sum(agree_gene)),
                 "n_axes_nichenet_top10pct": int(sum(nn_ok)), "multinichenet_axes_present": "; ".join(mnn_hits),
                 "in_window": bool(sum(lead_ok) >= 2)})
    log(f"{t}: H5 Fisher p {pf:.3f}; leads {sum(lead_ok)}/{len(lead_ok)}")
ax_df = pd.DataFrame(rows)
ax_df["peak_clock"] = phase.zt_to_clock(ax_df["peak_zt"])
write_csv(ax_df, MOD, "b_niche_axes.csv", "Niche → B-stage interactions of curated axes (IL-7, CXCL12, BAFF/APRIL; exploratory CXCL13, VCAM1, CD40L): rhythm, lead vs B-stage share, NicheNet",
          required=["tissue", "axis", "source", "target", "peak_zt", "peak_clock", "p_perm", "representative"])
write_csv(pd.DataFrame(prim), MOD, "b_primary_h5.csv", "Primary hypothesis H5 (niche signals lead): primary statistic and agreeing methods per tissue",
          required=["hypothesis", "tissue", "primary_p"])
log("\n" + ax_df[ax_df["representative"]][["tissue", "axis", "source", "target", "mean_lr_means", "peak_zt", "p_perm", "ligand_peak_zt", "lead_h",
                                            "ligand_rhythmic", "nichenet_best_percentile"]].round(3).to_string())
