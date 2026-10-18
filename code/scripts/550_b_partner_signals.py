"""Round 5, WP-P — B ↔ partner signals and spatial adjacency in the cochlea (H7, H8). Partner rhythms are not tested (D29).

Partners: Hair Cell (Outer/Inner), Spiral Ganglion Neuron, Schwann Cell, Oligodendrocyte, SV Marginal Cell,
SV Intermediate Cell (Functional / Pigmented).
(1) H7 primary (pre-specified): all B-stage ↔ partner LR interactions (either direction) of the cochlear LIANA per-sample
    scores (63) present in all 6 samples; cosinor of log2 lr_means; BH within this family; statistic = number with
    q < 0.1; null = the same count under the 59 non-identity dihedral relabelings applied jointly → exact p.
    Curated pair families (cytokine: Tnf / Lta / Ltb → Ltbr / Tnfrsf1a / Tnfrsf1b, Il10 → Il10ra, Il6 → Il6ra / Il6st,
    Tnfsf13b; neuropeptide: Calca → Calcrl / Ramp1, Npy → Npy1r, Vip → Vipr1 / Vipr2, Gal → Galr1–3; neurotrophin:
    Bdnf / Ntf3 / Ngf → Ntrk1–3 / Ngfr; glial Cxcl12 → Cxcr4) are flagged. Agreeing methods: B-side gene rhythm (64
    ligand / receptor columns); NicheNet percentile of the B-derived ligand in the partner receiver (541; receiving side).
(2) H8 primary (pre-specified): GeoMx GSE312646 (320 deconvolution): B-lineage proportion (sum of the five B labels)
    per AOI, spiral-ganglion-neuron vs supporting-cell AOIs, control slides (and all slides): one-sided Wilcoxon
    (neuron > supporting). Agreeing method: an external ImmGen B-lineage signature (560; union of BM pro-B → FrF stage
    signatures; genes in the GeoMx panel) scored per AOI (mean z of log2 counts), same test; prior knowledge from
    scripts/resources/spatial_priors.csv.
Outputs (results/r5_partners/): b_partner_interactions.csv, b_partner_global_test.csv, b_partner_geomx.csv, b_primary_h7_h8.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import CACHE, EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r5_partners"
BST = ["Pro-B Cell (Cycling)", "Pre-B Cell", "B Cell (Transitional)", "B Cell (Naïve)", "B Cell (Mature)"]
PART = ["Hair Cell (Outer/Inner)", "Spiral Ganglion Neuron", "Schwann Cell", "Oligodendrocyte", "SV Marginal Cell",
        "SV Intermediate Cell (Functional)", "SV Intermediate Cell (Pigmented)"]
FAM = {"cytokine": ({"Tnf", "Lta", "Ltb", "Il10", "Il6", "Tnfsf13b"}, {"Ltbr", "Tnfrsf1a", "Tnfrsf1b", "Il10ra", "Il6ra", "Il6st", "Tnfrsf13c"}),
       "neuropeptide": ({"Calca", "Npy", "Vip", "Gal"}, {"Calcrl", "Ramp1", "Npy1r", "Vipr1", "Vipr2", "Galr1", "Galr2", "Galr3"}),
       "neurotrophin": ({"Bdnf", "Ntf3", "Ngf"}, {"Ntrk1", "Ntrk2", "Ntrk3", "Ngfr"}),
       "chemokine (glial CXCL12)": ({"Cxcl12"}, {"Cxcr4", "Ackr3"})}
CLASSES = cosinor.dihedral_classes(6)

# ---------------- (1) H7
li = pd.read_parquet(CACHE / "63_ccc/liana_cochlea.parquet")
li = li[(li["source"].isin(BST) & li["target"].isin(PART)) | (li["source"].isin(PART) & li["target"].isin(BST))]
key = ["source", "target", "ligand_complex", "receptor_complex"]
W = li.pivot_table(index=key, columns="zt", values="lr_means")
W = W.dropna()
W = W[(W > 0).all(1)]
zt = np.array(sorted(W.columns), float)
Y = np.log2(W[zt].to_numpy(float)).T          # 6 × interactions


def count_q(Yv, z):
    f = cosinor.fit(Yv, z)
    q = multipletests(f["p"], method="fdr_bh")[1]
    return q, f


q_obs, f_obs = count_q(Y, zt)
n_obs = int((q_obs < 0.1).sum())
null = [int((count_q(Y, zt[list(p)])[0] < 0.1).sum()) for p in CLASSES[1:]]
p_glob = (1 + sum(x >= n_obs for x in null)) / len(CLASSES)
# per interaction permutation p (F)
Fs = np.vstack([cosinor.fit(Y, zt[list(p)])["F"] for p in CLASSES])
pperm = (Fs[1:] >= Fs[0][None, :]).sum(0) + 1
pperm = pperm / len(CLASSES)
inter = W.reset_index()[key].copy()
inter["direction"] = np.where(inter["source"].isin(BST), "B → partner", "partner → B")
inter["mean_lr_means"] = W.mean(1).to_numpy()
inter["amplitude_log2"] = f_obs["amp"]
inter["peak_zt"] = f_obs["peak"]
inter["peak_clock"] = phase.zt_to_clock(f_obs["peak"])
inter["p"] = f_obs["p"]
inter["p_perm"] = pperm
inter["q_BH_family"] = q_obs
lig = inter["ligand_complex"].str.split("_")
rec = inter["receptor_complex"].str.split("_")
inter["pair_family"] = ""
for fam, (L, R) in FAM.items():
    m = lig.apply(lambda x: bool(set(x) & L)) & rec.apply(lambda x: bool(set(x) & R))
    inter.loc[m, "pair_family"] = fam
r64 = pd.read_csv(RESULTS / "64_ccc_rhythm/ccc_rhythm_cochlea.csv")
inter = inter.merge(r64[key + ["ligand_peak_zt", "ligand_rhythmic", "receptor_peak_zt", "receptor_rhythmic", "mechanism"]], on=key, how="left")
inter["b_side_gene_rhythmic"] = np.where(inter["direction"] == "B → partner", inter["ligand_rhythmic"], inter["receptor_rhythmic"])
nnp = RESULTS / "r5_bcell/b_nichenet_ligand_activity.csv"
if nnp.exists():
    nn = pd.read_csv(nnp)
    nn = nn[(nn["tissue"] == "cochlea") & (nn["receiver_class"] == "partner")]
    best = nn.groupby(["receiver", "ligand"])["percentile"].max()
    inter["nichenet_partner_percentile"] = [best.get((r.target, r.ligand_complex.split("_")[0]), np.nan) if r.direction == "B → partner" else np.nan
                                            for r in inter.itertuples()]
write_csv(inter.sort_values("p"), MOD, "b_partner_interactions.csv",
          "Cochlear B-stage ↔ partner (hair cell, SGN, glia, stria) LR interactions: rhythm, family BH q, curated pair family, B-side gene rhythm, NicheNet",
          required=["source", "target", "ligand_complex", "receptor_complex", "direction", "peak_zt", "peak_clock", "p_perm", "q_BH_family"])
glob = pd.DataFrame([{"family": "all B ↔ partner interactions (6 samples)", "n_interactions": len(inter), "n_q_lt_0.1": n_obs,
                      "null_mean": float(np.mean(null)), "null_q95": float(np.quantile(null, 0.95)), "p_exact": p_glob}] +
                    [{"family": f"{d} (descriptive)", "n_interactions": int((inter["direction"] == d).sum()),
                      "n_q_lt_0.1": int(((inter["direction"] == d) & (inter["q_BH_family"] < 0.1)).sum())} for d in ["B → partner", "partner → B"]] +
                    [{"family": f"curated: {fam} (descriptive)", "n_interactions": int((inter["pair_family"] == fam).sum()),
                      "n_q_lt_0.1": int(((inter["pair_family"] == fam) & (inter["q_BH_family"] < 0.1)).sum()),
                      "n_p_perm_le_0.05": int(((inter["pair_family"] == fam) & (inter["p_perm"] <= 0.05)).sum())} for fam in FAM])
write_csv(glob, MOD, "b_partner_global_test.csv", "H7 global test: rhythmic B ↔ partner interactions (q < 0.1) vs 59 dihedral relabelings; curated families",
          required=["family", "n_interactions", "n_q_lt_0.1"])
log(f"H7: {len(inter)} interactions, {n_obs} with q < 0.1; null mean {np.mean(null):.1f}; exact p {p_glob:.3f}")

# ---------------- (2) H8 GeoMx
dec = pd.read_csv(RESULTS / "r3_spatial/geomx_deconvolution_aoi.csv")
bsum = dec[dec["cell_type"].isin(BST)].groupby(["aoi", "region", "noise"])["proportion"].sum().rename("b_lineage_proportion").reset_index()
gm = pd.read_csv(EXTERNAL / "geo/GSE312646/GSE312646_GeoMX_table.csv.gz", header=0, low_memory=False)
gm = gm.loc[:, ~gm.columns.str.startswith("Unnamed")]
gm = gm.iloc[4:].dropna(subset=["TargetName"]).drop_duplicates("TargetName").set_index("TargetName").apply(pd.to_numeric, errors="coerce").dropna()
sg = pd.read_csv(RESULTS / "r5_bcell/immgen_b_signatures.csv")
sg = sg[(sg["reference"] == "ImmGen microarray (GSE15907)") & sg["signature"].str.match(r"stage:(proB\.FrBC|preB\.Fr[CD]|B\.Fr[EF])\.BM")]
genes = sorted(set(sg["gene"]) & set(gm.index))
L = np.log2(gm.loc[genes] + 1)
Z = L.sub(L.mean(1), axis=0).div(L.std(1).replace(0, np.nan), axis=0)
bsum["immgen_b_signature_score"] = bsum["aoi"].map(Z.mean(0))
geo = []
for subset, d in [("control slides", bsum[bsum["noise"] == "control"]), ("all slides", bsum)]:
    for col in ["b_lineage_proportion", "immgen_b_signature_score"]:
        a = d.loc[d["region"] == "spiral_ganglion_neurons", col].dropna()
        b = d.loc[d["region"] == "supporting_cells", col].dropna()
        if len(a) and len(b):
            geo.append({"subset": subset, "measure": col, "n_neuron_aois": len(a), "n_supporting_aois": len(b),
                        "median_neuron": float(a.median()), "median_supporting": float(b.median()),
                        "p_wilcoxon_neuron_gt_supporting": float(stats.mannwhitneyu(a, b, alternative="greater").pvalue),
                        "n_signature_genes": len(genes) if col == "immgen_b_signature_score" else np.nan})
geo = pd.DataFrame(geo)
write_csv(bsum, MOD, "b_partner_geomx_aoi.csv", "GeoMx AOIs: B-lineage deconvolution proportion and ImmGen B-lineage signature score per AOI",
          required=["aoi", "region", "noise", "b_lineage_proportion", "immgen_b_signature_score"])
write_csv(geo, MOD, "b_partner_geomx.csv", "H8: B-lineage presence in spiral-ganglion-neuron vs supporting-cell AOIs (deconvolution and external signature)",
          required=["subset", "measure", "p_wilcoxon_neuron_gt_supporting"])
pri = geo[(geo["subset"] == "control slides") & (geo["measure"] == "b_lineage_proportion")].iloc[0]
agr = geo[(geo["subset"] == "control slides") & (geo["measure"] == "immgen_b_signature_score")]
prim = pd.DataFrame([
    {"hypothesis": "H7 B↔partner signals", "tissue": "cochlea", "primary_statistic": "number of B ↔ partner interactions with family q < 0.1 vs 59 relabelings (exact p)",
     "primary_p": p_glob, "detail": f"{n_obs} of {len(inter)} interactions q < 0.1 (null mean {np.mean(null):.1f})", "in_window": True},
    {"hypothesis": "H8 spatial adjacency", "tissue": "cochlea", "primary_statistic": "one-sided Wilcoxon: B-lineage deconvolution proportion, neuron > supporting AOIs (control slides)",
     "primary_p": pri["p_wilcoxon_neuron_gt_supporting"],
     "detail": f"median {pri['median_neuron']:.3f} vs {pri['median_supporting']:.3f}; external ImmGen B signature p "
               f"{agr['p_wilcoxon_neuron_gt_supporting'].iloc[0] if len(agr) else np.nan:.3f}", "in_window": bool(pri["median_neuron"] > pri["median_supporting"])}])
write_csv(prim, MOD, "b_primary_h7_h8.csv", "Primary hypotheses H7 (B↔partner signals) and H8 (spatial adjacency): primary statistics",
          required=["hypothesis", "tissue", "primary_p"])
log("\n" + geo.round(4).to_string())
log("\n" + inter[inter["q_BH_family"] < 0.1].head(25)[["direction", "source", "target", "ligand_complex", "receptor_complex", "peak_zt", "p_perm", "q_BH_family", "pair_family"]].round(3).to_string())
