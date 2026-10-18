"""Round 4, WP-C — conservation in a diurnal primate: baboon diurnal transcriptome atlas (GSE98965; Mure et al. 2018,
Science; 64 tissues incl. bone marrow 'BOM', one sample every 2 h over 24 h, log2(FPKM + 1)).

Baboons are diurnal, so phases are compared after clock alignment: per baboon tissue, offset = circular mean of
(peak − canonical mouse ZT phase) over core-clock genes rhythmic in that tissue (p < 0.05; lib/phase CANONICAL_CLOCK,
human symbols); our phases use this atlas's tissue-pool offsets (230). Mouse → human 1:1 orthologues (MGI HOM).
(1) baboon_clock_offsets.csv: offset, resultant, n clock genes per baboon tissue; clock-gene order in BOM vs canonical.
(2) baboon_bm_gene_concordance.csv: our BM Tier-1/2 genes (all, per cell type, non-clock) rhythmic in baboon BOM
    (p < 0.05): fraction within 3 h of our aligned phase vs a random-pairing null (1000 draws; as in 222).
(3) baboon_bm_signature_rhythms.csv: our BM cell-type marker signatures (construction of 222 B, orthologues) scored in
    BOM (mean z of log2 FPKM), cosinor (12 points), aligned peak; compared with our BM abundance peaks (r2_figure
    abundance_pct_rhythm, aligned): circular correlation across cell types with a type-permutation null.
(4) baboon_heatshock_phase.csv: HSF1 heat-shock set score per baboon tissue — raw ZT peak (light-referenced) and
    aligned peak; summary vs our night heat-shock programme (R05).
Outputs (results/r4_crossspecies/)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import cosinor, phase  # noqa: E402
from lib.io import EXTERNAL, RESULTS, log, write_csv  # noqa: E402

MOD = "r4_crossspecies"
rng = np.random.default_rng(7)
W = 2 * np.pi / 24
SETS = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))
HSF1 = set(SETS["systemic_cues"]["heat_shock_hsf1"])

# ---------------- orthologues
hom = pd.read_csv(EXTERNAL / "resources/mgi_hom_mouse_human.rpt", sep="\t", dtype=str)
hom = hom.rename(columns={"DB Class Key": "key", "Common Organism Name": "org", "Symbol": "symbol"})
mm = hom[hom["org"].str.startswith("mouse")].groupby("key")["symbol"].agg(list)
hs = hom[hom["org"].str.startswith("human")].groupby("key")["symbol"].agg(list)
m2h = {m[0]: h[0] for k in mm.index.intersection(hs.index) for h, m in [(hs[k], mm[k])] if len(h) == 1 and len(m) == 1}

# ---------------- baboon data
f = EXTERNAL / "geo/GSE98965/GSE98965_baboon_tissue_expression_FPKM.csv.gz"
raw = pd.read_csv(f)
raw = raw.dropna(subset=["Symbol"]).drop_duplicates("Symbol").set_index("Symbol").drop(columns="EnsemblID")
cols = {}
for c in raw.columns:
    m = re.match(r"^([A-Z]+)[._]ZT(\d+)", c)
    if m:
        cols[c] = (m.group(1), float(m.group(2)))
tissues = sorted({v[0] for v in cols.values()})
log(f"baboon: {raw.shape[0]} genes, {len(cols)} samples, {len(tissues)} tissues")


def tissue_fit(tcode):
    cc = [c for c, v in cols.items() if v[0] == tcode]
    zt = np.array([cols[c][1] for c in cc])
    x = raw[cc]
    x = x[(x.mean(1) >= 1)]
    y = np.log2(x + 1)
    fit = cosinor.fit(y.T.to_numpy(), zt)
    return pd.DataFrame({"gene": y.index, "mean_log2": y.mean(1).to_numpy(), "amp_log2": fit["amp"], "peak_zt": fit["peak"], "p": fit["p"]}), y, zt


CANON_H = {k.upper(): v for k, v in phase.CANONICAL_CLOCK.items()}
CANON_H["ARNTL"] = 23.5
off_rows, fits, mats = [], {}, {}
for t in tissues:
    fr, y, zt = tissue_fit(t)
    fits[t], mats[t] = fr.set_index("gene"), (y, zt)
    cg = fr[fr["gene"].isin(CANON_H) & (fr["p"] < 0.05)]
    if len(cg) >= 4:
        d = phase.circ_diff(cg["peak_zt"].to_numpy(), cg["gene"].map(CANON_H).to_numpy())
        mu, R = phase.circ_mean(d % 24)
        offv = float(phase.circ_diff(mu, 0))
    else:
        offv, R = np.nan, np.nan
    off_rows.append({"baboon_tissue": t, "n_clock_genes_rhythmic": len(cg), "offset_h": offv, "resultant": R,
                     "n_rhythmic_p05": int((fr["p"] < 0.05).sum()), "n_genes_tested": len(fr)})
offs = pd.DataFrame(off_rows)
MED_OFF = float(offs.loc[offs["n_clock_genes_rhythmic"] >= 4, "offset_h"].median())
offs["offset_used_h"] = offs["offset_h"].fillna(MED_OFF)
offs["offset_source"] = np.where(offs["offset_h"].notna(), "own clock genes", f"imputed: median of tissues with >= 4 rhythmic clock genes ({MED_OFF:+.2f} h)")
OFFB = offs.set_index("baboon_tissue")["offset_used_h"]
log("\n" + offs.sort_values("baboon_tissue").to_string())

ours_off = pd.read_csv(RESULTS / "r2_phase/clock_alignment_offsets.csv")
OFF = {t: float(ours_off[(ours_off["dataset"] == "this_atlas") & (ours_off["tissue"] == t) & (ours_off["unit"] == "tissue_pool")]["offset_canonical_h"].iloc[0])
       for t in ["bm", "cochlea"]}
UNITS = {"BOM": "bone marrow", "SPL": "spleen (immune-organ comparator)"}
BM = "BOM"
bm = fits[BM].copy()
bm["aligned"] = phase.wrap24(bm["peak_zt"] - OFFB[BM])
clock_bm = bm[bm.index.isin(CANON_H)].assign(canonical_zt=lambda d: d.index.map(CANON_H))
clock_bm["aligned_minus_canonical_h"] = phase.circ_diff(clock_bm["aligned"], clock_bm["canonical_zt"])
offs["note"] = offs["baboon_tissue"].map(UNITS).fillna("")
write_csv(offs, MOD, "baboon_clock_offsets.csv",
          "Baboon (diurnal) tissue clock offsets vs canonical mouse ZT phases (core-clock genes, p < 0.05); imputed where < 4 rhythmic clock genes",
          required=["baboon_tissue", "offset_h", "offset_used_h", "offset_source", "resultant", "n_clock_genes_rhythmic"])
write_csv(clock_bm.reset_index().rename(columns={"index": "gene"}), MOD, "baboon_bm_clock_genes.csv",
          "Core-clock genes in baboon bone marrow: raw ZT peak, aligned peak (imputed offset) and difference from the canonical mouse phase",
          required=["gene", "peak_zt", "aligned", "canonical_zt", "p"])

# ---------------- (2) gene-level phase concordance (our BM Tier-1/2 vs baboon BOM / SPL)
cat = pd.read_csv(RESULTS / "80_atlas/rhythmic_catalogue.csv")
cat = cat[cat["tissue"] == "bm"].copy()
cat["human"] = cat["gene"].map(m2h)
cat["aligned"] = phase.wrap24(cat["peak_zt"] - OFF["bm"])
CLOCKG = {k for k in phase.CANONICAL_CLOCK} | {"Clock", "Bhlhe40", "Bhlhe41", "Rora", "Arntl"}
conc = []
for unit in UNITS:
    fu = fits[unit].copy()
    fu["aligned"] = phase.wrap24(fu["peak_zt"] - OFFB[unit])
    brh = fu[fu["p"] < 0.05]
    for scope, g in [("ALL (Tier 1/2)", cat), ("Tier 1", cat[cat["tier"] == "tier1"])] + [(ct, s_) for ct, s_ in cat.groupby("cell_type")]:
        g = g.dropna(subset=["human"])
        g = g[g["human"].isin(brh.index)].sort_values("p").drop_duplicates("human")
        for subset, mm_ in [("all", g), ("non_clock", g[~g["gene"].isin(CLOCKG)])]:
            if len(mm_) < 8:
                continue
            a = mm_["aligned"].to_numpy()
            b = brh.loc[mm_["human"], "aligned"].to_numpy()
            d = np.abs(phase.circ_diff(a, b))
            pool = brh["aligned"].to_numpy()
            null = np.array([(np.abs(phase.circ_diff(a, rng.choice(pool, len(a), replace=False))) < 3).mean() for _ in range(1000)])
            mu, R = phase.circ_mean(phase.circ_diff(a, b) % 24)
            conc.append({"baboon_tissue": unit, "baboon_offset_source": offs.set_index("baboon_tissue").loc[unit, "offset_source"],
                         "n_baboon_rhythmic_p05": len(brh), "our_scope": scope, "genes": subset, "n_genes": len(mm_),
                         "frac_within_3h": float((d < 3).mean()), "null_frac_within_3h": float(null.mean()),
                         "p_perm": float((1 + (null >= (d < 3).mean()).sum()) / 1001), "median_abs_diff_h": float(np.median(d)),
                         "mean_offset_ours_minus_baboon_h": float(phase.circ_diff(mu, 0)), "offset_resultant": R})
conc = pd.DataFrame(conc)
write_csv(conc, MOD, "baboon_bm_gene_concordance.csv",
          "Clock-aligned phase concordance of our BM Tier-1/2 rhythms with baboon bone marrow and spleen (GSE98965; random-pairing null)",
          required=["baboon_tissue", "our_scope", "genes", "n_genes", "frac_within_3h", "null_frac_within_3h", "p_perm"])
log("\n" + conc[conc["our_scope"].isin(["ALL (Tier 1/2)", "Tier 1"]) | (conc["p_perm"] < 0.05)].round(3).to_string())

# ---------------- (3) cell-type signatures in baboon BM vs our BM abundance phases
mk = pd.read_csv(RESULTS / "00_audit/label_markers.csv")
mk = mk[(mk["logfoldchanges"] >= 2.5) & (mk["pvals_adj"] < 1e-10) & (mk["pct_nz_group"] >= 0.4)]
bm_types = pd.read_csv(RESULTS / "r2_figure/abundance_pct_rhythm.csv")
bm_types = bm_types[bm_types["tissue"] == "bm"].set_index("cell_type")
sigs = {}
for c, g in mk.groupby("cell_type"):
    if c not in bm_types.index:
        continue
    g = g.sort_values("scores", ascending=False)
    strict = g.loc[g["pct_nz_reference"] <= 0.1, "names"].tolist()
    relaxed = [x for x in g.loc[g["pct_nz_reference"] <= 0.25, "names"] if x not in strict]
    sigs[c] = (strict + relaxed)[:25]


def circ_corr(x, y):
    mx, my = np.angle(np.exp(1j * x).mean()), np.angle(np.exp(1j * y).mean())
    sx, sy = np.sin(x - mx), np.sin(y - my)
    den = np.sqrt((sx ** 2).sum() * (sy ** 2).sum())
    return float((sx * sy).sum() / den) if den > 0 else np.nan


allsr = []
for unit in UNITS:
    y, zt = mats[unit]
    z = y.sub(y.mean(1), axis=0).div(y.std(1).replace(0, np.nan), axis=0)
    srow = []
    for c, genes in sigs.items():
        hg = [m2h[g] for g in genes if g in m2h and m2h[g] in z.index]
        if len(hg) < 5:
            continue
        sc_ = z.loc[hg].mean(0).to_numpy()
        fit = cosinor.fit(sc_[:, None], zt)
        ab = bm_types.loc[c]
        srow.append({"baboon_tissue": unit, "cell_type": c, "n_genes": len(hg), "baboon_peak_zt": float(fit["peak"][0]),
                     "baboon_p": float(fit["p"][0]), "baboon_amp": float(fit["amp"][0]),
                     "baboon_aligned": float(phase.wrap24(fit["peak"][0] - OFFB[unit])),
                     "our_abundance_peak_zt": ab["peak_zt"], "our_abundance_p_perm": ab["p_perm"],
                     "our_aligned": float(phase.wrap24(ab["peak_zt"] - OFF["bm"]))})
    sr = pd.DataFrame(srow)
    sr["aligned_diff_h"] = phase.circ_diff(sr["our_aligned"], sr["baboon_aligned"])
    summ = []
    for label, s_ in [("all BM types", sr), ("types rhythmic in both (p < 0.1)", sr[(sr["baboon_p"] < 0.1) & (sr["our_abundance_p_perm"] < 0.1)])]:
        if len(s_) < 4:
            summ.append({"baboon_tissue": unit, "cell_type": f"SUMMARY: {label}", "n_genes": len(s_)})
            continue
        a, b = W * s_["our_aligned"].to_numpy(), W * s_["baboon_aligned"].to_numpy()
        obs = circ_corr(a, b)
        null = [circ_corr(a, rng.permutation(b)) for _ in range(2000)]
        summ.append({"baboon_tissue": unit, "cell_type": f"SUMMARY: {label}", "n_genes": len(s_), "circ_corr": obs,
                     "p_perm": (1 + sum(x >= obs for x in null)) / 2001, "frac_within_3h": float((np.abs(s_["aligned_diff_h"]) <= 3).mean())})
    allsr.append(pd.concat([sr, pd.DataFrame(summ)], ignore_index=True))
sr = pd.concat(allsr, ignore_index=True)
write_csv(sr, MOD, "baboon_bm_signature_rhythms.csv",
          "Our BM cell-type marker signatures in baboon bone marrow and spleen (cosinor, aligned) vs our BM abundance peaks; SUMMARY rows = circular correlation across types",
          required=["baboon_tissue", "cell_type", "n_genes"])
log("\n" + sr[sr["cell_type"].str.startswith("SUMMARY") | (sr["baboon_p"] < 0.05)].round(3).to_string())

# ---------------- (4) heat-shock programme across baboon tissues
hrow = []
hh = [g.upper() for g in HSF1]
for t in tissues:
    y, zt = mats[t]
    zz = y.sub(y.mean(1), axis=0).div(y.std(1).replace(0, np.nan), axis=0)
    g = [x for x in hh if x in zz.index]
    if len(g) < 5:
        continue
    fit = cosinor.fit(zz.loc[g].mean(0).to_numpy()[:, None], zt)
    hrow.append({"baboon_tissue": t, "n_genes": len(g), "peak_zt": float(fit["peak"][0]), "p": float(fit["p"][0]),
                 "aligned": float(phase.wrap24(fit["peak"][0] - OFFB[t])) if pd.notna(OFFB.get(t)) else np.nan})
hr = pd.DataFrame(hrow)
ours_hs = pd.read_csv(RESULTS / "80_atlas/rhythmic_catalogue.csv")
ours_hs = ours_hs[ours_hs["gene"].isin(HSF1)]
mu_raw, R_raw = phase.circ_mean(hr.loc[hr["p"] < 0.05, "peak_zt"].to_numpy())
mu_al, R_al = phase.circ_mean(hr.loc[(hr["p"] < 0.05) & hr["aligned"].notna(), "aligned"].to_numpy())
mo, Ro = phase.circ_mean(ours_hs["peak_zt"].to_numpy())
hr = pd.concat([hr, pd.DataFrame([
    {"baboon_tissue": "SUMMARY baboon (tissues p < 0.05): raw ZT", "n_genes": int((hr["p"] < 0.05).sum()), "peak_zt": mu_raw, "aligned": np.nan, "p": R_raw},
    {"baboon_tissue": "SUMMARY baboon (tissues p < 0.05): aligned", "n_genes": int(((hr["p"] < 0.05) & hr["aligned"].notna()).sum()), "peak_zt": np.nan, "aligned": mu_al, "p": R_al},
    {"baboon_tissue": "SUMMARY mouse (this atlas, HSF1 Tier-1/2 calls)", "n_genes": len(ours_hs), "peak_zt": mo,
     "aligned": float(phase.wrap24(mo - np.mean(list(OFF.values())))), "p": Ro}])], ignore_index=True)
hr["note"] = np.where(hr["baboon_tissue"].str.startswith("SUMMARY"), "p column = resultant length for summary rows", "")
write_csv(hr, MOD, "baboon_heatshock_phase.csv",
          "HSF1 heat-shock set rhythm per baboon tissue (raw ZT and clock-aligned) vs our night heat-shock programme",
          required=["baboon_tissue", "peak_zt", "aligned", "p"])
log("\n" + hr.tail(3).round(3).to_string())
