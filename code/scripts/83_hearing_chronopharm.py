"""Module 6.6 — hearing-relevant rhythms and chronopharmacology windows.

From the gene-level rhythm tables (cell_type level): genes in the curated hearing sets (K⁺ recycling, hair-cell
function, synaptic ribbon, SGN, blood–labyrinth barrier, ototoxic uptake/protection, deafness genes, GR/MR axis)
per tissue × cell type with rhythm statistics. Chronopharmacology window table for drug-relevant genes (GR/MR axis,
ototoxic uptake/protection, BLB transporters) that are Tier 1/2 or nominally rhythmic with FC >= 1.25:
peak window (peak ± 3 h) and trough window (trough ± 3 h) in ZT and in wall-clock time (lights on 08:00).
Outputs (results/83_hearing/): hearing_gene_rhythm.csv, chronopharmacology_windows.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import LIGHTS_ON, RESULTS, log, write_csv  # noqa: E402

MOD = "83_hearing"
SC = yaml.safe_load(open(Path(__file__).resolve().parent / "resources/curated_sets.yaml"))["hearing"]
DRUG_SETS = ["glucocorticoid_mineralocorticoid", "ototoxic_uptake_protection", "blood_labyrinth_barrier"]
membership = {}
for s, genes in SC.items():
    for g in genes:
        membership.setdefault(g, []).append(s)
rows = []
for t in ["cochlea", "bm"]:
    df = pd.read_csv(RESULTS / f"33_rhythm_tiers/rhythm_genes_all_cell_type_{t}.csv")
    df = df[df["gene"].isin(membership)]
    df["hearing_sets"] = df["gene"].map(lambda g: ";".join(membership[g]))
    rows.append(df)
hg = pd.concat(rows, ignore_index=True)
keep = ["tissue", "group", "gene", "hearing_sets", "mean_logCPM", "AveExpr", "ptt_fc", "amp_log2", "peak_zt",
        "peak_ci_low", "peak_ci_high", "p", "q_BH", "q_perm", "loto_phase_sd_h", "tier"]
hg = hg[[c for c in keep if c in hg.columns]].rename(columns={"group": "cell_type"})
write_csv(hg.sort_values(["tissue", "cell_type", "p"]), MOD, "hearing_gene_rhythm.csv",
          "Rhythm statistics of hearing-relevant genes per tissue × cell type",
          required=["tissue", "cell_type", "gene", "hearing_sets", "peak_zt", "p", "tier"])
d = hg[hg["hearing_sets"].str.contains("|".join(DRUG_SETS)) &
       (hg["tier"].isin(["tier1", "tier2"]) | ((hg["p"] < 0.05) & (hg["ptt_fc"] >= 1.25)))].copy()
d["trough_zt"] = (d["peak_zt"] + 12) % 24
for k in ["peak", "trough"]:
    d[f"{k}_window_zt"] = d[f"{k}_zt"].map(lambda z: f"ZT{(z - 3) % 24:04.1f}–ZT{(z + 3) % 24:04.1f}")
    d[f"{k}_clock_time"] = d[f"{k}_zt"].map(lambda z: f"{(z + LIGHTS_ON) % 24:04.1f}h")
write_csv(d.sort_values(["gene", "tissue", "cell_type"]), MOD, "chronopharmacology_windows.csv",
          "Peak/trough windows (ZT and clock time) of rhythmic drug-target/transport genes per cell type",
          required=["tissue", "cell_type", "gene", "peak_zt", "trough_zt", "ptt_fc", "tier"])
log("done")
