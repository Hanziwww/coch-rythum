"""Round 2, WP6 — feasibility of public treatment-response and spatial datasets (user decision: public data only).

One row per candidate: what it is (accession, publication, species/strain/age, tissue, modality, design, samples), how
it can be obtained from the server (repository, processed level, size, route), what processing it needs, whether our
annotations can be transferred, which question it answers for the circadian immune atlas, value and effort, and the
decision (integrated / pilot done / go next / blocked / no-go) with the reason. Accessions, designs and sizes were
checked against GEO (series records, sample characteristics, supplementary-file headers) on 2026-10-12. Pilot
results (251/252) are summarised from results/r2_pilots/.
Output (results/r2_feasibility/): feasibility_treatment_spatial.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, log, write_csv  # noqa: E402

P = RESULTS / "r2_pilots"


def rd(name):
    f = P / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


val, ct, ov, sg, kin, gsx, gsets = (rd("pilot_daynight_rhythm_validation.csv"), rd("pilot_collection_time_inference.csv"),
                                    rd("pilot_noise_geneset_overlap.csv"), rd("pilot_noise_signatures.csv"),
                                    rd("pilot_noise_kinetics_genesets.csv"), rd("pilot_geomx_signatures.csv"),
                                    rd("pilot_geomx_genesets.csv"))
de1 = rd("pilot_GSE107086_de.csv")


def summ_107086():
    if not len(val):
        return "pilot not run"
    b = ct[ct["is_best"]]
    v = val[(val["collection_time"] == "inferred from GSE88954") & (val["genes"] == "non-clock rhythmic (p<0.05)")]
    n_int = int(((de1["contrast"] == "noise_night_vs_day_shamsurg") & (de1["q_BH"] < 0.1)).sum()) if len(de1) else np.nan
    n_nd = int(((de1["contrast"] == "noise_day_shamsurg") & (de1["q_BH"] < 0.05)).sum()) if len(de1) else np.nan
    n_nn = int(((de1["contrast"] == "noise_night_shamsurg") & (de1["q_BH"] < 0.05)).sum()) if len(de1) else np.nan
    s = "; ".join(f"collection time ({r.reference}, {r.genes}) {r.day_collection_internal_h:.1f} h (r {r.pearson_r:.2f})"
                  for r in b.itertuples())
    if len(v):
        s += (f"; our cochlear rhythms predict the independent night−day difference: non-clock r {v['pearson_r'].iloc[0]:.2f} "
              f"(n {int(v['n_genes'].iloc[0])}, perm p {v['p_perm'].iloc[0]:.4f})")
    s += f"; noise-responsive genes q<0.05: day {n_nd}, night {n_nn}; noise × time-of-day q<0.1: {n_int}"
    if len(sg):
        top = sg.sort_values("p_noise_x_night_shamsurg").head(3)
        s += "; signatures with time-dependent noise response: " + ", ".join(
            f"{r.signature} (p {r.p_noise_x_night_shamsurg:.3f})" for r in top.itertuples())
    return s


def summ_geomx():
    if not len(gsx):
        return "pilot not run"
    loc = gsx.sort_values("neuron_minus_supporting_control", ascending=False)
    s = ("localisation (neuron − supporting, z): top " + ", ".join(f"{r.signature} {r.neuron_minus_supporting_control:+.2f}"
                                                                 for r in loc.head(3).itertuples()) +
         "; bottom " + ", ".join(f"{r.signature} {r.neuron_minus_supporting_control:+.2f}" for r in loc.tail(3).itertuples()))
    if len(gsets):
        h = gsets[gsets["gene_set"].isin(["heat shock", "IEG"])]
        s += "; noise effect: " + ", ".join(f"{r.gene_set}/{r.region} {r.mean_noise_log2FC:+.2f}" for r in h.itertuples())
    return s


def summ_kin():
    if not len(kin):
        return "pilot not run"
    k = kin[kin["gene_set"].isin(["IEG", "heat shock", "our cochlea rhythmic, night peak", "our cochlea rhythmic, day peak"])]
    return "; ".join(f"{r.gene_set} {r.contrast.replace('_vs_mock', '')} {r.mean_logFC:+.2f}" for r in k.itertuples())


C = [
    dict(candidate="Day vs night noise trauma × adrenalectomy (cochlea, bulk)", accession="GSE107086 (SuperSeries GSE107087)",
         reference="Cederroth et al. 2019 Curr Biol, PMID 31353184", species="mouse CBA/Sca, 12 wk", tissue="whole cochlea",
         modality="bulk RNA-seq (raw counts)", design="noise (6–12 kHz, 100 dB SPL, 1 h) at ZT3 or ZT15 vs sham noise × ADX vs "
         "sham surgery; GEO lists collection ZT11/ZT23", n_samples=24, data_level="processed raw counts", repository="GEO",
         size_mb=0.85, route="direct NCBI (server)", processing="none (counts)", label_transfer="signatures / tissue-level "
         "predictions (bulk)", question="are our cochlear rhythms reproduced in an independent cohort; does the noise "
         "response depend on time of day and on glucocorticoids; which cell-type programmes respond", value="high",
         effort="low", decision="pilot done", reason="same laboratory as GSE88954; direct time-of-day × treatment design",
         pilot_outputs="r2_pilots/pilot_GSE107086_de.csv; pilot_collection_time_inference.csv; pilot_daynight_rhythm_validation.csv; "
         "pilot_noise_geneset_overlap.csv; pilot_noise_signatures.csv", pilot_result=summ_107086()),
    dict(candidate="Noise-trauma spatial transcriptomics (GeoMx; spiral ganglion and supporting-cell AOIs)",
         accession="GSE312646", reference="PMID 41415422 (heat-shock activation by TRC-051384 protects SGNs from NIHL)",
         species="mouse CBA/CaJ, 16 wk, male", tissue="cochlea sections", modality="GeoMx DSP (AOI × gene, normalised)",
         design="100 dB broadband noise 2 h vs mock (N = 5 mice per group); 51 AOIs", n_samples=51,
         data_level="normalised AOI table", repository="GEO", size_mb=0.37, route="direct NCBI (server)",
         processing="none (table)", label_transfer="signature scores of our cell types per AOI",
         question="spatial localisation of our cell-type signatures and rhythmic genes; regional noise response of the "
         "night stress programme", value="medium–high", effort="low", decision="pilot done",
         reason="only adult cochlear spatial dataset found; two regions only, no time-of-day",
         pilot_outputs="r2_pilots/pilot_geomx_signatures.csv; pilot_geomx_genesets.csv", pilot_result=summ_geomx()),
    dict(candidate="Noise-response kinetics (cochlea, bulk)", accession="GSE312476", reference="PMID 41415422",
         species="mouse CBA/CaJ, 16 wk, male", tissue="whole cochlea", modality="bulk RNA-seq (normalised per sample)",
         design="mock vs 100 dB 2 h, sacrificed 0, 4, 8 h after (n = 3)", n_samples=12, data_level="normalised values",
         repository="GEO", size_mb=6.9, route="direct NCBI (server)", processing="none", label_transfer="gene sets",
         question="are night-programme (IEG / heat-shock / NF-κB) genes acutely noise-inducible", value="medium",
         effort="low", decision="pilot done", reason="small, clean time course",
         pilot_outputs="r2_pilots/pilot_GSE312476_de.csv; pilot_noise_kinetics_genesets.csv", pilot_result=summ_kin()),
    dict(candidate="Day (ZT2–4) vs night (ZT14–16) noise exposure, cochlear scRNA-seq (97k cells, ~40 % immune)",
         accession="NGDC GSA CRA019820 (scRNA), CRA019827 (bulk)", reference="Ma et al. 2025 Neurosci Bull",
         species="mouse", tissue="cochlea", modality="scRNA-seq + bulk RNA-seq (raw FASTQ)",
         design="identical noise in the day or night, tissue 3 d later", n_samples=np.nan, data_level="raw reads only",
         repository="NGDC (China)", size_mb=np.nan, route="NGDC download pages sit behind browser verification (not "
         "bypassed); needs user-assisted download or direct FTP links", processing="alignment (kb-python / kallisto, Ensembl "
         "cDNA index), QC, label transfer from our cochlear embedding (several hours on 24 cores)",
         label_transfer="yes (10x; same tissue)", question="do our rhythmic immune states / abundance phases predict the "
         "day-vs-night noise response at single-cell level", value="highest", effort="high", decision="blocked (go next)",
         reason="best scientific match; access requires the user", pilot_outputs="", pilot_result=""),
    dict(candidate="CD45+ cochlear immune cells after noise (3, 7, 14 d) and control", accession="GSE167078",
         reference="PMID 34592158", species="mouse", tissue="cochlea (CD45+ sorted)", modality="scRNA-seq",
         design="noise exposure, 3 time points after + control (2 GEO samples, multiplexed)", n_samples=2,
         data_level="per-sample count matrices", repository="GEO", size_mb=21.7, route="direct NCBI",
         processing="QC, demultiplexing by sample annotation, label transfer to our cochlear immune types",
         label_transfer="yes (immune)", question="which of our rhythmic immune types expand / change state after noise",
         value="medium–high", effort="medium", decision="go next", reason="small download; complements GSE107086",
         pilot_outputs="", pilot_result=""),
    dict(candidate="CCR2-dependent SPP1+ macrophages after noise", accession="GSE346691", reference="GEO series 2026-09",
         species="mouse", tissue="cochlea", modality="scRNA-seq", design="noise (3 d) vs control; Ccr2 / Spp1 knockouts",
         n_samples=7, data_level="count matrices", repository="GEO", size_mb=446.4, route="direct NCBI",
         processing="QC, integration, label transfer", label_transfer="yes", question="monocyte recruitment (our classical "
         "monocyte rhythm) in noise injury", value="medium", effort="medium", decision="go next",
         reason="recruitment mechanism relevant to the monocyte rhythm", pilot_outputs="", pilot_result=""),
    dict(candidate="Cochlear scRNA-seq in hidden hearing loss vs NIHL", accession="GSE252285", reference="GEO series",
         species="mouse", tissue="cochlea", modality="scRNA-seq", design="HHL vs NIHL vs control (3 samples)", n_samples=3,
         data_level="combined matrix", repository="GEO", size_mb=340.4, route="direct NCBI", processing="QC, label transfer",
         label_transfer="yes", question="cell-type injury responses", value="low", effort="medium", decision="no-go",
         reason="no time-of-day information; one sample per condition", pilot_outputs="", pilot_result=""),
    dict(candidate="Cisplatin-treated adult stria vascularis scRNA-seq", accession="GSE165662",
         reference="Taukulis et al. 2021 Front Mol Neurosci, PMID 34566577", species="mouse CBA/J", tissue="stria vascularis",
         modality="scRNA-seq", design="single cisplatin-treated sample (controls in other series)", n_samples=1,
         data_level="count matrix", repository="GEO", size_mb=22.1, route="direct NCBI", processing="QC, label transfer",
         label_transfer="SV marginal / intermediate cells", question="chronotoxicity: overlap of cisplatin-responsive SV genes "
         "with SV rhythmic genes", value="low", effort="medium", decision="no-go",
         reason="no within-series control; treatment effect confounded with batch", pilot_outputs="", pilot_result=""),
    dict(candidate="Lung ± neutrophil depletion at ZT4 and ZT16", accession="GSE106349", reference="PMID 30282719",
         species="mouse C57BL/6", tissue="lung", modality="bulk RNA-seq", design="anti-Ly6G or DT depletion × ZT4/ZT16",
         n_samples=35, data_level="processed", repository="GEO", size_mb=np.nan, route="direct NCBI", processing="none",
         label_transfer="signatures", question="neutrophil dependence of time-of-day expression",
         value="low", effort="low", decision="no-go", reason="redundant with the GSE198770 time courses already integrated",
         pilot_outputs="", pilot_result=""),
    dict(candidate="Treatment datasets already integrated", accession="GSE88954; GSE198770; GSE86619",
         reference="GSE88954: SuperSeries GSE107087 (Cederroth et al. 2019, PMID 31353184; ADX); GSE198770: neutrophil "
         "depletion (anti-Ly6G, DT) and neutrophil Bmal1 KO time courses; GSE86619: PMID 30709741 (Cxcr2/Cxcr4/Arntl KO "
         "neutrophils)", species="mouse", tissue="cochlea; BM, gut, liver, lung, skin; "
         "blood neutrophils", modality="bulk RNA-seq time courses", design="ADX vs sham; neutrophil depletion / KO vs control",
         n_samples=np.nan, data_level="processed", repository="GEO", size_mb=np.nan, route="done", processing="done (70/71, 220/221)",
         label_transfer="BayesPrism, signatures", question="glucocorticoid / neutrophil dependence of rhythms",
         value="high", effort="done", decision="integrated", reason="rounds 1–2", pilot_outputs="71_external/*; r2_external/*",
         pilot_result=""),
    dict(candidate="Bone-marrow niche spatial / LCM data (mouse)", accession="e.g. GSE276784 (inflammation remodels BM niche; "
         "modality to verify)", reference="GEO series 2025", species="mouse", tissue="long-bone marrow",
         modality="spatial / LCM (to verify)", design="inflammation vs control", n_samples=89, data_level="to verify",
         repository="GEO", size_mb=np.nan, route="direct NCBI", processing="deconvolution with our BM atlas",
         label_transfer="yes (BM)", question="co-localisation of rhythmic immune states with niche cells",
         value="low–medium", effort="medium", decision="no-go (this round)",
         reason="no skull / temporal-bone marrow spatial data found; long-bone inflammation design", pilot_outputs="", pilot_result=""),
    dict(candidate="Adult cochlea / temporal bone spatial (Visium, Slide-seq, MERFISH)", accession="none found",
         reference="GEO / bioRxiv searches 2026-10-12", species="mouse", tissue="cochlea", modality="spatial", design="",
         n_samples=np.nan, data_level="", repository="", size_mb=np.nan, route="", processing="", label_transfer="",
         question="", value="", effort="", decision="not available",
         reason="only the GeoMx dataset (GSE312646) and a developing-cochlea spatial preprint were found", pilot_outputs="",
         pilot_result=""),
]
fz = pd.DataFrame(C)
write_csv(fz, "r2_feasibility", "feasibility_treatment_spatial.csv",
          "Feasibility of public treatment-response and spatial datasets: design, access, processing, question, decision, "
          "pilot results", required=["candidate", "accession", "decision", "reason"])
log("\n" + fz[["accession", "decision", "value", "effort"]].to_string())
