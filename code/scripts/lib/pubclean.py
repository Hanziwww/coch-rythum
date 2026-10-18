"""Publication clean-up of the tables released with the paper (Supplementary Tables and Source Data; round 9, phase G).

Released tables carry no project-internal provenance:
  columns   internal paths and provenance (source_table, source_file, transform, finding_id, register_id,
            call_set_finding) and display helpers (colour, class_colour, order, plot_order, class_order) are dropped;
  text      decision / log identifiers (D17, D29, D54, D77, I098, O01) and script cross-references ('(77 / R14)',
            '(as G02)') are removed; agreement rules given as pointers to script documentation are written out
            (RULES); "the user's" becomes "the authors'"; the noise-cohort shorthand ('Ma 2025',
            'Ma-') and analysis-round tags are reworded; 'nan' fragments of joined labels are removed; raw image files
            are named relative to their image set (without the local data/photos/ prefix).
Finding identifiers (A01, G02, R14, ...) remain only in Supplementary Tables 8 and 9 (D134 numbering), where Supplementary Table 9
defines them. Hypothesis labels (H1-H12, M1-M9), timing templates (T1-T3) and exploratory grades (E0-E2) are defined in
the legends. residual() lists internal markers still present (908 and 909 assert that it is empty).
"""
import re

import pandas as pd

from lib.io import RESULTS

DROP = {"source_table", "source_file", "transform", "finding_id", "register_id", "call_set_finding", "colour",
        "class_colour", "order", "plot_order", "class_order"}
SUBS = [
    (r"\s*\(\d{2,3} / [A-Z]\d{2}\)", ""),                     # '(77 / R14)': script / finding cross-reference
    (r"\(raw, as [A-Z]\d{2}\)", "(raw)"),                     # 'A/B our data (raw, as G02)'
    (r"\s*\(as [A-Z]\d{2}\)", ""),                            # 'raw (as G02)'
    (r"\s*\((?:D|I|O)\d{2,3}\)", ""),                         # '(D54)', '(D17)', '(D77)'
    (r"[;,]\s*(?:D|I|O)\d{2,3}\b", ""),                       # '; D54', ', D29'
    (r"\bthe user's\b", "the authors'"),
    (r": / marrow-content index", ": divided by the marrow-content index"),
    (r"\b680 B1 definition\b", "time-matched marker definition"),
    (r"\bB1 follow-up\b", "time-matched follow-up"),
    (r"\bMa-(?=inconclusive|supported|opposed)", ""),
    (r"\(Ma 2025\)", "(noise cohort)"),
    (r"\bMa 2025 ", ""),
    (r"\(round-\d+ exploratory\)", "(exploratory)"),
    (r"\s*\(round \d+\)", ""),
    (r"\s*\(register rule\)", ""),
    (r"\bsee memory circadian-dataset\b", "as in Methods"),
    (r"(?<=\S) nan\b", ""),                                   # 'external baboon spleen nan'
    (r"\bdata/photos/", ""),                                  # raw image files named relative to the image set
    (r"NeuMAP / GSE200145", "NeuMap, GSE266518"),             # data analysed from GSE266518 (Supplementary Table 20)
    (r"\bNeuMAP\b", "NeuMap"),
    (r"\b210 PC1\b", "PC1"),                                  # tissue-wide sample pattern (script cross-reference)
]
# agreement rules of the myeloid method lines that the resource tables gave as pointers to script documentation
# (scripts 650 and 660), keyed by the start of the method text
RULES = {
    "Cxcl12 mRNA rhythm in the representative sender": "exact P <= 0.05 (pseudobulk log2 CPM of the sender)",
    "NicheNet CXCL12 activity percentile": "CXCL12 activity percentile >= 0.9 for a day or night programme of a "
                                           "neutrophil receiver",
    "glucocorticoid-response phase aligned with Cxcr4 / Cxcr2": "glucocorticoid-response programme rhythmic (exact "
        "P <= 0.05) in >= 1 neutrophil label, with its peak within 4 h of the Cxcr4 / Cxcr2 peak",
    "GSE198770 BM maturation-axis phase": "cosinor P < 0.05 and peak within 4 h of our marrow M1 peak",
    "GSE198770 BM granule-class order": "primary, secondary and tertiary granule programmes peak in forward order within "
                                        "12 h, with >= 2 of 3 at P < 0.05",
    "external tissue-vs-marrow (ageing / circulating) axis": "rhythmic in our pooled neutrophils (exact P <= 0.05) with "
                                                             "its peak within 4 h of the M1 peak",
    "GSE312298 BM programme night": "Spearman r > 0 and P < 0.05 between observed and predicted night - day changes of "
                                    "the function programmes",
    "GSE25585 macrophage mono": "cosinor P < 0.05 and peak within 4 h of our M5 peak",
    "time-of-day clock genes in external neutrophils": "r > 0 and P < 0.05 in >= 2 of the three data sets",
    "GSE25585 macrophage clock offset": "resultant length >= 0.7 and offset within 4 h of our monocyte and macrophage "
                                        "offset",
    "external tissue-vs-marrow (NeuMAP / GSE200145) higher": "score higher in cochlear than in marrow neutrophils at "
                                                             "all six time points (sign test)",
}
RESIDUAL = re.compile(r"\b(?:D\d{2,3}|I\d{3}|O\d{2})\b|\buser\b|\bheadline\b|\bround[- ]\d|\bregister (?:rule|finding)\b|"
                      r"\bsee memory\b|\b(?:r\d+_[a-z_]+|results|data)/|\(\d{2,3} / [A-Z]\d{2}\)|\bas [AGR]\d{2}\b|"
                      r"\bMa 2025\b|\bMa-|\bdocstring\b|\b210 PC1\b")


def _claim_ladder(df):
    """Fig. 7i: the claim as displayed (claim_short) and the atlas grade of register findings spelled out."""
    reg = pd.read_csv(RESULTS / "r2_audit" / "positive_findings_register.csv").set_index("finding_id")["verdict"]
    first = df["claim"].astype(str).str.extract(r"^([A-Z]\d{2})\b", expand=False)
    st = df["confirmatory_status (unchanged)"].astype(str)
    st = st.where(~st.str.startswith("register finding"), first.map(reg).fillna("descriptive"))
    df = df.assign(**{"confirmatory_status (unchanged)": st}).drop(columns=["claim"])
    df = df.rename(columns={"claim_short": "claim", "confirmatory_status (unchanged)": "atlas_status",
                            "updated_exploratory_grade": "exploratory_grade"})
    return df[["claim"] + [c for c in df.columns if c != "claim"]]


def clean(df: pd.DataFrame, name: str = "", drop=()) -> pd.DataFrame:
    """Copy of a released table without internal columns and with internal references removed from text cells."""
    df = df.drop(columns=[c for c in df.columns if c in DROP or c in drop])
    if {"rule", "method"} <= set(df.columns):
        df = df.copy()
        pointer = df["rule"].astype(str).str.match(r"^see \d{3} docstring$")
        for i in df.index[pointer]:
            hits = [r for k, r in RULES.items() if str(df.at[i, "method"]).startswith(k)]
            assert len(hits) == 1, df.at[i, "method"]
            df.at[i, "rule"] = hits[0]
    if name.startswith("f6i_claim_ladder"):
        df = _claim_ladder(df)
    if name.startswith("f6a_design_audit"):
        df = df.rename(columns={"ma2025": "noise_cohort"})
        df["source"] = df["source"].astype(str).str.replace(r"data/(CRA\d+)\.xlsx", r"GSA \1 records", regex=True)
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object:
            continue
        m = df[c].notna()
        v = df.loc[m, c].astype(str)
        for pat, rep in SUBS:
            v = v.str.replace(pat, rep, regex=True)
        df.loc[m, c] = v
    return df


def residual(df: pd.DataFrame, skip=("gene", "human_orthologue")) -> list:
    """(column, value) pairs that still carry an internal marker (empty for a clean table)."""
    out = [(c, "<column name>") for c in df.columns if RESIDUAL.search(str(c))]
    for c in df.columns:
        if c in skip or df[c].dtype != object:
            continue
        s = df[c].dropna().astype(str)
        out += [(c, x) for x in s[s.str.contains(RESIDUAL)].unique()[:3]]
    return out
