"""Round 10 (D143 plan, verification; D146) — consistency of the two versions: every number of the Science text (main
text, abstract, captions, Supplementary Text and captions) must occur in the Nature Immunology source (main text,
legends, Supplementary Information), and the key claims must use the same numbers and the same terms.
Numbers are compared as written (thousands separators dropped; '0.67 to 0.85' and '0.67--0.85' give the same numbers);
integers <= 20 without decimals are skipped (counts too small to trace, as in 905).
Output (lib/io.write_csv, results/r10_science/): version_consistency.csv (one row per Science number or claim check
with its status: found / missing).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/978_version_consistency.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import ROOT, log, write_csv  # noqa: E402

NI, SCI = ROOT / "manuscript" / "ni", ROOT / "manuscript" / "science"
NI_FILES = ["abstract.tex", "intro.tex", "results.tex", "discussion.tex", "legends_main.tex", "legends_ed.tex",
            "legends_si.tex", "supplementary.tex", "methods.tex"]
SCI_FILES = ["main.tex", "captions.tex", "sm_text.tex", "sm_captions.tex"]
NUM = re.compile(r"(?<![\w.^])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
# claims that must read the same in both versions (number strings that carry the argument)
CLAIMS = {"atlas cells": ["160,643", "116,278", "44,365"],
          "parabiosis ratios": ["0.03", "0.08", "0.04", "0.14", "1.49", "0.67", "4.39"],
          "perilymph": ["6--8", "55--86"],
          "typing": ["306", "80"],
          "trace check": ["91", "84", "29", "45"],
          "flow cohort": ["0.67", "0.52", "0.85", "0.0025", "0.40", "0.30", "0.53", "1.68", "1.34", "2.10"],
          "neutrophil split": ["0.35", "0.62", "0.33", "0.63", "0.00018", "0.040", "0.022", "0.099"],
          "CXCR4": ["0.29", "0.18", "0.47", "0.91", "0.70", "1.20"],
          "heat shock": ["17.1", "17.3", "0.22", "0.0019", "0.45", "0.06", "817", "183"],
          "clocks": ["0.55", "0.18", "0.92", "0.42", "0.05", "20.8", "2,845", "14.8"]}
TERMS = {"host-derived": "host-derived", "partner-derived": "partner-derived",
         "consistent with local supply": "consistent with local supply",
         "blood programme/program": "blood program"}


def plain(t):
    t = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in t.split("\n") if "\\newcommand" not in ln)
    t = re.sub(r"\\(cite|SF|label|ref|input|includegraphics|smFig|sciCap|citenum)\{[^}]*\}(\{[^}]*\})*", " ", t)
    t = re.sub(r"\b(?:ZT|GSE|CRA|Tier|Fig|figs?|data|S)~?\d+", " ", t)          # design labels and references
    t = re.sub(r"\b[A-Za-z]+[-]?\d[\w\-]*", " ", t)                              # gene / marker names with digits
    t = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)                                    # years
    return t


def numbers(t):
    out = []
    for m in NUM.finditer(plain(t)):
        s = m.group(1) + (m.group(2) or "")
        if "." not in s and int(s.replace(",", "")) <= 20:
            continue
        out.append(s)
    return out


def main():
    ni_text = "\n".join((NI / f).read_text(encoding="utf-8") for f in NI_FILES)
    ni_nums = set(numbers(ni_text))
    rows = []
    for f in SCI_FILES:
        for n in sorted(set(numbers((SCI / f).read_text(encoding="utf-8")))):
            rows.append(dict(kind="number", where=f, item=n, status="found" if n in ni_nums else "missing"))
    sci_text = " ".join("\n".join((SCI / f).read_text(encoding="utf-8") for f in ["main.tex", "captions.tex"]).split())
    ni_main = " ".join("\n".join((NI / f).read_text(encoding="utf-8") for f in ["abstract.tex", "results.tex",
                                                                              "discussion.tex", "legends_main.tex"]).split())
    for claim, nums in CLAIMS.items():
        for n in nums:
            n_sci = n.replace("--", " to ")
            ok = (n in sci_text or n_sci in sci_text) and n in ni_main
            rows.append(dict(kind="claim", where=claim, item=n, status="found" if ok else "missing"))
    for term, sci_form in TERMS.items():
        ni_form = sci_form.replace("program", "programme")
        ok = sci_form in sci_text and ni_form in ni_main
        rows.append(dict(kind="term", where=term, item=sci_form, status="found" if ok else "missing"))
    df = pd.DataFrame(rows)
    write_csv(df, "r10_science", "version_consistency.csv", "Science vs Nature Immunology versions (978, D146): every "
              "number of the Science text traced to the NI source, key claim numbers and terms in both main texts",
              required=["kind", "where", "item", "status"])
    for k, g in df.groupby("kind"):
        miss = g[g.status == "missing"]
        log(f"{k}: {len(g) - len(miss)}/{len(g)} found" + (": missing " + "; ".join(f"{r.where} {r.item}" for r in
                                                                                      miss.itertuples())
                                                       if len(miss) else ""))


if __name__ == "__main__":
    main()
