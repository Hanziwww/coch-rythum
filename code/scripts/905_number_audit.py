"""Round 9 writing, phase F (D99 plan section 7) — number audit of the LaTeX manuscript.

Every number in the abstract, Introduction, Results, Discussion, the figure / Extended Data / Supplementary legends
and the Supplementary Information is extracted from the active manuscript (config paths.manuscript; manuscript/ni since round 10) *.tex and matched against the registered result
tables (results/r9_figures, results/r9_flow, results/r4_resource, results/r10_parabiosis since D133, key_findings,
r3_conclusions, the findings register): a number matches when some table value, its percentage (x 100) or its absolute value equals it at the
number's own precision (decimals; significant figures for values below 1, including scientific notation). Numbers
inside table cells written as text (for example '0.67 (0.52-0.85)') are indexed too.

Not audited (status 'design' or 'trivial'): figure / table / note references, accession numbers, gene and marker names
with digits, integer ZT values (sampling design), clock times, tube / readout / tier labels, citations, and integers
<= 20 without a unit (counts too small to verify against tables); model and sampler settings fenced by the comment lines
'% audit:design-begin' / '% audit:design-end' (status 'design'; Supplementary Note 3). The Methods are out of scope
(parameters, checked against the scripts when drafted); the Reporting Summary draft is audited like the SI (phase G). Unmatched numbers are listed for review; each must be corrected in the text or
signed off with its derivation in <manuscript>/number_signoff.csv (columns: file, number, context, reason).
P values written with one significant figure are flagged ('p_format') so that text and figures share two
significant figures.

Output (lib/io.write_csv): results/r9_text/number_audit.csv.
Run (local; the manuscript is local): COCH_ROOT=/path/to/coch-rythum python scripts/905_number_audit.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import MANUSCRIPT, RESULTS, ROOT, log, write_csv  # noqa: E402

SCIENCE = "--science" in sys.argv[1:]               # D144: audit the Science version (manuscript/science) instead
NN = (ROOT / "manuscript" / "science") if SCIENCE else MANUSCRIPT   # active manuscript (config paths.manuscript)
TEXT_FILES = ["abstract.tex", "intro.tex", "results.tex", "discussion.tex", "legends_main.tex", "legends_ed.tex",
              "legends_si.tex", "supplementary.tex", "reporting_summary.tex"]
if SCIENCE:
    TEXT_FILES = [f for f in ["main.tex", "captions.tex", "sm_text.tex", "sm_supmethods.tex", "sm_captions.tex",
                              "sm_data.tex", "cover_letter.tex"] if (NN / f).exists()]
DESIGN_FENCE = "% audit:design"                      # '-begin' / '-end' comment lines around model settings
SIGNOFF = NN / "number_signoff.csv"
SKIP_TABLES = re.compile(r"(umap|contours|outliers|_events|cell_margin)")
MAX_BYTES = 60e6

# ---------------------------------------------------------------------------------------------- text -> numbers
REF = re.compile(r"(?:Extended\s+Data\s+|Supplementary\s+)?(?:Figs?|Tables?|Notes?|Methods|Note)\.?[~\s]*\d+[a-z]*"
                 r"(?:\s*(?:,|--|and|-)\s*(?:\d+[a-z]*|[a-z]\b))*")
MASK = [
    re.compile(r"\\(?:cite|citenum|citep|citet|ref|label|input|authorinput|addcontentsline|newcommand)\{[^}]*\}"
               r"(?:\{[^}]*\})?"),
    re.compile(r"\b(?:GSE|GSM|CRA|SRR|PRJNA)\d+\b"),
    re.compile(r"\bZT\d+(?!\.\d)"),                       # integer ZT = sampling design
    re.compile(r"\b\d{1,2}:\d{2}\b"),                    # clock times
    re.compile(r"\b(?:Tier|Tiers)[-~\s]*\d(?:/\d)?"),
    re.compile(r"\b[A-Za-z]+[-~]?\d[\w\-]*"),           # gene / marker / label names with digits (Hspa1a, CD45, P1, T8, E2)
    re.compile(r"\\log_?\{?2\}?|log\$?_\{?2\}?\$?|\\pi_\{?1\}?|\\pi_1"),
    re.compile(r"\b\d+[-~](?:h|kb|df|day|days|week|weeks|month|months|bit|nm|colour|fold change)\b"),
    re.compile(r"\b(?:19|20)\d{2}\b"),                    # years
    re.compile(r"\$\^\{?[+\-]?\}?\$"),                   # bare superscript signs
]
NUM = re.compile(r"(?<![\w.])([+\-\u2212]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
                 r"(\s*\\?times\s*10\^\{?([+\-\u2212]?\d+)\}?)?\s*(\\?%|-fold|~?h\b)?")


def clean_line(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)
    for m in MASK:
        s = m.sub(" ", s)
    s = REF.sub(" ", s)
    s = re.sub(r"\\(?:textbf|textit|emph|pnl|mathrm|text)\{([^}]*)\}", r"\1", s)
    s = s.replace("$-$", "-").replace("$+$", "+").replace("--", "-").replace("~", " ")
    s = s.replace("\\leq", "<=").replace("\\geq", ">=").replace("\\pm", "+-")
    s = re.sub(r"\$([+\-]?\d)", r"\1", s)
    return s.replace("$", " ")


RESULT_SCOPES = ["fig1", "fig2", "fig3", "fig4", "fig4", "fig5", "fig6"]   # Results subsections in order (D118, D134: the
                                                                           # channel (Fig. 4) and the alternation (Fig. 5) share the f4 tables)
SCOPE_TABLES = {                                                           # scope -> source-table name prefixes
    "fig1": ("f1", "ed1_", "ed2_", "ed3_", "ed4_"), "fig2": ("f2", "ed5_", "ed6_"), "fig3": ("f3", "ed6_", "ed7_"),
    "fig4": ("f4", "ed1_", "ed8_", "ed9_", "flow_", "f5i_", "f5a_"), "fig5": ("f5", "ed9_", "flow_", "f4f_"),
    "fig6": ("f6", "ed10_"), "s1": ("flow_",), "note1": ("flow_", "f4a_", "f4b_", "f5i_")}
for k in range(1, 11):
    SCOPE_TABLES[f"ed{k}"] = (f"ed{k}_",) + {1: ("f1", "f4a_"), 2: ("f1",), 3: ("f1",), 4: (), 5: ("f2",), 6: ("f2", "f3"),
                                             7: ("f3",), 8: ("f4",), 9: ("f4", "f5", "flow_"), 10: ("f6",)}[k]


def scope_of(f, line, state):
    """Scope (source-table family) of a line: Results subsection, figure legend, or global."""
    if SCIENCE:                                      # Science panels recombine the source figures: match anywhere
        return "global"
    if f == "results.tex" and line.startswith("\\subsection*"):
        state["i"] += 1
    if f == "results.tex":
        return RESULT_SCOPES[max(state["i"] - 1, 0)] if state["i"] else "global"
    m = re.search(r"\\figlegend\{(Extended Data |Supplementary )?Fig\.~(\d+)\}", line)
    if m:
        state["leg"] = ("ed" if m.group(1) == "Extended Data " else "s" if m.group(1) else "fig") + m.group(2)
    if f.startswith("legends_"):
        return state.get("leg", "global")
    if f == "supplementary.tex":
        if "subsection*{Supplementary Note 10 |" in line:         # the pre-specified flow plan (Note 1 until D134)
            state["note"] = True
        elif "subsection*{Supplementary Note 11 |" in line:
            state["note"] = False
        return "note1" if state.get("note") else "global"
    return "global"


def extract():
    rows = []
    for f in TEXT_FILES:
        state = {"i": 0}
        for i, raw in enumerate((NN / f).read_text(encoding="utf-8").split("\n"), 1):
            scope = scope_of(f, raw, state)
            if raw.lstrip().startswith(DESIGN_FENCE):
                state["design"] = raw.lstrip().startswith(DESIGN_FENCE + "-begin")
                continue
            if raw.lstrip().startswith("%") or "\\newcommand" in raw or "\\usepackage" in raw:
                continue
            s = clean_line(raw)
            for m in NUM.finditer(s):
                sign, body, sci, exp, unit = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or ""
                if s[max(0, m.start() - 1):m.start()] in ("-",) and not sign:
                    sign = "-"
                txt = body.replace(",", "")
                val = float(txt) * (10 ** int(exp.replace("\u2212", "-")) if sci else 1)
                if sign in ("-", "\u2212"):
                    val = -val
                dec = len(txt.split(".")[1]) if "." in txt else 0
                sig = len(txt.replace(".", "").lstrip("0")) if (abs(val) < 1 or sci) else None
                ctx = s[max(0, m.start() - 45):m.end() + 25].strip()
                kind = "check"
                if state.get("design"):
                    kind = "design"
                elif not sci and dec == 0 and abs(val) <= 20 and "%" not in unit:
                    kind = "trivial"
                ptxt = re.search(r"(?:P|q)\s*(?:=|<|<=)\s*$", s[max(0, m.start() - 8):m.start()])
                rows.append(dict(file=f, line=i, scope=scope, number=(sign + body + (sci or "")).strip(), value=val,
                                 decimals=dec, sig=sig, unit=unit.replace("\\", "").replace("~", "").strip(), kind=kind,
                                 is_p=bool(ptxt), context=ctx))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- table values
def table_values():
    """Distinct numeric values per source table (dict name -> array)."""
    vals = {}
    paths = sorted((RESULTS / "r9_figures").glob("*.csv")) + sorted((RESULTS / "r9_flow").glob("*.csv")) + \
        sorted((RESULTS / "r4_resource").glob("*.csv")) + sorted((RESULTS / "r10_parabiosis").glob("*.csv")) + \
        [RESULTS / "key_findings.csv", RESULTS / "r3_conclusions.csv", RESULTS / "r2_audit" / "positive_findings_register.csv"]
    numre = re.compile(r"[+\-\u2212]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")
    n_files = 0
    for p in paths:
        if not p.exists() or SKIP_TABLES.search(p.name) or p.stat().st_size > MAX_BYTES:
            continue
        df = pd.read_csv(p, low_memory=False)
        n_files += 1
        got = []
        for c in df.columns:
            col = df[c]
            if pd.api.types.is_numeric_dtype(col):
                v = pd.to_numeric(col, errors="coerce").dropna().unique()
                got.append(v[np.isfinite(v)])
            elif len(df) <= 60000:
                txt = " ".join(col.dropna().astype(str).unique()[:50000])
                found = [float(x.replace("\u2212", "-")) for x in numre.findall(txt.replace(",", ""))]
                if found:
                    got.append(np.unique(np.array(found, float)))
        if len(df) <= 60000:                                # derived quantities: row count and column sums
            got.append(np.array([len(df)], float))
            num = df.select_dtypes("number")
            if num.shape[1]:
                got.append(num.sum(numeric_only=True).to_numpy(float))
        if got:
            vals[p.name] = np.unique(np.concatenate(got))
    gates = ROOT / "data" / "flow_cochlea" / "gates.csv"   # fixed flow gates (Supplementary Fig. 1 legend)
    if gates.exists():
        g = pd.read_csv(gates, encoding="utf-8-sig")
        vals["flow_gate_thresholds"] = np.unique(pd.to_numeric(g["value"], errors="coerce").dropna().to_numpy(float))
    log(f"indexed {sum(a.size for a in vals.values()):,} values from {n_files} tables")
    return vals


def scope_values(vals, scope):
    pre = SCOPE_TABLES.get(scope)
    if not pre:
        return None
    picked = [a for n, a in vals.items() if n.startswith(pre)]
    return np.unique(np.concatenate(picked)) if picked else None


def sigkey(x, s):
    """|x| rounded to s significant figures, as a hashable float (same arithmetic for table and text values)."""
    x = np.abs(np.asarray(x, float))
    scale = 10.0 ** (np.floor(np.log10(x)) - s + 1)
    return np.round(np.round(x / scale) * scale, 15)


def build_index(v):
    cand = np.unique(np.concatenate([v, 100 * v, np.abs(v), np.abs(100 * v)]))
    cand = cand[np.isfinite(cand) & (np.abs(cand) < 1e9)]
    dec = {d: set(np.round(cand, d).tolist()) for d in range(0, 5)}
    small = np.unique(np.abs(cand[(cand != 0) & (np.abs(cand) < 1)]))
    sig = {s: set(sigkey(small, s).tolist()) for s in range(1, 5)}
    return dec, sig


def matches(r, dec, sig):
    x = r.value
    if r.sig is not None and abs(x) < 1 and x != 0:
        return float(sigkey(x, min(int(r.sig), 4))) in sig[min(int(r.sig), 4)]
    d = min(r.decimals, 4)
    return round(x, d) in dec[d] or round(abs(x), d) in dec[d]


def main():
    t = extract()
    vals = table_values()
    glob_idx = build_index(np.unique(np.concatenate(list(vals.values()))))
    scoped = {}
    t["status"] = np.where(t.kind == "design", "design", "trivial")
    t["match_level"] = ""
    for i, r in t[t.kind == "check"].iterrows():
        if r.scope not in scoped:
            sv = scope_values(vals, r.scope)
            scoped[r.scope] = build_index(sv) if sv is not None else None
        if scoped[r.scope] is not None and matches(r, *scoped[r.scope]):
            t.loc[i, ["status", "match_level"]] = ["matched", "figure tables"]
        elif matches(r, *glob_idx):
            t.loc[i, ["status", "match_level"]] = ["matched" if r.scope == "global" else "matched_elsewhere",
                                                   "all tables"]
        else:
            t.loc[i, "status"] = "unmatched"
    t["p_format"] = t.is_p & (t.sig == 1) & ~t.context.str.contains("<")
    if SIGNOFF.exists():
        so = pd.read_csv(SIGNOFF)
        for s_ in so.itertuples():
            hit = (t.file == s_.file) & (t.number == str(s_.number)) & t.context.str.contains(re.escape(str(s_.context)))
            t.loc[hit & t.status.isin(["unmatched", "matched_elsewhere"]), "status"] = "signed_off"
            t.loc[hit, "note"] = s_.reason
    if "note" not in t:
        t["note"] = ""
    cnt = t.status.value_counts().to_dict()
    log(f"numbers: {len(t)}; " + ", ".join(f"{k} {v_}" for k, v_ in cnt.items()) + f"; P values with one significant "
        f"figure: {int(t.p_format.sum())}")
    for r in t[t.status.isin(["unmatched", "matched_elsewhere"]) | t.p_format].itertuples():
        log(f"  {r.status:9s} {'P1sf ' if r.p_format else ''}{r.file}:{r.line}  {r.number}{r.unit}  | {r.context}")
    write_csv(t.drop(columns=["kind"]), "r9_text", "number_audit_science.csv" if SCIENCE else "number_audit.csv",
              ("Science version (D144): " if SCIENCE else "") +
              "Phase F number audit: every number of the LaTeX manuscript (abstract, main text, legends, Supplementary "
              "Information) with its match against the registered result tables (matched / unmatched / signed_off / "
              "trivial) and the P-value format flag (905)",
              required=["file", "line", "number", "status", "context"])


if __name__ == "__main__":
    main()
