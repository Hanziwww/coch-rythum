"""Round 10 (D144) — Supplementary Materials of the Science version, converted from the approved Nature Immunology
sources (manuscript/ni, ni-v0.13, D144: NI text frozen as the source version). Mechanical conversion only:
  * American spelling (lib.figstyle.american word list, extended for running text);
  * every figure / table / note reference re-targeted: an NI panel goes to the Science main figure that shows it
    (Fig. 1-5, upper-case panels) or else to its full Science-style copy in the Supplementary Materials
    (fig. \\SF{key}; NI Figs 1, 2, 3, 5 and 7, all Extended Data and all Supplementary figures); Supplementary
    Table k -> data Sk; Supplementary Note k -> Supplementary Methods k; Methods -> Materials and Methods;
  * legends to Science captions ("Fig. S#. Title. (A) ...").
Outputs (manuscript/science/, generated - edit the NI source or the rules here, not the outputs):
  sm_methods.tex     Materials and Methods (NI methods.tex without Reporting summary / Data and Code availability)
  sm_supmethods.tex  Supplementary Methods 1-14 (NI Supplementary Notes 1-14)
  sm_text.tex        Supplementary Text (NI Results passages not in the Science main text)
  sm_captions.tex    one page per Supplementary figure (figure + caption), in the order of sm_numbers.tex
  sm_data.tex        captions of Data S1 to S20 (NI Supplementary Tables 1-20)
  sm_conversion_log.csv (results/r10_science via lib.io.write_csv): every reference rewritten (source -> target)
Run: COCH_ROOT=/path/to/coch-rythum python scripts/976_science_convert.py   (then 977 builds and checks)
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import _US_WORDS  # noqa: E402
from lib.io import ROOT, write_csv  # noqa: E402

NI = ROOT / "manuscript" / "ni"
SCI = ROOT / "manuscript" / "science"
LOG = []

# ---------------------------------------------------------------------------------------------- figure maps
ED_KEY = {1: "qc", 2: "markers", 3: "external", 4: "calibration", 5: "clocksED", 6: "programmesED", 7: "heatED",
          8: "blineage", 9: "myeloid", 10: "macohort"}
SI_KEY = {1: "route", 2: "pbtissues", 3: "pbperilymph", 4: "pbnoise", 5: "pbcells", 6: "robust", 7: "gating"}
MAIN_FULL = {1: "atlas", 2: "clocksfull", 3: "heatfull", 5: "marrowfull", 7: "noisefull"}
MAIN_WHOLE = {4: ("main", 2), 6: ("main", 4)}          # NI Figs 4 and 6 have no full copy (Science Figs 2 and 4)
# NI main-figure panel -> (kind, id, panel): 'main' = Science main figure number, 'sm' = Supplementary key
PANEL = {}
for p_, t in zip("abcdefg", [("main", 1, "A"), ("main", 1, "B"), ("sm", "atlas", "C"), ("main", 1, "C"),
                             ("sm", "atlas", "E"), ("sm", "atlas", "F"), ("sm", "atlas", "G")]):
    PANEL[(1, p_)] = t
for p_, t in zip("abcdef", [("sm", "clocksfull", "A"), ("main", 5, "A"), ("sm", "clocksfull", "C"), ("main", 5, "B"),
                            ("sm", "clocksfull", "E"), ("sm", "clocksfull", "F")]):
    PANEL[(2, p_)] = t
for p_, t in zip("abcdefg", [("sm", "heatfull", "A"), ("main", 5, "C"), ("sm", "heatfull", "C"),
                             ("sm", "heatfull", "D"), ("sm", "heatfull", "E"), ("main", 5, "D"), ("sm", "heatfull", "G")]):
    PANEL[(3, p_)] = t
for p_, t in zip("abcdefghi", [("main", 2, "A"), ("main", 2, "B"), ("main", 2, "C"), ("main", 1, "E"), ("main", 2, "E"),
                               ("main", 2, "F"), ("main", 2, "G"), ("main", 2, "H"), ("main", 2, "I")]):
    PANEL[(4, p_)] = t
for p_, t in zip("abcdefg", [("main", 3, "A"), ("main", 3, "B"), ("main", 3, "C"), ("sm", "marrowfull", "D"),
                             ("sm", "marrowfull", "E"), ("sm", "marrowfull", "F"), ("sm", "marrowfull", "G")]):
    PANEL[(5, p_)] = t
for p_ in "abcdefgh":
    PANEL[(6, p_)] = ("main", 4, p_.upper())
for p_, t in zip("abcdefghi", [("sm", "noisefull", "A"), ("sm", "noisefull", "B"), ("sm", "noisefull", "C"),
                               ("sm", "noisefull", "D"), ("main", 5, "E"), ("main", 5, "F"), ("sm", "noisefull", "G"),
                               ("sm", "noisefull", "H"), ("sm", "noisefull", "I")]):
    PANEL[(7, p_)] = t

# Supplementary figure key -> (Science-style figure file, NI legend file, NI legend label)
SF_FILES = {
    "atlas": ("fig1_atlas_v7.pdf", "legends_main.tex", "Fig.~1"),
    "clocksfull": ("fig2_local_clocks_composite_v5.pdf", "legends_main.tex", "Fig.~2"),
    "heatfull": ("fig3_heat_shock_v3.pdf", "legends_main.tex", "Fig.~3"),
    "marrowfull": ("fig5_immune_timetable_v3.pdf", "legends_main.tex", "Fig.~5"),
    "noisefull": ("fig7_independent_cohort_v5.pdf", "legends_main.tex", "Fig.~7"),
    "qc": ("ed1_qc_v2.pdf", "legends_ed.tex", "Extended Data Fig.~1"),
    "markers": ("ed2_markers_v1.pdf", "legends_ed.tex", "Extended Data Fig.~2"),
    "external": ("ed3_external_v1.pdf", "legends_ed.tex", "Extended Data Fig.~3"),
    "calibration": ("ed4_calibration_v1.pdf", "legends_ed.tex", "Extended Data Fig.~4"),
    "clocksED": ("ed5_clocks_v2.pdf", "legends_ed.tex", "Extended Data Fig.~5"),
    "programmesED": ("ed6_programmes_v1.pdf", "legends_ed.tex", "Extended Data Fig.~6"),
    "heatED": ("ed7_heat_shock_v1.pdf", "legends_ed.tex", "Extended Data Fig.~7"),
    "blineage": ("ed8_b_lineage_v1.pdf", "legends_ed.tex", "Extended Data Fig.~8"),
    "myeloid": ("ed9_myeloid_v3.pdf", "legends_ed.tex", "Extended Data Fig.~9"),
    "macohort": ("ed10_ma2025_v1.pdf", "legends_ed.tex", "Extended Data Fig.~10"),
    "route": ("sfig_route_v1.pdf", "legends_si.tex", "Supplementary Fig.~1"),
    "pbtissues": ("sfig_parabiosis_tissues_v1.pdf", "legends_si.tex", "Supplementary Fig.~2"),
    "pbperilymph": ("sfig_parabiosis_perilymph_v1.pdf", "legends_si.tex", "Supplementary Fig.~3"),
    "pbnoise": ("sfig_parabiosis_noise_v1.pdf", "legends_si.tex", "Supplementary Fig.~4"),
    "pbcells": ("sfig_parabiosis_cell_types_v2.pdf", "legends_si.tex", "Supplementary Fig.~5"),
    "robust": ("sfig_robustness_v2.pdf", "legends_si.tex", "Supplementary Fig.~6"),
    "gating": ("sfig1_flow_gating_v2.pdf", "legends_si.tex", "Supplementary Fig.~7")}
# full copies of NI main figures: which of their panels the Science main figures show
FULL_NOTE = {"atlas": "Panels A, B and D are shown in Fig.~1, A to C.",
             "clocksfull": "Panels B and D are shown in Fig.~5, A and B.",
             "heatfull": "Panels B and F are shown in Fig.~5, C and D.",
             "marrowfull": "Panels A to C are shown in Fig.~3.",
             "noisefull": "Panels E and F are shown in Fig.~5, E and F."}

# ---------------------------------------------------------------------------------------------- spelling
US_TEXT = dict(_US_WORDS)
US_TEXT.update({"haematopoietic": "hematopoietic", "haematopoiesis": "hematopoiesis", "haematoxylin": "hematoxylin",
                "anaesthetic": "anesthetic", "anaesthetised": "anesthetized", "leukaemia": "leukemia",
                "oestrogen": "estrogen", "paediatric": "pediatric", "oedema": "edema", "labelling": "labeling",
                "relabelling": "relabeling", "unlabelled": "unlabeled", "labelled": "labeled", "modelled": "modeled",
                "travelled": "traveled", "programmes": "programs", "favoured": "favored", "colour-blind": "color-blind",
                "analyses": "analyses", "analysed": "analyzed", "catalysed": "catalyzed", "dialysed": "dialyzed",
                "learnt": "learned", "amongst": "among", "millilitre": "milliliter", "litres": "liters",
                "sulphate": "sulfate", "neighbouring": "neighboring", "neighbour": "neighbor", "neighbours": "neighbors",
                "behavioural": "behavioral", "centrifuged": "centrifuged", "optimise": "optimize",
                "haemorrhage": "hemorrhage", "anaemia": "anemia", "fibres": "fibers", "colours": "colors",
                "coloured": "colored", "greyscale": "grayscale", "favourable": "favorable", "artefactual": "artifactual"})
_US_TEXT_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in US_TEXT), key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)


def us(s):
    def sub(m):
        w = m.group(0)
        u = US_TEXT[w.lower()]
        return u[0].upper() + u[1:] if w[0].isupper() else u
    return _US_TEXT_RE.sub(sub, s)


# ---------------------------------------------------------------------------------------------- references
PRE = (r"(?P<pre>Extended\s+Data\s+Figs?\.|Supplementary\s+Figs?\.?|Supplementary\s+Tables?|Supplementary\s+Notes?|"
       r"Figs?\.?|Tables?|Notes?)")
BODY = r"(?P<body>\d+[a-z]?(?:(?:--|,)[a-z])*(?:(?:,\s*|\s+and\s+|--)\d+[a-z]?(?:(?:--|,)[a-z])*)*)"
REF = re.compile(PRE + r"(?:~|\s)" + BODY)


def expand_panels(pan):
    out = []
    for part in [p for p in pan.split(",") if p]:
        if "--" in part:
            a, b = part.split("--")
            out += [chr(c) for c in range(ord(a), ord(b) + 1)]
        else:
            out.append(part)
    return out


def parse_body(body):
    items, prev, rng = [], None, False
    for tok in re.findall(r"\d+[a-z]?(?:(?:--|,)[a-z])*|--|,\s*|\s+and\s+", body):
        if tok[0].isdigit():
            num = int(re.match(r"\d+", tok).group())
            if rng and prev is not None:
                items += [(k, []) for k in range(prev + 1, num)]
            rng = False
            items.append((num, expand_panels(tok[len(str(num)):])))
            prev = num
        elif tok == "--":
            rng = True
    return items


def fmt_panels(ps):
    ps = list(dict.fromkeys(ps))
    if len(ps) == 1:
        return ps[0]
    idx = [ord(p) for p in ps]
    if len(ps) >= 3 and idx == list(range(idx[0], idx[0] + len(ps))):
        return f"{ps[0]} to {ps[-1]}"
    if len(ps) == 2:
        return f"{ps[0]} and {ps[1]}"
    return ", ".join(ps[:-1]) + f", and {ps[-1]}"


def targets(pre, items):
    """[(kind, id, [panels])] for one NI reference."""
    pre = " ".join(pre.split())
    out = []
    if pre.startswith("Extended Data"):
        out = [("sm", ED_KEY[n], [p.upper() for p in ps]) for n, ps in items]
    elif pre.startswith("Supplementary Fig"):
        out = [("sm", SI_KEY[n], [p.upper() for p in ps]) for n, ps in items]
    elif pre.startswith("Supplementary Table") or pre.startswith("Table"):
        out = [("data", n, []) for n, _ in items]
    elif pre.startswith("Supplementary Note") or pre.startswith("Note"):
        out = [("note", n, []) for n, _ in items]
    else:                                                # NI main figures
        for n, ps in items:
            if not ps:
                out.append(("sm", MAIN_FULL[n], []) if n in MAIN_FULL else (MAIN_WHOLE[n][0], MAIN_WHOLE[n][1], []))
            else:
                for p in ps:
                    k, i, P = PANEL[(n, p)]
                    out.append((k, i, [P]))
    merged = []
    for k, i, ps in out:                                 # merge consecutive references to the same target
        if merged and merged[-1][0] == k and merged[-1][1] == i:
            merged[-1][2].extend(ps)
        else:
            merged.append((k, i, list(ps)))
    return merged


def fmt_targets(tg):
    parts, j = [], 0
    while j < len(tg):
        k, i, ps = tg[j]
        if k == "data":                                  # data S runs
            run = [i]
            while j + 1 < len(tg) and tg[j + 1][0] == "data":
                j += 1
                run.append(tg[j][1])
            if len(run) == 1:
                parts.append(f"data~S{run[0]}")
            elif run == list(range(run[0], run[-1] + 1)) and len(run) > 2:
                parts.append(f"data~S{run[0]} to S{run[-1]}")
            else:
                parts.append("data~" + " and ".join(f"S{r}" for r in run))
        elif k == "note":
            run = [i]
            while j + 1 < len(tg) and tg[j + 1][0] == "note":
                j += 1
                run.append(tg[j][1])
            parts.append("Supplementary Methods~" + " and ".join(str(r) for r in run))
        elif k == "sm" and not ps:                       # whole Supplementary figures, possibly several
            run = [i]
            while j + 1 < len(tg) and tg[j + 1][0] == "sm" and not tg[j + 1][2]:
                j += 1
                run.append(tg[j][1])
            if len(run) >= 3:                            # a range of Supplementary figures (977 checks contiguity)
                parts.append(f"figs.~\\SF{{{run[0]}}} to \\SF{{{run[-1]}}}")
            else:
                parts.append(("fig.~" if len(run) == 1 else "figs.~") + " and ".join(f"\\SF{{{r}}}" for r in run))
        elif k == "sm":
            parts.append(f"fig.~\\SF{{{i}}}{ps[0]}" if len(ps) == 1 else f"fig.~\\SF{{{i}}}, {fmt_panels(ps)}")
        else:                                            # Science main figure
            if not ps:
                parts.append(f"Fig.~{i}")
            elif len(ps) == 1:
                parts.append(f"Fig.~{i}{ps[0]}")
            else:
                parts.append(f"Fig.~{i}, {fmt_panels(ps)}")
        j += 1
    if len(parts) == 1:
        return parts[0]
    serial = len(parts) > 2 or any(" and " in p or " to " in p or ", " in p for p in parts)
    return ", ".join(parts[:-1]) + (", and " if serial else " and ") + parts[-1]


def convert_refs(s, where):
    def sub(m):
        pre, body = m.group("pre"), m.group("body")
        new = fmt_targets(targets(pre, parse_body(body)))
        LOG.append(dict(file=where, source=m.group(0), target=new))
        return new
    return REF.sub(sub, s)


def convert(s, where, caption=False):
    """One NI LaTeX passage -> Science (comments dropped, references re-targeted, American spelling)."""
    keep = ("% audit:design-begin", "% audit:design-end")    # 905 fences around parameters stay
    s = "\n".join(ln for ln in s.split("\n") if not ln.lstrip().startswith("%") or ln.lstrip().startswith(keep))
    s = "\n".join(ln if ln.lstrip().startswith(keep) else re.sub(r"(?<!\\)%.*", "", ln) for ln in s.split("\n"))
    s = convert_refs(s, where)
    s = re.sub(r"(?<=[^\s~(\[{])\\cite\{", r"~\\cite{", s)     # Science: a space before "(12)"
    s = re.sub(r"\(Methods\)", "(materials and methods)", s)
    s = re.sub(r"(?<!Supplementary )(?<!Materials and )\bMethods\b", "Materials and Methods", s)
    s = s.replace("Supplementary Information", "Supplementary Materials")
    s = s.replace("exact time-relabelling", "exact time-relabeling")
    if caption:
        s = re.sub(r"\\pnl\{(\w)\}--\\pnl\{(\w)\},", lambda m: f"\\textbf{{({m.group(1).upper()} to "
                   f"{m.group(2).upper()})}}", s)
        s = re.sub(r"\\pnl\{(\w)\},", lambda m: f"\\textbf{{({m.group(1).upper()})}}", s)
        s = re.sub(r"\\pnl\{(\w)\}", lambda m: f"\\textbf{{({m.group(1).upper()})}}", s)
        for old, new in [("areas enlarged in d.", "areas enlarged in (D)."), ("replicated in c;", "replicated in (C);"),
                         ("(as in c)", "(as in C)"), ("(encoding as in a)", "(encoding as in A)"),
                         ("View 2, as in a.", "View 2, as in (A)."), ("(display as in a)", "(display as in A)"),
                         ("nucleated cell of c;", "nucleated cell of (C);"),
                         ("(a, b, d--g)", "(A, B and D to G)")]:
            s = s.replace(old, new)
    return us(s)


# ---------------------------------------------------------------------------------------------- sections
def methods():
    t = (NI / "methods.tex").read_text(encoding="utf-8")
    t = t.replace("\\section*{Methods}", "\\section*{Materials and Methods}")
    cut = t.index("\\subsection*{Reporting summary}")
    keep_ai = t[t.index("\\subsection*{Use of artificial-intelligence tools}"):]
    t = t[:cut] + keep_ai
    return convert(t, "methods.tex")


def supmethods():
    t = (NI / "supplementary.tex").read_text(encoding="utf-8")
    t = t[t.index("\\section*{Supplementary Notes}"):t.index("\\section*{Supplementary Tables}")]
    t = t.replace("\\section*{Supplementary Notes}", "\\section*{Supplementary Methods}")
    t = re.sub(r"\\addcontentsline\{toc\}\{section\}\{Supplementary Notes\}", "", t)
    t = re.sub(r"\\subsection\*\{Supplementary Note (\d+) \| ", r"\\subsection*{Supplementary Methods \1: ", t)
    t = t.replace("Bench-level and parameter detail for the Methods and the pre-specified plan of the flow-cytometry "
                  "cohort, numbered in\nthe order in which the Methods cite them.",
                  "Bench-level and parameter detail for the Materials and Methods and the pre-specified plan of the "
                  "flow cytometry\ncohort, numbered in the order in which the Materials and Methods cite them.")
    return convert(t, "supplementary.tex (Notes)")


def paragraphs(sec_title):
    """Paragraphs of one NI Results subsection (by its title)."""
    t = (NI / "results.tex").read_text(encoding="utf-8")
    i0 = t.index("\\subsection*{" + sec_title + "}")
    i0 = t.index("\n", i0) + 1
    i1 = t.find("\\subsection*{", i0)
    body = t[i0:i1 if i1 > 0 else len(t)].strip()
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def suptext():
    parts = ["\\section*{Supplementary Text}",
             "Results that support the main text in more detail, in the order of the main text; numbers as in the "
             "figures and data files."]
    plan = [("Validation of the atlas and of the rhythm calls",
             paragraphs("A paired day--night atlas of the cochlea and its temporal-bone marrow")[1:]),
            ("Structural clocks of the cochlea",
             paragraphs("Cochlear structural cells keep synchronized, light-entrained clocks")),
            ("A systemic heat-shock program rises at night in cochlea and marrow",
             paragraphs("A systemic heat-shock programme rises at night in cochlea and marrow")),
            ("Marrow rhythms follow production and neutrophil state",
             paragraphs("The cochlear immune compartment alternates over the day")[3:]),
            ("The independent noise cohort",
             paragraphs("An independent cohort replicates the rhythms, and the night heat-shock rise is lost after "
                        "noise"))]
    for title, paras in plan:
        assert paras, title
        parts.append(f"\\subsection*{{{title}}}")
        parts += [convert(p, f"results.tex ({title})") for p in paras]
    return "\n\n".join(parts) + "\n"


def legends():
    """NI legend text per label (figure title + body)."""
    out = {}
    for f in ("legends_main.tex", "legends_ed.tex", "legends_si.tex"):
        t = (NI / f).read_text(encoding="utf-8")
        t = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", t, flags=re.S).replace("\\clearpage", "")
        for m in re.finditer(r"\\figlegend\{([^}]*)\}\{([^}]*)\}(.*?)(?=\\figlegend\{|\Z)", t, flags=re.S):
            out[(f, m.group(1))] = (m.group(2).strip(), m.group(3).strip())
    return out


def captions(order):
    lg = legends()
    pages = ["% Supplementary figures of the Science version (generated by 976; order = sm_numbers.tex)",
             "\\newcommand{\\smFig}[3]{\\begin{center}\\includegraphics[width=\\textwidth,height=0.60\\textheight,"
             "keepaspectratio]{#1}\\end{center}\\noindent\\textbf{Fig.~\\SF{#2}. #3}\\ }"]
    for key in order:
        file, src, label = SF_FILES[key]
        title, body = lg[(src, label)]
        title = convert(title, f"{src} {label} (title)", caption=True)
        body = convert(body, f"{src} {label}", caption=True)
        if key in FULL_NOTE:
            body = body.rstrip() + " " + FULL_NOTE[key]
        pages.append(f"\\smFig{{{file}}}{{{key}}}{{{title}}}\n{body}\n\\clearpage")
    return "\n\n".join(pages) + "\n"


def data_captions():
    t = (NI / "supplementary.tex").read_text(encoding="utf-8")
    t = t[t.index("\\begin{description}", t.index("\\section*{Supplementary Tables}")):t.index("\\end{description}") + 17]
    t = re.sub(r"\\item\[Supplementary Tables (\d+)--(\d+)\]", r"\\item[Data S\1 to S\2]", t)
    t = re.sub(r"\\item\[Supplementary Table (\d+)\]", r"\\item[Data S\1]", t)
    t = convert(t, "supplementary.tex (Tables)")
    return ("\\section*{Data S1 to S20}\nProvided as separate files (comma-separated values) with a data dictionary of "
            "every column.\n\n" + t + "\n")


def main():
    order = [k for k in SF_FILES]                        # placeholder order; 977 rewrites sm_numbers.tex and calls
    if (SCI / "sm_order.txt").exists():                  # captions() again in the order of first citation
        order = [k for k in (SCI / "sm_order.txt").read_text(encoding="utf-8").split() if k in SF_FILES]
    out = {"sm_methods.tex": methods(), "sm_supmethods.tex": supmethods(), "sm_text.tex": suptext(),
           "sm_captions.tex": captions(order), "sm_data.tex": data_captions()}
    for name, text in out.items():
        (SCI / name).write_text(f"% GENERATED by scripts/976_science_convert.py from manuscript/ni (D144); do not "
                                f"edit\n{text}", encoding="utf-8", newline="\n")
    log = pd.DataFrame(LOG).drop_duplicates()
    write_csv(log, "r10_science", "sm_conversion_log.csv", "Science version (976, D144): every figure, table and note "
              "reference of the Nature Immunology sources rewritten for the Supplementary Materials (source -> target)",
              required=["file", "source", "target"])
    left = [n for n, t in out.items() if re.search(r"Extended Data|Supplementary (Fig|Table|Note)", t)]
    print("files with NI reference forms left:", left)


if __name__ == "__main__":
    main()
