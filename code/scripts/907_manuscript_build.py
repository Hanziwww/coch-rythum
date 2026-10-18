"""Round 9 writing, phase F (D99 plan sections 6-7); round 10 (D111) — build and check the LaTeX manuscript of the
active manuscript folder (config paths.manuscript) against the journal limits in config.yaml (journal).

Builds main.tex and supplementary.tex with latexmk and checks:
  words      abstract and main text (Introduction + Results + Discussion) within the journal limits; every main and Extended Data
             legend <= 250 (Word-like count: relation symbols in math count as words); Methods reported
  items      main display items <= 8; Extended Data figures <= 10
  refs       cited keys, uncited library entries, references first cited in the main text vs only in Methods
  order      first citations of main figures and Extended Data figures ascending in the main text; panels of each
             figure first cited in alphabetical order
  lint       register / decision IDs in the text; 'circadian' outside clock genes, the heat-shock programme and cited
             work (listed for review); 'aged' near flow phenotypes; -ise spellings (Oxford -ize house style);
             lower-case p statistics; author tags (\\authorinput) still open
  build      LaTeX errors, undefined citations or references (main text, Supplementary Information, cover letter and
             Reporting Summary draft)
  legends    (D134) no symbol glyphs in legends (key cues in words); each Supplementary Figure on one page with its
             legend; Supplementary Figures, Notes and Tables first cited in sequence (order)
  consistency (D134) facts repeated across Methods, Supplementary Notes, Reporting Summary, legends and cover
             letter: approval number, atlas cells, ages, instruments, antibody identifiers, software versions,
             perfusion
--package (phase G) assembles <manuscript>/submission/: Manuscript.pdf, Supplementary_Information.pdf,
Cover_letter.pdf, Reporting_Summary_draft.pdf, figures/ (main, Extended Data and Supplementary figures as PDF),
Supplementary_Tables.zip (results/r9_suppl), LaTeX_source.zip (sources, coch_nn.bib, the compiled main.bbl and the
figure files, compilable on its own) and README.md (contents, open author tags per file, steps before submission);
source_data/ (909) and code_release/ (904) are listed as they are. The file list with sizes and MD5 is registered as
results/r9_release/submission_contents.csv.
Output (lib/io.write_csv): results/r9_text/manuscript_checks.csv.
--final (D134) also writes <manuscript>/submission/latex_final/main_final.tex (one file: inputs and personal macros
expanded, references pasted from main.bbl, comments and line numbers removed, no figure pages) and compiles it, and
<manuscript>/submission/Inventory_of_Supporting_Information.md (--package does both).
Run (local): COCH_ROOT=/path/to/coch-rythum python scripts/907_manuscript_build.py [--final] [--package]
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import JOURNAL, MANUSCRIPT, RESULTS, ROOT, log, write_csv  # noqa: E402

NN = MANUSCRIPT                                       # active manuscript (config paths.manuscript; round 10: ni)
SUB = NN / "submission"
FIGS = ROOT / "figures" / "figures"
DOCS = ["main", "supplementary", "cover_letter", "reporting_summary"]
TEX_SOURCES = ["main.tex", "abstract.tex", "intro.tex", "results.tex", "discussion.tex", "methods.tex",
               "legends_main.tex", "legends_ed.tex", "legends_si.tex", "figures_pages.tex", "supplementary.tex",
               "macros.tex", "coch_nn.bib", "latexmkrc"]
MAIN = ["intro.tex", "results.tex", "discussion.tex"]
LIMITS = {"abstract": JOURNAL.get("abstract_max_words", 150), "abstract_min": JOURNAL.get("abstract_min_words", 0),
          "main_text": JOURNAL.get("main_text_words", 4500), "legend": JOURNAL.get("legend_words", 250),
          "main_items": JOURNAL.get("main_items", 8), "ed_items": JOURNAL.get("ed_items", 10)}
JNAME = f"{JOURNAL.get('name', 'Nature Neuroscience')} {JOURNAL.get('article_type', 'Article')}"
ISE = re.compile(r"\b\w*(?:normalis|organis|synchronis|characteris|recognis|summaris|randomis|minimis|maximis|optimis|"
                 r"visualis|stabilis|localis|categoris|standardis|emphasis|parameteris|harmonis)"
                 r"(?:e|ed|es|er|ers|ing|ation|ations)\b", re.I)          # -ise verb forms only (not 'organisms')
REGISTER = re.compile(r"\b(?:[AGRBM]\d{2}|D\d{2,3}|I\d{3}|O\d{2})\b")
REL = {"leq": "<=", "geq": ">=", "times": "x", "pm": "+-", "rightarrow": "->", "approx": "~"}
rows = []


def add(check, scope, value, limit, status, detail=""):
    rows.append(dict(check=check, scope=scope, value=value, limit=limit, status=status, detail=detail))


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in s.split("\n"))


def plain(s):
    """LaTeX -> plain words, counted like Word (relation symbols in math count as words; citations do not)."""
    s = strip_comments(s)
    s = re.sub(r"\\(?:cite|citenum|label|ref|input)\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:sub)?section\*\{([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("--", "-").replace("~", " ")
    s = re.sub(r"\\(?:textbf|textit|pnl|mathrm|text|emph)\{([^}]*)\}", r"\1", s)

    def math(m):
        x = re.sub(r"\\(leq|geq|times|pm|rightarrow|approx)\b", lambda k: f" {REL[k.group(1)]} ", m.group(1))
        x = re.sub(r"([<>=])", r" \1 ", x)
        x = re.sub(r"\\[a-zA-Z]+|[{}^_]", "", x)
        return " ".join(x.split()) or "x"
    s = re.sub(r"\$([^$]*)\$", math, s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return re.sub(r"[{}]", "", s)


def words(s):
    return len([w for w in plain(s).split() if re.search(r"[A-Za-z0-9<>=~+x-]", w)])


def legends(path):
    t = strip_comments((NN / path).read_text(encoding="utf-8"))
    for p in re.split(r"\\figlegend\{", t)[1:]:
        yield p.split("}", 1)[0].replace("~", " "), words("\\figlegend{" + p)


def build(doc):
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(x for x in env["PATH"].split(os.pathsep) if "tensorflow_jni" not in x)
    r = subprocess.run(["latexmk", "-g", "-interaction=nonstopmode", "-halt-on-error", f"{doc}.tex"], cwd=NN, env=env,
                       capture_output=True, text=True)
    lg = (NN / "build" / f"{doc}.log").read_text(encoding="utf-8", errors="replace")
    errs = [ln for ln in lg.split("\n") if ln.startswith("!")]
    undef = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", lg)))
    status = "pass" if r.returncode == 0 and not errs and not undef else "fail"
    add("build", doc, r.returncode, 0, status, "; ".join(errs[:3] + [f"undefined {u}" for u in undef[:5]]))
    return lg


def first_citations(text):
    """Ordered first citations of main figures, Extended Data figures and their panels in a LaTeX text."""
    order = {"fig": [], "ed": [], "panels": {}}
    pat = re.compile(r"(Extended\s+Data\s+|Supplementary\s+)?Figs?\.~?((?:\s*(?:,|and)?\s*\d+[a-z]*"
                     r"(?:--[a-z]|,[a-z](?![a-z]))*)+)")
    for m in pat.finditer(text):
        pre = re.sub(r"\s+", " ", m.group(1) or "")
        kind = "ed" if pre == "Extended Data " else ("supp" if pre else "fig")
        if kind == "supp":
            continue
        for item in re.finditer(r"(\d+)([a-z](?:--[a-z]|,[a-z](?![a-z]))*)?", m.group(2)):
            n = int(item.group(1))
            if n not in order[kind]:
                order[kind].append(n)
            if item.group(2) and kind == "fig":
                letters = []
                for part in re.finditer(r"([a-z])(?:--([a-z]))?", item.group(2)):
                    a, b = part.group(1), part.group(2) or part.group(1)
                    letters += [chr(c) for c in range(ord(a), ord(b) + 1)]
                seen = order["panels"].setdefault(n, [])
                seen += [x for x in letters if x not in seen]
    return order


# ---------------------------------------------------------------------------------------------- D134 checks
GLYPHS = re.compile(r"\\diam\b|\\(?:black)?lozenge|\\bullet|(?<!\^\{)\\circ\b|\\(?:big)?triangle\w*|\\(?:black)?square|"
                    r"\\star\b|[\u25A0-\u25FF\u2605\u2606\u2666]")
INSTRUMENTS = ["ZEISS LSM 900", "Olympus FV3000", "NovaSeq 6000", "Chromium X", "MobiNova-100", "FACSCelesta",
               "FACSDiva 9.0"]


def flat(f):
    """A source file without comments, line breaks or ties (for fact matching)."""
    return re.sub(r"\s+", " ", strip_comments((NN / f).read_text(encoding="utf-8")).replace("~", " "))


def legend_glyphs():
    """Legends name their key cues in words (Nature Portfolio: no symbols in legends)."""
    for f in ["legends_main.tex", "legends_ed.tex", "legends_si.tex"]:
        hits = sorted(set(GLYPHS.findall(strip_comments((NN / f).read_text(encoding="utf-8")))))
        add("legends", f"symbol glyphs, {f}", len(hits), 0, "pass" if not hits else "fail", " ".join(hits))


def si_order():
    """First citations of Supplementary Figures, Notes and Tables run 1, 2, 3 ... through the main text and Methods."""
    t = strip_comments("\n".join((NN / f).read_text(encoding="utf-8") for f in ["abstract.tex"] + MAIN + ["methods.tex"]))
    pat = re.compile(r"Supplementary\s+(Fig|Note|Table)s?\.?(?:~|\s)+(\d+)((?:\s*(?:,|and|--|–)\s*\d+)*)")
    seen = {"Fig": [], "Note": [], "Table": []}
    for m in pat.finditer(t):
        nums = [int(m.group(2))] + [int(x) for x in re.findall(r"\d+", m.group(3) or "")]
        if re.search(r"--|–", m.group(3) or ""):
            nums = list(range(nums[0], nums[-1] + 1))
        for n in nums:
            if n not in seen[m.group(1)]:
                seen[m.group(1)].append(n)
    for k, seq in seen.items():
        ok = seq == list(range(1, len(seq) + 1))
        add("order", f"first citation of Supplementary {k}s", " ".join(map(str, seq)), "1, 2, 3 ...",
            "pass" if ok else ("fail" if k != "Table" else "warn"))


def si_pages():
    """Each Supplementary Figure shares one page with its whole legend: the page of legend k holds the figure (a form
    XObject) and the next legend (or the Supplementary Notes heading) starts on the following page."""
    import fitz
    doc = fitz.open(NN / "build" / "supplementary.pdf")
    texts = [pg.get_text() for pg in doc]
    title = {}
    for i, t in enumerate(texts):
        for m in re.finditer(r"Supplementary Fig\.\s*(\d+)\s*\|", t):
            title.setdefault(int(m.group(1)), i)
    notes = next((i for i, t in enumerate(texts) if re.search(r"^Supplementary Notes\s*$", t, re.M) and i > min(title.values())),
                 None)
    for k in sorted(title):
        nxt = title.get(k + 1, notes)
        has_fig = len(doc[title[k]].get_xobjects()) > 0 or len(doc[title[k]].get_images()) > 0
        ok = has_fig and nxt == title[k] + 1
        add("legends", f"Supplementary Fig. {k} on one page", f"page {title[k] + 1}", "figure and legend on one page",
            "pass" if ok else "fail", "" if ok else f"next item on page {None if nxt is None else nxt + 1}")


def consistency():
    """Facts repeated across the Methods, Supplementary Notes, Reporting Summary, legends and cover letter agree."""
    F = {f: flat(f) for f in ["abstract.tex", "methods.tex", "supplementary.tex", "reporting_summary.tex",
                              "cover_letter.tex", "legends_main.tex"]}
    ap = {f: set(re.findall(r"AHGDMU-LAC-B-\d+-\d+", t)) for f, t in F.items()}
    vals = set().union(*ap.values())
    ok = len(vals) == 1 and bool(ap["methods.tex"]) and bool(ap["reporting_summary.tex"])
    add("consistency", "approval number", len(vals), 1, "pass" if ok else "fail", "; ".join(sorted(vals)))
    cells = {f: set(re.findall(r"\b(\d{3},\d{3}) cells\b", F[f])) for f in
             ("abstract.tex", "methods.tex", "cover_letter.tex", "reporting_summary.tex")}
    cells["legends_main.tex (Fig. 1)"] = set(re.findall(r"\b(\d{3},\d{3}) cells\b",
                                                        F["legends_main.tex"].split("figlegend{Fig. 2}")[0]))
    vals = set().union(*cells.values())
    add("consistency", "atlas cells", len(vals), 1, "pass" if len(vals) == 1 else "fail",
        "; ".join(f"{f}: {', '.join(sorted(v))}" for f, v in cells.items() if v))
    age_m = sorted(set(a or b for a, b in re.findall(r"aged (\d+(?:--\d+)?) weeks|(\d+)-week-old", F["methods.tex"])))
    lab = re.search(r"field\{Laboratory animals\}\{(.*?)\\field\{Wild animals\}", F["reporting_summary.tex"])
    age_r = sorted(set(re.findall(r"(\d+(?:--\d+)?) weeks", lab.group(1)))) if lab else []
    add("consistency", "animal ages (weeks)", f"Methods {age_m}; Reporting Summary {age_r}", "equal",
        "pass" if age_m == age_r and age_m else "fail")
    for name in INSTRUMENTS:
        where = [f for f in ("methods.tex", "supplementary.tex", "reporting_summary.tex") if name in F[f]]
        ok = ("reporting_summary.tex" in where) == any(w != "reporting_summary.tex" for w in where)
        add("consistency", f"instrument {name}", ", ".join(where) or "none", "Methods or Notes and Reporting Summary",
            "pass" if ok else "fail")
    rrid = {f: set(re.findall(r"AB\\?_\d+", F[f])) for f in ("supplementary.tex", "reporting_summary.tex")}
    diff = rrid["supplementary.tex"] ^ rrid["reporting_summary.tex"]
    add("consistency", "antibody RRIDs (Supplementary Notes vs Reporting Summary)",
        f"{len(rrid['supplementary.tex'])} vs {len(rrid['reporting_summary.tex'])}", "equal sets",
        "pass" if not diff else "fail", ", ".join(sorted(diff)))
    cats = ["bs-1702R", "60287-1-Ig", "DL-1178-1", "553142"]
    miss = [c for c in cats if not (c in F["reporting_summary.tex"] and (c in F["methods.tex"] or c in F["supplementary.tex"]))]
    add("consistency", "imaging and Fc-block catalogue numbers", len(cats) - len(miss), len(cats),
        "pass" if not miss else "fail", ", ".join(miss))
    ver = re.compile(r"\b([A-Za-z][A-Za-z0-9+-]*) (\d+\.\d+(?:\.\d+)*[a-z]?)\b")
    note14 = F["supplementary.tex"].split("Supplementary Note 14 |")[-1].split("Supplementary Tables")[0]
    rs = F["reporting_summary.tex"]
    n14 = set(ver.findall(note14))
    rsv = set(ver.findall(rs))
    names_rs = {}
    for n, v in rsv:
        names_rs.setdefault(n.lower(), set()).add(v)
    conflict = sorted(f"{n} {v}" for n, v in n14 if n.lower() in names_rs and v not in names_rs[n.lower()])
    add("consistency", "software versions (Supplementary Note 14 vs Reporting Summary)", len(conflict), 0,
        "pass" if not conflict else "review", ", ".join(conflict))
    perf = {f: sorted(set(re.findall(r"(\d+(?:--\d+)?) ml of ice-cold PBS", F[f])))
            for f in ("methods.tex", "supplementary.tex", "reporting_summary.tex")}
    add("consistency", "perfusion volumes (atlas and parabiosis 20 ml; flow cohort 20--25 ml)",
        "; ".join(f"{f}: {v}" for f, v in perf.items() if v), "", "report")


def main():
    texts = {f: (NN / f).read_text(encoding="utf-8") for f in MAIN + ["abstract.tex", "methods.tex"]}
    # ---- words
    na = words(texts["abstract.tex"])
    add("words", "abstract", na, f"{LIMITS['abstract_min']}-{LIMITS['abstract']}",
        "pass" if LIMITS["abstract_min"] <= na <= LIMITS["abstract"] else "fail")
    mt = sum(words(texts[f]) for f in MAIN)
    add("words", "main text", mt, LIMITS["main_text"], "pass" if mt <= LIMITS["main_text"] else "fail",
        "; ".join(f"{f} {words(texts[f])}" for f in MAIN))
    add("words", "methods", words(texts["methods.tex"]), "", "report", "no fixed limit (concise but complete)")
    n_main = n_ed = 0
    for f in ["legends_main.tex", "legends_ed.tex", "legends_si.tex"]:
        for name, n in legends(f):
            lim = LIMITS["legend"] if f != "legends_si.tex" else ""
            add("words", name, n, lim, ("pass" if n <= LIMITS["legend"] else "fail") if lim else "report")
            n_main += f == "legends_main.tex"
            n_ed += f == "legends_ed.tex"
    add("items", "main display items", n_main, LIMITS["main_items"], "pass" if n_main <= LIMITS["main_items"] else "fail")
    add("items", "Extended Data figures", n_ed, LIMITS["ed_items"], "pass" if n_ed <= LIMITS["ed_items"] else "fail")
    # ---- build (also yields the citation list for the reference checks)
    for doc in DOCS:
        build(doc)
    aux = (NN / "build" / "main.aux").read_text(encoding="utf-8", errors="replace")
    cited = []
    for c in re.findall(r"\\citation\{([^}]*)\}", aux):
        cited += [k for k in c.split(",") if k not in cited]
    bib = re.findall(r"^@\w+\{([^,]+),", (NN / "coch_nn.bib").read_text(encoding="utf-8"), re.M)
    uncited = [k for k in bib if k not in cited]
    main_keys = []
    for f in ["abstract.tex"] + MAIN:
        for c in re.findall(r"\\cite\{([^}]*)\}", strip_comments(texts[f])):
            main_keys += [k for k in c.split(",") if k not in main_keys]
    add("refs", "cited in total", len(cited), "", "report", f"library {len(bib)} entries")
    add("refs", "first cited in the main text", len(main_keys), "about 50", "pass" if len(main_keys) <= 60 else "warn")
    add("refs", "uncited library entries", len(uncited), 0, "pass" if not uncited else "warn", ", ".join(uncited))
    # ---- citation order (main text)
    order = first_citations(strip_comments("\n".join(texts[f] for f in MAIN)))
    for kind, lab in (("fig", "main figures"), ("ed", "Extended Data figures")):
        seq = order[kind]
        ok = seq == sorted(seq)
        add("order", f"first citation of {lab}", " ".join(map(str, seq)), "ascending", "pass" if ok else "fail")
    for n, letters in sorted(order["panels"].items()):
        ok = letters == sorted(letters)
        add("order", f"Fig. {n} panels", "".join(letters), "alphabetical", "pass" if ok else "warn")
    # ---- lint
    lint_files = MAIN + ["abstract.tex", "legends_main.tex", "legends_ed.tex", "legends_si.tex", "supplementary.tex",
                         "reporting_summary.tex"]
    for f in lint_files + ["methods.tex", "main.tex", "cover_letter.tex"]:
        t = strip_comments((NN / f).read_text(encoding="utf-8"))
        ids = sorted(set(REGISTER.findall(re.sub(r"\\cite\w*\{[^}]*\}", "", t))))
        add("lint", f"register / decision IDs, {f}", len(ids), 0, "pass" if not ids else "fail", ", ".join(ids))
        ise = sorted(set(w.lower() for w in ISE.findall(t)))
        add("lint", f"-ise spellings, {f}", len(ise), 0, "pass" if not ise else "warn", ", ".join(ise))
        lp = re.findall(r"\$p\s*[<=>]|\bp\s*[<=]\s*0\.|\\leq\s*p\b", t)
        add("lint", f"lower-case p statistics, {f}", len(lp), 0, "pass" if not lp else "warn")
        tags = t.count("\\authorinput{") - t.count("\\newcommand{\\authorinput}")
        if tags:
            add("lint", f"author tags open, {f}", tags, 0, "open")
    for f in lint_files:
        t = plain((NN / f).read_text(encoding="utf-8"))
        for s in re.split(r"(?<=[.;])\s+", t):
            if re.search(r"\bcircadian\b", s, re.I):
                add("lint", f"'circadian' (review), {f}", 1, "", "review", s.strip()[:180])
            if re.search(r"\baged\b", s, re.I) and re.search(r"CXCR4|CD62L|Ly6G|flow|cytometr", s):
                add("lint", f"'aged' near flow phenotypes (review), {f}", 1, "", "review", s.strip()[:180])
    legend_glyphs()
    si_order()
    si_pages()
    consistency()
    if "--final" in sys.argv or "--package" in sys.argv:
        flatten()
        inventory()
    df = pd.DataFrame(rows)
    for st in ("fail", "warn", "open", "review"):
        for r in df[df.status == st].itertuples():
            log(f"{st:6s} {r.check:6s} {r.scope}: {r.value} (limit {r.limit}) {r.detail}")
    log("summary: " + ", ".join(f"{k} {v}" for k, v in df.status.value_counts().items()))
    write_csv(df, "r9_text", "manuscript_checks.csv",
              f"Manuscript checks (907): word counts vs {JNAME} limits, display items, references, "
              "citation order of figures and panels, terminology lint, open author tags and the LaTeX build",
              required=["check", "scope", "value", "limit", "status"])
    if "--package" in sys.argv:
        assert not (df.status == "fail").any(), "fix failing checks before packaging"
        package(df)


# ---------------------------------------------------------------------------------------------- final single-file LaTeX
MACROS = {"figlegend": (2, "\\par\\bigskip\\noindent\\textbf{#1\\,\\textbar\\,#2}\\ "), "pnl": (1, "\\textbf{#1}"),
          "authorinput": (1, "\\textcolor{red}{\\textbf{[AUTHOR: #1]}}"), "um": (0, "\\,\\textmu m"),
          "diam": (0, "$\\blacklozenge$")}


def brace_arg(t, i):
    """The balanced {...} argument starting at t[i] == '{': (content, index after the closing brace)."""
    assert t[i] == "{", t[i:i + 20]
    depth, j = 0, i
    while True:
        if t[j] == "\\":
            j += 2
            continue
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                return t[i + 1:j], j + 1
        j += 1


def expand_macros(t):
    for name, (n, body) in MACROS.items():
        pat = re.compile(r"\\" + name + r"(?![A-Za-z])")
        while True:
            m = pat.search(t)
            if not m:
                break
            j, args = m.end(), []
            for _ in range(n):
                while t[j] in " \n":
                    j += 1
                a, j = brace_arg(t, j)
                args.append(expand_macros(a))
            out = body
            for k, a in enumerate(args, 1):
                out = out.replace(f"#{k}", a)
            t = t[:m.start()] + out + t[j:]
    return t


def strip_all_comments(t):
    out = []
    for ln in t.split("\n"):
        if ln.lstrip().startswith("%"):
            continue
        out.append(re.sub(r"(?<!\\)%.*", "", ln).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def flatten():
    """submission/latex_final/main_final.tex: one file, standard packages, no personal macros, references pasted."""
    t = (NN / "main.tex").read_text(encoding="utf-8")
    t = t.replace("\\input{macros}\n", "")
    t = t.replace("\\usepackage[mathlines]{lineno}\n", "").replace("\\linenumbers\n", "")
    t = re.sub(r"% ---- Figure pages.*?\\input\{figures_pages\}\n", "", t, flags=re.S)
    for _ in range(3):
        t = re.sub(r"\\input\{([^}]+)\}", lambda m: (NN / (m.group(1) + ".tex")).read_text(encoding="utf-8"), t)
    bbl = (NN / "build" / "main.bbl").read_text(encoding="utf-8")
    t = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}", lambda m: bbl, t)
    t = expand_macros(strip_all_comments(t))
    assert "\\input{" not in t and "\\bibliography{" not in t, "flatten incomplete"
    for name in MACROS:
        assert not re.search(r"\\" + name + r"(?![A-Za-z])", t), name
    out = SUB / "latex_final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_final.tex").write_text(t, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(x for x in env["PATH"].split(os.pathsep) if "tensorflow_jni" not in x)
    ok = True
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main_final.tex"], cwd=out,
                           env=env, capture_output=True, text=True)
        ok &= r.returncode == 0
    lg = (out / "main_final.log").read_text(encoding="utf-8", errors="replace")
    undef = re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", lg)
    add("final", "single-file LaTeX (main_final.tex)", "compiled" if ok else "failed", "compiles, no undefined",
        "pass" if ok and not undef else "fail", "; ".join(sorted(set(undef))[:5]))
    log(f"final LaTeX: {out / 'main_final.tex'} ({len(t) / 1e3:.0f} kB), build {'ok' if ok else 'FAILED'}")


def plain_title(s):
    s = re.sub(r"\$\^\{([+\-])\}\$", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s).replace("$\\mu$", "µ").replace("\\%", "%")
    s = s.replace("--", "\u2013").replace("~", " ").replace("$", "")
    return re.sub(r"\s+", " ", s).strip()


def inventory():
    """submission/Inventory_of_Supporting_Information.md: every Extended Data and Supplementary item with its title."""
    rows = []
    for f, kind in (("legends_ed.tex", "Extended Data Fig."), ("legends_si.tex", "Supplementary Fig.")):
        t = strip_comments((NN / f).read_text(encoding="utf-8"))
        for m in re.finditer(r"\\figlegend\{([^}]*)\}\{", t):
            title, _ = brace_arg(t, m.end() - 1)
            rows.append((kind, m.group(1).replace("~", " ").split()[-1], plain_title(title)))
    sup = strip_comments((NN / "supplementary.tex").read_text(encoding="utf-8"))
    for m in re.finditer(r"\\subsection\*\{Supplementary Note (\d+) \| ([^}]*)\}", sup):
        rows.append(("Supplementary Note", m.group(1), plain_title(m.group(2))))
    pat = r"\\item\[Supplementary Tables? ([0-9]+(?:--[0-9]+)?)\]\s*(.*?)(?=\\item|\\end\{description\})"
    for m in re.finditer(pat, sup, flags=re.S):
        rows.append(("Supplementary Table", m.group(1).replace("--", "\u2013"), plain_title(m.group(2))))
    title = re.search(r"\\title\{([^}]*)\}", (NN / "main.tex").read_text(encoding="utf-8")).group(1)
    n_si = sum(1 for r in rows if r[0] == "Supplementary Fig.")
    files = {"Extended Data Fig.": "figures/ED_Fig{n}.pdf", "Supplementary Fig.": "Supplementary_Information.pdf",
             "Supplementary Note": "Supplementary_Information.pdf", "Supplementary Table": "Supplementary_Tables.zip"}
    lines = ["# Inventory of Supporting Information", "", f"{JNAME}: {plain_title(title)}", "",
             "| Item | Number | Title or content | File |", "|---|---|---|---|"]
    lines += [f"| {k} | {n} | {t_} | {files[k].format(n=n)} |" for k, n, t_ in rows]
    lines += [f"| Source Data | Fig. 1\u20137, Extended Data Fig. 1\u201310, Supplementary Fig. 1\u2013{n_si} | statistics "
              "and image panels per figure | source_data/Source_Data_<figure>.xlsx and .zip |", ""]
    (SUB / "Inventory_of_Supporting_Information.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    add("final", "inventory of supporting information", len(rows), "", "report")

# ---------------------------------------------------------------------------------------------- submission package
def figure_files():
    """(file in figures/figures, submission name) for the main, Extended Data and Supplementary figures, in order."""
    out = []
    for f, lab in re.findall(r"\\figpage\{([^}]+)\}\{([^}]+)\}", (NN / "figures_pages.tex").read_text(encoding="utf-8")):
        num = lab.replace("~", " ").split()[-1]
        out.append((f, f"ED_Fig{num}" if lab.startswith("Extended") else f"Fig{num}"))
    sup = (NN / "supplementary.tex").read_text(encoding="utf-8")          # each SI figure sits with its legend in
    sup = sup.replace("\\input{legends_si}", (NN / "legends_si.tex").read_text(encoding="utf-8"))   # legends_si (D133)
    si = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", strip_comments(sup))
    out += [(f, f"Supplementary_Fig{i}") for i, f in enumerate(si, 1)]
    return out


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(checks):
    figs = figure_files()
    (SUB / "figures").mkdir(parents=True, exist_ok=True)
    pdfs = [("main", "Manuscript.pdf"), ("supplementary", "Supplementary_Information.pdf"),
            ("cover_letter", "Cover_letter.pdf"), ("reporting_summary", "Reporting_Summary_draft.pdf")]
    for doc, name in pdfs:
        shutil.copy2(NN / "build" / f"{doc}.pdf", SUB / name)
    for f, name in figs:
        shutil.copy2(FIGS / f, SUB / "figures" / f"{name}.pdf")
    with zipfile.ZipFile(SUB / "Supplementary_Tables.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((RESULTS / "r9_suppl").glob("*.csv")):
            z.write(p, arcname=f"Supplementary_Tables/{p.name}")
    with zipfile.ZipFile(SUB / "LaTeX_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in TEX_SOURCES:
            z.write(NN / f, arcname=f"latex_source/{f}")
        z.write(NN / "build" / "main.bbl", arcname="latex_source/main.bbl")
        for f, _ in figs:
            z.write(FIGS / f, arcname=f"latex_source/{f}")
    tags = checks[checks.scope.str.startswith("author tags open")]
    tag_lines = "\n".join(f"- `{r.scope.split(', ', 1)[1]}`: {r.value}" for r in tags.itertuples())
    n_sd = len(list((SUB / "source_data").glob("*.zip")))
    n_si = sum(1 for _, name in figs if name.startswith("Supplementary_Fig"))      # D143: 7 since the renumbering
    n_sm = len(re.findall(r"\\subsection\*\{Supplementary Note \d+ ", (NN / "supplementary.tex").read_text(encoding="utf-8")))
    m_title = re.search(r"\\title\{([^{}]+)\}", (NN / "main.tex").read_text(encoding="utf-8"))
    assert m_title, "no \\title{} in main.tex"
    title = m_title.group(1).replace("---", "\u2014").replace("--", "\u2013")   # current title, LaTeX dashes as text
    readme = f"""# Submission package: {JNAME}

{title}

## Contents
- `Manuscript.pdf`: title, abstract, main text, Methods, references, figure and Extended Data legends, and all
  figures at final size (for review; the journal also receives the figure files below).
- `Supplementary_Information.pdf`: Supplementary Figs 1-{n_si}, Supplementary Notes 1-{n_sm} and the list of
  Supplementary Tables.
- `Supplementary_Tables.zip`: Supplementary Tables 1-20 as CSV files with a data dictionary of every column.
- `figures/`: Fig. 1-7, Extended Data Fig. 1-10 and Supplementary Figs 1-{n_si} as vector PDFs (180 mm wide).
- `source_data/`: Source Data for every figure: one Excel workbook per figure (an index sheet and one sheet per
  panel table) and {n_sd} zip files with the same tables as CSV, a panel-to-file index and the image
  panels as displayed.
- `LaTeX_source.zip`: LaTeX sources, the bibliography (`coch_nn.bib` and the compiled `main.bbl`) and the figure
  files; compiles on its own with `latexmk main.tex` or pdflatex.
- `Cover_letter.pdf`: cover letter to the editor.
- `Reporting_Summary_draft.pdf`: draft answers for the Nature Portfolio Reporting Summary form; transfer them into the
  journal's form (not for upload).
- `code_release/`: code archive for editors and reviewers (README, all scripts, software versions).

## Open author items
Text marked `[AUTHOR: ...]` (red in the PDFs) needs the authors' input or confirmation. Open tags per source file:
{tag_lines}

## Before submission
1. Fill in the open author items (`{NN.relative_to(ROOT).as_posix()}/author_inputs.md` lists them, with where each
   answer goes).
2. Publish the code (GitHub) with an archived release (Zenodo DOI) and add both to the Code availability statement and
   `code_release/README.md`.
3. Choose suggested and excluded reviewers in the cover letter.
4. Fill in the Reporting Summary form from `Reporting_Summary_draft.pdf`.

## During review and at acceptance
5. Obtain the raw reads of the atlas (FASTQ or BAM) from the sequencing providers and deposit them with the processed
   count matrices privately in GEO; give editors and reviewers the access token if they ask (the Data availability
   statement says the data are available to them on request).
6. At acceptance, add the GEO accession to the Data availability statement, the Reporting Summary and
   `code_release/README.md`.
"""
    (SUB / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    roles = {"Manuscript.pdf": "manuscript", "Supplementary_Information.pdf": "supplementary information",
             "Cover_letter.pdf": "cover letter", "Reporting_Summary_draft.pdf": "reporting summary draft (authors)",
             "Supplementary_Tables.zip": "supplementary tables", "LaTeX_source.zip": "LaTeX source",
             "README.md": "package readme"}
    rows = []
    for p in sorted(SUB.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(SUB).as_posix()
        role = roles.get(rel) or {"figures": "figure", "source_data": "source data",
                                  "code_release": "code release"}.get(rel.split("/")[0], "other")
        rows.append(dict(file=rel, role=role, bytes=p.stat().st_size, md5=md5(p)))
    cont = pd.DataFrame(rows)
    write_csv(cont, "r9_release", "submission_contents.csv", "Submission package (907 --package): every file of "
              f"{NN.relative_to(ROOT).as_posix()}/submission with its role, size and MD5",
              required=["file", "role", "bytes", "md5"])
    log(f"package: {len(cont)} files, {cont.bytes.sum() / 1e6:.1f} MB; figures {len(figs)}; source data zips {n_sd}")


if __name__ == "__main__":
    main()
