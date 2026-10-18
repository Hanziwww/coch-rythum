"""Round 10 (D143 plan step 6; D147) — submission package of the Science version (manuscript/science/submission/):
  Manuscript.pdf                main text, references, figure captions and Figs 1-5 (977 build)
  Supplementary_Materials.pdf   Materials and Methods, Supplementary Methods, Supplementary Text, Figs. S1-S22,
                                References (continued numbering), captions of Data S1-S20 (977 build)
  Cover_letter.pdf
  figures/Fig1.pdf ... Fig5.pdf vector figure files (figures/science/sci_fig*, 970)
  Data_S1_to_S20.zip            the Supplementary Tables of the Nature Immunology version as Data S1-S20 (same
                                numbering), with the data dictionary; cross-references inside the files re-targeted to
                                the Science numbering (976 rules: Supplementary Table k -> data Sk, Extended Data /
                                Supplementary figures -> fig. S#, Supplementary Note k -> Supplementary Methods k);
                                cell-type labels and data values unchanged (gold standard, never re-annotated)
  LaTeX_source.zip              sources, coch_nn.bib, the compiled main.bbl and filtered sm.bbl, figure files
  README.md                     contents, open author items, steps before submission (incl. the assumed limits, D145)
Run after 977: COCH_ROOT=/path/to/coch-rythum python scripts/979_science_package.py
Output (lib/io.write_csv): results/r10_science/science_submission_contents.csv (file, role, size, MD5).
"""
import hashlib
import importlib.util
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io import RESULTS, ROOT, log, write_csv  # noqa: E402

SCI = ROOT / "manuscript" / "science"
SUB = SCI / "submission"
FIGS = ROOT / "figures" / "science"
SUPPL = RESULTS / "r9_suppl"
_spec = importlib.util.spec_from_file_location("s976", str(Path(__file__).resolve().parent / "976_science_convert.py"))
S976 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S976)
SOURCES = ["main.tex", "captions.tex", "sm.tex", "sm_methods.tex", "sm_supmethods.tex", "sm_text.tex", "sm_captions.tex",
           "sm_data.tex", "sm_numbers.tex", "sm_mainrefs.tex", "macros.tex", "science.bst", "cover_letter.tex",
           "latexmkrc"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def science_text(s, num):
    """NI cross-references in a data cell -> Science numbering, as plain text."""
    if not isinstance(s, str) or not re.search(r"Supplementary|Extended Data", s):
        return s
    s = S976.convert_refs(s, "data files").replace("~", " ")
    return re.sub(r"\\SF\{(\w+)\}", lambda m: num[m.group(1)], s)


def data_zip(num):
    buf = SUB / "Data_S1_to_S20.zip"
    files = sorted(SUPPL.glob("Supplementary_Table_*.csv"))
    assert len(files) == 20, len(files)
    changed = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            k = int(re.match(r"Supplementary_Table_(\d+)_", p.name).group(1))
            d = pd.read_csv(p, low_memory=False)
            for c in d.columns[d.dtypes == object]:
                new = d[c].map(lambda v: science_text(v, num))
                changed += int((new.fillna("") != d[c].fillna("")).sum())
                d[c] = new
            name = f"Data_S{k}_" + p.stem.split("_", 3)[3] + ".csv"
            z.writestr(f"Data_S1_to_S20/{name}", d.to_csv(index=False, lineterminator="\n"))
        dd = pd.read_csv(SUPPL / "Supplementary_Tables_data_dictionary.csv")
        dd = dd.rename(columns={"supplementary_table": "data_file"})
        dd["data_file"] = "S" + dd["data_file"].astype(str)
        for c in ("column", "description"):
            new = dd[c].map(lambda v: science_text(v, num))
            changed += int((new.fillna("") != dd[c].fillna("")).sum())
            dd[c] = new
        z.writestr("Data_S1_to_S20/Data_S_dictionary.csv", dd.to_csv(index=False, lineterminator="\n"))
    log(f"Data S1-S20: {len(files)} files + dictionary; {changed} text cells re-targeted")
    return changed


def main():
    order = (SCI / "sm_order.txt").read_text(encoding="utf-8").split()
    num = {k: f"S{i}" for i, k in enumerate(order, 1)}
    (SUB / "figures").mkdir(parents=True, exist_ok=True)
    for doc, name in [("main", "Manuscript.pdf"), ("sm", "Supplementary_Materials.pdf"),
                      ("cover_letter", "Cover_letter.pdf")]:
        shutil.copy2(SCI / "build" / f"{doc}.pdf", SUB / name)
    main_tex = (SCI / "main.tex").read_text(encoding="utf-8")
    figs = re.findall(r"\\figpage\{([^}]*)\}\{(\d)\}", main_tex)
    assert len(figs) == 5, figs
    for f, n in figs:
        shutil.copy2(FIGS / f, SUB / "figures" / f"Fig{n}.pdf")
    changed = data_zip(num)
    used = set(re.findall(r"\\smFig\{([^}]*)\}", (SCI / "sm_captions.tex").read_text(encoding="utf-8")))
    with zipfile.ZipFile(SUB / "LaTeX_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in SOURCES:
            z.write(SCI / f, arcname=f"latex_source/{f}")
        z.write(ROOT / "manuscript" / "ni" / "coch_nn.bib", arcname="latex_source/coch_nn.bib")
        for b in ("main.bbl", "sm.bbl"):
            z.write(SCI / "build" / b, arcname=f"latex_source/{b}")
        for f in sorted({f for f, _ in figs} | used):
            z.write(FIGS / f, arcname=f"latex_source/{f}")
    tags = {f: len(re.findall(r"\\authorinput\{", (SCI / f).read_text(encoding="utf-8")))
            for f in SOURCES if f.endswith(".tex") and (SCI / f).exists()}
    tags = {f: n for f, n in tags.items() if n}
    title = " ".join(re.search(r"\{\\Large\\textbf\{(.*?)\}\}", main_tex, flags=re.S).group(1).split())
    title_txt = title.replace("--", chr(8211))
    tag_txt = "\n".join(f"- `{f}`: {n}" for f, n in tags.items())
    readme = f"""# Submission package: Science, Research Article

{title_txt}

## Contents
- `Manuscript.pdf`: title, one-sentence summary, abstract, main text, references 1-35, acknowledgments, figure
  captions and Figs 1-5.
- `Supplementary_Materials.pdf`: Materials and Methods, Supplementary Methods 1-14, Supplementary Text, Figs. S1-S22
  (each with its caption), references continuing the main-text numbering, and the captions of Data S1-S20.
- `figures/`: Figs 1-5 as vector PDFs (180 mm wide).
- `Data_S1_to_S20.zip`: Data S1-S20 as CSV files with a data dictionary (the Nature Immunology Supplementary Tables
  1-20, same numbering; {changed} cross-references inside the files re-targeted to the Science numbering).
- `LaTeX_source.zip`: LaTeX sources (the Supplementary Materials sections are generated from the Nature Immunology
  version by scripts/976), the bibliography with the compiled `main.bbl` and the filtered `sm.bbl`, and the figure
  files.
- `Cover_letter.pdf`: cover letter to the editor.

## Open author items
Text marked `[AUTHOR: ...]` (red in the PDFs) needs the authors' input or confirmation. Open tags per source file:
{tag_txt}

## Before submission
1. Check the Science author instructions (the limits used here were assumed, D145: about 4,500 words including
   references, notes and captions; up to 6 figures; about 40 references; abstract about 125 words; title at most
   96 characters) and adjust if the current instructions differ.
2. Fill in the open author items (shared with the Nature Immunology version: manuscript/ni/author_inputs.md).
3. Choose suggested and excluded reviewers in the cover letter.
4. Submit sequentially: Science first; the Nature Immunology Resource (manuscript/ni/submission) is the fallback.
"""
    (SUB / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    roles = {"Manuscript.pdf": "manuscript", "Supplementary_Materials.pdf": "supplementary materials",
             "Cover_letter.pdf": "cover letter", "Data_S1_to_S20.zip": "data S1-S20", "LaTeX_source.zip":
             "LaTeX source", "README.md": "package readme"}
    rows = []
    for p in sorted(SUB.rglob("*")):
        if p.is_file():
            rel = p.relative_to(SUB).as_posix()
            rows.append(dict(file=rel, role=roles.get(rel) or ("figure" if rel.startswith("figures/") else "other"),
                             bytes=p.stat().st_size, md5=md5(p)))
    cont = pd.DataFrame(rows)
    write_csv(cont, "r10_science", "science_submission_contents.csv", "Science submission package (979, D147): every "
              "file of manuscript/science/submission with its role, size and MD5", required=["file", "role", "md5"])
    log(f"package: {len(cont)} files, {cont.bytes.sum() / 1e6:.1f} MB; open author tags {sum(tags.values())}")


if __name__ == "__main__":
    main()
