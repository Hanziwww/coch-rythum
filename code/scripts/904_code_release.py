"""Public release of the study (round 10, phase G; D134) and the code index of the submission.

Assembles <out>/coch-rythum_v<version>/ and its upload files in <out>/upload/ (default out: <project>/release). Every part
is read from the project at run time, so the final run picks up the final manuscript, figures and tables.

Release layout
  README.md, LICENSE-code (Apache-2.0, as stated in the Methods), LICENSE-data (CC BY 4.0), CITATION.cff, MANIFEST.tsv
  (path, size_bytes, md5, description) and MD5SUMS.txt (md5sum -c format)
  code/scripts        all analysis scripts (Python, R, shell) with lib/, setup/, resources/ and config.yaml, rewritten by
                      lib/release.Sanitizer (server, environment and local paths -> COCH_ROOT, COCH_ENVS or paths relative
                      to the project root; proxy lines removed; hosts and names redacted); config.yaml points to the
                      release layout
  code/env            server and local software versions with conda, pip and R specifications derived from them
  code/docs           pipeline.md (module -> script -> outputs from results_manifest.csv; figure -> panels -> tables from
                      panel_index.csv) and scripts_index.csv
  code/prepare_workspace.py  links scripts/ -> code/scripts (scripts run from the project root) and extracts the image
                      and flow archives to the folders the imaging and flow scripts read
  data/processed      coca_circadian.h5ad (--h5ad, or from --server; MD5 equal to the source) with a schema README read
                      from the file
  data/flow           the flow export (data/flow_cochlea) and its original archive, with a README
  data/imaging        the original image archives, a README (instruments, channels, pixel sizes, naming, use in the
                      figures) and figure_image_sources.csv (figure panel -> archive member; MD5 checked where recorded)
  data/external       accessions.csv (the public-datasets Supplementary Table, further public data downloaded by the code, reference
                      resources) and fetch_public_data.sh: public data are referenced, never copied
  data/raw_reads      only with --fastq PATH (FASTQ / BAM); otherwise the README carries the GEO accession placeholder
  results             every analysis table, results_manifest.csv (MD5 of redacted copies updated), README and GLOSSARY
                      (identifiers that appear in the tables)
  supplementary       Supplementary Tables (908) with their data dictionary and Source Data (909: zip and xlsx)
  figures             the figure files named in figures_pages.tex and legends_si.tex (pdf, svg, jpg, png), panels/
                      (image tiles as displayed) and the Fig. 1a illustration
  manuscript          LaTeX sources (comments removed except the 905 audit fences; graphicspath -> figures/), the
                      figure PDFs they include, the compiled PDFs (main text, Supplementary Information, Reporting
                      Summary) with main.bbl, and submission/latex_final/ when present (its .tex, .bbl, PDF and
                      image files; no LaTeX build by-products such as .log or .aux)
Not released: memory/, cache/, logs/, CLAUDE.md, author_inputs.md, outline.tex and cover_letter.tex (internal plan and
correspondence with the editor), manuscript/nn, figures/qc, figures/drafts, figures/requests, superseded figure versions,
data/coca_main_atlas_with_obs.h5ad (30 libraries outside this study), copies of public data (data/CRA0198*), extracted
copies of archives (data/photos/*), scripts/dev, 95_memory_update.py (internal log), caches and _-prefixed scratch files,
and results/r4_resource/S_external_datasets.csv (a copy of the internal datasets log; data/external/accessions.csv
replaces it). Raw reads are not in the project; the authors deposit them in GEO / SRA.
Rules
  - text is written UTF-8 with LF endings; the release folder is generated (files a run does not place are removed) and
    unchanged large files are not copied again
  - README, docs and manifest descriptions carry no internal log identifiers and no local or server paths
    (release.public_text, then a scan)
  - every released text file is scanned for server, home and drive paths, the ssh host, cloud-provider names, the
    user's name, e-mail addresses (except --contact), credentials and key material, binary files for the same literals;
    the original data archives are not opened. Result tables with hits are released redacted (release_redactions.csv)
  - Supplementary Tables and Source Data (CSV in the zips, xlsx sheets) pass lib/pubclean.residual()
  - released Python files compile and shell scripts pass bash -n; with --latex-check the released LaTeX sources and the
    project sources are compiled in throwaway copies (<out>/_latex_check, removed afterwards) and must typeset the same
    text without errors or undefined references
  - MANIFEST.tsv and MD5SUMS.txt come from one hashing pass and are verified by a second; every upload file is at most
    --limit-gb (default 10, decimal GB); larger files are split into byte parts (.001, .002; joined by cat or copy /b)
    whose joined MD5 is checked; zip archives are checked against their files (CRC too with --crc)
Figures are named as published: main and Extended Data figures from figures_pages.tex, Supplementary figures from the
\\figlegend labels of legends_si.tex; panel_index.csv keys of Supplementary figures are internal IDs and are mapped to
the published numbers with SI_NUMBER of 909 (read without running 909); rows of retired figures are ignored.
Outputs (results/r9_release/): scripts_index.csv and local_software_versions.csv (as before), release_files.csv,
release_checks.csv, release_scan.csv, release_pubclean.csv, release_redactions.csv, release_archives.csv and
release_code_changes.csv (every line the sanitiser changed; the release_* tables describe the build and are not copied
into the release). --stage-submission also stages <manuscript>/submission/code_release/: README.md (written here from
the current scripts, groups and figures), the scripts index, local and server software versions and
coch_rythum_code.zip (sanitised scripts, env/ specifications, LICENSE, README).
Run (local): COCH_ROOT=/path/to/coch-rythum python scripts/904_code_release.py --server HOST [--h5ad SRC] [--fastq PATH]
  [--version 1.0] [--out DIR] [--limit-gb 10] [--skip-archives] [--crc] [--latex-check] [--geo ACC] [--doi DOI]
  [--github URL] [--article TEXT] [--authors "Family, Given; ..."] [--contact "name, e-mail"] [--stage-submission]
"""
import argparse
import difflib
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import release as rl  # noqa: E402
from lib.io import CFG, JOURNAL, LIGHTS_ON, MANUSCRIPT, RESULTS, ROOT, log, sample_table, write_csv  # noqa: E402
from lib.pubclean import SUBS, residual  # noqa: E402

S = ROOT / "scripts"
STAGE = MANUSCRIPT / "submission" / "code_release"
NAME = "coch-rythum"
MOD = "r9_release"
LOCAL_PKGS = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "scikit-image", "tifffile", "Pillow",
              "PyMuPDF", "PyYAML", "openpyxl", "h5py", "czifile", "imagecodecs", "scikit-learn", "torch",
              "py7zr"]   # distributions; PyMuPDF reads the figure PDFs (906), czifile / imagecodecs the CZI images
#                          (934, 959), scikit-learn 942, torch 959, py7zr the archive inventory of 950
GROUPS = [(r"^0\d_|^1\d_", "setup and data audit"), (r"^2\d[a-z]?_", "dimension reduction and embeddings"),
          (r"^3\d[a-z]?_", "pseudobulk and gene rhythms"), (r"^4\d[a-z]?_", "pathways, enrichment and motifs"),
          (r"^5\d[a-z]?_", "composition and neighbourhoods"), (r"^6\d[a-z]?_", "migration axes and communication"),
          (r"^7\d[a-z]?_", "external circadian data"), (r"^8\d_", "resource and applied modules"),
          (r"^9[5-9]_", "bookkeeping (manifest, key findings, plan status)"),
          (r"^2\d\d[a-z]?_", "validation (time axis, external validation, audit, evidence register)"),
          (r"^3\d\d[a-z]?_", "external single-cell, spatial and noise data"),
          (r"^4\d\d[a-z]?_", "design benchmark, attribution, drivers, cross-species, atlas concordance"),
          (r"^5\d\d_", "B-lineage hypotheses"), (r"^6\d\d_", "myeloid hypotheses and neutrophil timer"),
          (r"^7\d\d_", "independent noise cohort (exploratory)"),
          (r"^90\d_", "figure data, figure QC, manuscript, tables and release"), (r"^91\d_", "main figures"),
          (r"^92\d_", "Extended Data figures"), (r"^93\d_", "imaging and flow cytometry"),
          (r"^9[45]\d_", "image verification, robustness, 3D image inventory and Supplementary imaging figures"),
          (r"^96\d_", "parabiosis cell counts with the upgraded rules"),
          (r"^97\d_", "Science version: figures, Supplementary Materials, manuscript build and package")]
EXCLUDE_SCRIPTS = {"95_memory_update.py"}             # maintains the internal project log (memory/, not released)
SKIP_DIRS = {"__pycache__", "dev"}
EXCLUDE_TEX = {"outline.tex", "cover_letter.tex"}
LATEX_FINAL_KEEP = {".tex", ".bbl", ".bib", ".bst", ".cls", ".sty", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".tif"}
BUILD_DOCS = {"main": "Manuscript (main text, Methods, references, legends and figure pages)",
              "supplementary": "Supplementary Information", "reporting_summary": "Reporting Summary (draft answers)"}
FIG_EXT = ["pdf", "svg", "jpg", "png"]
PLATFORMS = {"10x3'-V4": "10x Genomics Chromium 3' v4", "mobiseq-V2": "MobiSeq v2"}
FLOW_DIR, FLOW_ZIP = ROOT / "data" / "flow_cochlea", ROOT / "data" / "Cochlear_Flow_Plotting_Only.zip"
FIG1A = ROOT / "data" / "photos" / "fig1a_workflow" / "fig1a_workflow_pasted.png"
IMAGE_ARCHIVES = [   # (source in the project, description)
    (ROOT / "data" / "Parabiosis_Mice_ALL.zip",
     "Parabiosis survey: confocal acquisitions (ZEISS CZI, Olympus OIR), exported TIFF planes and acquisition reports of "
     "wild-type mice joined to Lyz2-EGFP partners (5 to 54 days); INDEX.md inside lists every file"),
    (ROOT / "data" / "photos" / "FUT4-Neutrophils-English.zip",
     "FUT4 / CD45 / DAPI images of marrow, perilymph and endolymph (wild type) and 54-day parabiont sections (Olympus "
     "FV3000, 20x and 60x); README-English.txt inside"),
    (ROOT / "data" / "photos" / "cochlea_3d.zip",
     "Rendered exports (8-bit PNG) of the 3D reconstructions of the cochlea and its marrow, with manual vessel "
     "annotations"),
    (ROOT / "data" / "Brain_spleen.zip", "Orthogonal projections of brain and liver of a 31-day wild-type parabiont")]
WORKSPACE_MAP = [    # (folder the scripts read, archive in the release, folder inside the archive)
    ("data/photos/FUT4_WT_2026-10-12/", "data/imaging/FUT4-Neutrophils-English.zip",
     "FUT4-Neutrophils/WT-Celltype/2026.10.12-WT-Celltype-FUT4/"),
    ("data/photos/parabiosis_54d/", "data/imaging/FUT4-Neutrophils-English.zip",
     "FUT4-Neutrophils/Parabiotic_Mouse_Section_Staining-Immune_Cell_Types/"),
    ("data/photos/cochlea_3d/", "data/imaging/cochlea_3d.zip", ""),
    ("data/photos/parabiosis_all/Parabiosis_Mice/", "data/imaging/Parabiosis_Mice_ALL.zip", "Parabiosis_Mice/")]
UPLOAD_GROUPS = [("core", ""), ("results", "results/"), ("atlas_h5ad", "data/processed/"), ("flow", "data/flow/"),
                 ("imaging", "data/imaging/"), ("raw_reads", "data/raw_reads/")]
RELEASE_TABLES = ["release_files.csv", "release_checks.csv", "release_scan.csv", "release_pubclean.csv",
                  "release_redactions.csv", "release_archives.csv", "release_code_changes.csv"]
RESULTS_EXCLUDE = {   # result tables that are copies of internal notes (path -> what replaces them in the release)
    "results/r4_resource/S_external_datasets.csv": "data/external/accessions.csv"}
OBS_DOC = [   # (column or prefix*, description)
    ("batch_id", "sample (library); one pool of three mice per tissue and time point"),
    ("condition", "experimental condition (a single level: untreated)"),
    ("treat_time", "not used (no levels)"),
    ("tissue", "cochlea, or bone marrow: the temporal-bone marrow next to the cochlea"),
    ("platform", "library chemistry (10x Genomics Chromium 3' v4 or MobiSeq v2)"),
    ("day_time", "wall-clock hour of collection; ZT = (day_time - 8) mod 24 (lights on at 08:00, 12 h light : 12 h dark)"),
    ("log1p_*", "natural log of 1 + the metric named after the prefix"),
    ("n_genes_by_counts", "genes with at least one count (Scanpy quality-control metrics)"),
    ("total_counts_mt", "counts of mitochondrially encoded genes (Scanpy)"),
    ("total_counts_ribo", "counts of ribosomal-protein genes (Scanpy)"),
    ("total_counts_hb", "counts of haemoglobin genes (Scanpy)"),
    ("total_counts", "UMI counts (Scanpy)"),
    ("pct_counts_in_top_*", "share of counts in the most highly expressed genes (Scanpy)"),
    ("pct_counts_mt", "mitochondrial share of counts, % (Scanpy)"),
    ("pct_counts_ribo", "ribosomal-protein share of counts, % (Scanpy)"),
    ("pct_counts_hb", "haemoglobin share of counts, % (Scanpy)"),
    ("nUMIs", "UMIs (OmicVerse quality control)"), ("mito_perc", "mitochondrial share (OmicVerse quality control)"),
    ("detected_genes", "detected genes (OmicVerse quality control)"),
    ("cell_complexity", "cell complexity (OmicVerse quality control)"),
    ("passing_*", "quality-control flag (OmicVerse thresholds; all cells passed)"),
    ("n_genes", "detected genes"),
    ("doublet_score", "Scrublet doublet score"), ("predicted_doublet", "Scrublet doublet call"),
    ("_scvi_*", "codes registered by scVI for the integration model (batch, labels)"),
    ("is_landmark", "landmark-based clustering: cell chosen as a landmark (weighted k-medoids)"),
    ("landmark_id", "landmark-based clustering: most probable landmark"),
    ("landmark_mem_top*", "landmark-based clustering: largest and second-largest landmark membership"),
    ("landmark_mem_margin", "landmark-based clustering: difference of the two largest memberships"),
    ("landmark_mem_entropy", "landmark-based clustering: entropy of the memberships"),
    ("landmark_mem_*", "landmark-based clustering: membership in landmark cluster k (random walk with restart)"),
    ("landmark_blend", "landmark-based clustering: display colour (hex code) blended from the landmark memberships"),
    ("cell_type", "cell subtype: reference annotation of the atlas, used without re-annotation"),
    ("cell_class", "cell class"), ("cell_type_abbr", "abbreviation of cell_type used in the figures")]
EXTRA_DATASETS = {   # public data downloaded by the code beyond the public-datasets Supplementary Table (not in the article)
    "GSE192616": ("mouse", "skull and long-bone marrow", "scRNA-seq", "skull-marrow reference for label transfer"),
    "GSE312476": ("mouse", "cochlea", "bulk RNA-seq", "noise time course, pilot of the noise analyses"),
    "GSE244055": ("human", "adult utricle", "scRNA-seq", "human inner-ear expression check"),
    "GSE175635": ("mouse", "intestinal IgA+ plasma cells", "bulk RNA-seq", "time-of-day reference, B-lineage hypotheses"),
    "GSE87467": ("mouse", "bone marrow (clock-gene knockout)", "expression profiling", "B-lineage hypotheses"),
    "GSE200145": ("mouse", "neutrophils of several tissues", "scRNA-seq",
                  "count matrix lacks sample tags; the same experiment was analysed from GSE266518"),
    "GPL6246": ("mouse", "", "microarray platform annotation", "probe annotation of the ImmGen and macrophage arrays")}
RESOURCE_SECTIONS = {
    "genesets": "MSigDB mouse gene-set collections (hallmark, Reactome, GO biological process), release 2024.1.Mm",
    "regulons": "CollecTRI transcription-factor regulons (OmniPath) and MGI mouse-human orthology",
    "annotation": "PROGENy mouse pathway model and Ensembl GRCm39 release 113 gene annotation",
    "cistarget": "cisTarget mm10 motif rankings and motif annotations",
    "nichenet": "NicheNet v2 mouse ligand-receptor network and ligand-target matrix",
    "gwas": "GWAS Catalog associations",
    "promoters": "UCSC mm10 RefSeq genes and 1-kb upstream sequences (promoter motifs)"}
PH = {"authors": "[AUTHOR: author list]", "article": "[AUTHOR: citation of the article and its DOI once published]",
      "geo": "[AUTHOR: GEO accession of the raw reads]", "doi": "[AUTHOR: DOI of this release]",
      "github": "[AUTHOR: code repository URL]", "contact": "[AUTHOR: corresponding author and e-mail]"}


# ------------------------------------------------------------------------------------------------ code index (904)
ABBREV = {"Fig", "Figs", "e.g", "i.e", "vs", "al", "cf", "No", "approx", "Suppl", "ref", "refs", "resp", "min", "max"}


def header(p: Path) -> str:
    """First sentence of the first paragraph of a script's docstring or leading comment block."""
    t = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".py":
        m = re.search(r'^\s*(?:#[^\n]*\n\s*)*[rRuU]?"""(.*?)"""', t, re.S)
        para = m.group(1).strip().split("\n\n")[0] if m else ""
    else:
        lines = []
        for ln in t.split("\n"):
            if ln.startswith("#!"):
                continue
            if not ln.startswith("#"):
                break
            lines.append(ln.lstrip("#").strip())
        para = " ".join(lines)
    s = re.sub(r"\s+", " ", para).strip()
    for m in re.finditer(r"\.(?=\s+[A-Z(]|\s*$)", s):
        head, word = s[:m.start()], re.search(r"([\w.]+)$", s[:m.start()])
        if m.start() >= 40 and head.count("(") == head.count(")") and not (word and word.group(1) in ABBREV):
            return s[:m.start() + 1]
    return s[:300]


def group_of(name: str) -> str:
    hit = ""
    for pat, g in GROUPS:                       # later (more specific, three-digit) patterns win
        if re.match(pat, name):
            hit = g
    return hit or "other"


def released_script(p: Path) -> bool:
    r = p.relative_to(S)
    return (p.is_file() and not set(r.parts) & SKIP_DIRS and p.suffix != ".pyc" and p.name not in EXCLUDE_SCRIPTS
            and not (p.name.startswith("_") and p.name != "__init__.py"))


def script_range(prefix, q=""):
    """'930-938': first and last released script number with this prefix (q: quote character around each)."""
    nums = sorted(int(m.group(1)) for p in S.glob(f"{prefix}*") if (m := re.match(r"(\d+)_", p.name))
                  and released_script(p) and p.suffix in (".py", ".R", ".sh"))
    return f"{q}{nums[0]}{q}-{q}{nums[-1]}{q}" if nums else prefix


def index(san):
    man = pd.read_csv(RESULTS / "results_manifest.csv")
    n_out = man.groupby("script").size()
    rows = []
    for p in sorted(S.glob("*")):
        if p.suffix not in (".py", ".R", ".sh") or not released_script(p):
            continue
        purpose = rl.public_text(san.apply(header(p), "text")[0], SUBS)
        rows.append(dict(script=p.name, language={".py": "Python", ".R": "R", ".sh": "shell"}[p.suffix],
                         group=group_of(p.name), purpose=purpose[:300], registered_outputs=int(n_out.get(p.name, 0))))
    idx = pd.DataFrame(rows)
    write_csv(idx, MOD, "scripts_index.csv", "Code release: every released analysis script with its language, group, "
              "purpose (from its header, without internal identifiers) and the number of registered outputs it writes "
              "(904)", required=["script", "language", "group", "purpose", "registered_outputs"])
    loc = [("python", platform.python_version())]
    for m in LOCAL_PKGS:
        try:
            loc.append((m, md.version(m)))
        except md.PackageNotFoundError:
            loc.append((m, "not installed"))
    lv = pd.DataFrame(loc, columns=["package", "version"])
    write_csv(lv, MOD, "local_software_versions.csv", "Code release: local environment for imaging, flow cytometry, "
              "figures, tables and the manuscript (904)", required=["package", "version"])
    return idx, lv


def stage_submission(tree, idx, lv, args, san, checks):
    """The submission's code archive for editors and reviewers (<manuscript>/submission/code_release/): README.md
    (written here from the current scripts, groups and figures), the scripts index, local and server software
    versions, and coch_rythum_code.zip (the sanitised scripts, environment specifications, licence and README)."""
    STAGE.mkdir(parents=True, exist_ok=True)
    for src in [RESULTS / MOD / "scripts_index.csv", RESULTS / MOD / "local_software_versions.csv",
                RESULTS / "00_env" / "software_versions.csv"]:
        shutil.copy2(src, STAGE / src.name)
    readme = code_release_readme(idx, lv, args)
    rl.write_text(STAGE / "README.md", readme)
    hits = [h for h in rl.scan_file(STAGE / "README.md", rl.leak_patterns(san) + rl.DOC_PATTERNS) if h[1] == "leak"]
    checks.append(("submission code_release README: no paths, hosts, names or internal IDs", len(hits),
                   "pass" if not hits else "fail", "; ".join(f"line {h[2]} {h[0]}" for h in hits[:5])))
    zp = STAGE / "coch_rythum_code.zip"
    code, env = tree.root / "code" / "scripts", tree.root / "code" / "env"
    tmp = zp.with_name(zp.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(code.rglob("*")):
            if p.is_file():
                z.write(p, arcname=f"coch-rythum/scripts/{p.relative_to(code).as_posix()}")
        for p in sorted(env.glob("*")):
            if p.is_file():
                z.write(p, arcname=f"coch-rythum/env/{p.name}")
        z.writestr("coch-rythum/README.md", readme)
        z.writestr("coch-rythum/LICENSE", rl.APACHE_2_0)
        for f in ("scripts_index.csv", "local_software_versions.csv", "software_versions.csv"):
            z.write(STAGE / f, arcname=f"coch-rythum/{f}")
    os.replace(tmp, zp)
    log(f"submission code archive {zp.name} ({zp.stat().st_size / 1e6:.1f} MB)")


def code_release_readme(idx, lv, args):
    """README of the submission's code archive (no internal identifiers or paths; author items as [AUTHOR: ...])."""
    title = rl.tex_to_text(re.search(r"\\title\{([^{}]+)\}", (MANUSCRIPT / "main.tex").read_text(encoding="utf-8"))
                           .group(1))
    sv = pd.read_csv(RESULTS / "00_env" / "software_versions.csv")
    pyv = sv.loc[(sv.language == "python") & (sv.package == "python"), "version"]
    rv = sv.loc[(sv.language == "R") & (sv.package == "R"), "version"]
    g = idx.assign(num=idx.script.map(lambda s: int(re.match(r"(\d+)", s).group(1)) if re.match(r"\d", s) else 999))
    order = (g.sort_values("num", kind="stable").groupby("group", sort=False)
             .agg(first=("script", "first"), last=("script", "last"), n=("script", "count")).reset_index())
    groups = pd.DataFrame({"Scripts": [f"`{a.split('_')[0]}`-`{b.split('_')[0]}` ({n})" for a, b, n in
                                       zip(order["first"], order["last"], order["n"])], "Group": order["group"]})
    pi = load_panel_index()
    figs = pd.DataFrame([dict(Figure=f"{lab}: {t}" if t else lab, File=f"`{stem}.pdf`",
                              Script=", ".join(f"`{x}`" for x in figure_script(stem, fig_key(lab), pi)))
                         for stem, lab, t in figure_list()])
    n_st, _ = st_file("public_datasets")
    return f"""# {title}: analysis code

Code of all analyses, figures and tables of the article {args.article or PH['article']}.
Repository: {args.github or PH['github']}; archived release with data and results: {args.doi or PH['doi']}.

## Data

- **Atlas.** Single-cell RNA sequencing of the mouse cochlea and the adjacent temporal-bone marrow at six times of
  day. The processed count matrix with cell annotations (h5ad) is part of the archived release; raw reads:
  {args.geo or PH['geo']}. The scripts read the atlas at `data/processed/coca_circadian.h5ad` (paths in
  `scripts/config.yaml`).
- **Public data.** Accessions, uses and references are listed in Supplementary Table {n_st} of the article.
- **Flow cytometry and imaging.** The event-level flow export and the original image archives are part of the
  archived release.

## Environments

- Server analyses: Python {pyv.iat[0] if len(pyv) else ''} and R {rv.iat[0] if len(rv) else ''} (package versions in
  `software_versions.csv`; conda and R specifications in `env/`).
- Imaging, flow cytometry, figures, tables and manuscript checks: Python
  {lv.loc[lv.package == 'python', 'version'].iat[0]} (versions in `local_software_versions.csv`).
- Thread limits for shared machines: `scripts/setup/threads.sh`.

## Layout

`scripts/` holds the numbered analysis modules; `scripts_index.csv` lists every script with its group, purpose and the
number of result tables it registers. Every result table is written through `scripts/lib/io.py` (`write_csv`), which
records path, rows, columns, MD5 and the producing script; `99_results_manifest.py` compiles the catalogue
`results/results_manifest.csv`.

{rl.md_table(groups)}

## Figures

{rl.md_table(figs)}

## Running

1. Run the scripts from the project root, with `scripts/` there; paths in `scripts/config.yaml` are relative to it
   (the environment variable `COCH_ROOT` sets the root). Shell chains in `scripts/setup/` also read `COCH_PYTHON`,
   `COCH_RSCRIPT` and `COCH_ENVS`.
2. Run the modules of each group in numeric order; `scripts/setup/chain_*.sh` record the order used, and
   `scripts/setup/chain_tail.sh` rebuilds the figure index, evidence register, audits and catalogue after any analysis.
3. Figures: `900_figure_data.py` writes the figure tables (they double as Source Data), and the figure scripts above
   draw them. Supplementary Tables and Source Data: `908_supplementary_tables.py` and `909_source_data.py`.

Script headers cite the authors' internal analysis log as `D<number>` or `I<number>`; the log is not part of this
archive and is not needed to run the code.

## Licence

Apache License 2.0 (`LICENSE`).

## Citation

{args.article or PH['article']}
"""


# ------------------------------------------------------------------------------------------------- release tree
class Tree:
    """The release folder: placed files and their manifest descriptions."""

    def __init__(self, root: Path):
        assert root.name.startswith(NAME + "_v"), root
        self.root = root
        self.desc = {}

    def copy(self, src, rel, desc):
        rl.copy_file(src, self.root / rel)
        self.desc[rel] = desc

    def text(self, rel, text, desc):
        rl.write_text(self.root / rel, text)
        self.desc[rel] = desc

    def size(self, prefix=""):
        return sum((self.root / r).stat().st_size for r in self.desc if r.startswith(prefix))

    def prune(self):
        """Remove what this run did not place (files of earlier runs: renamed figures, dropped tables; links such as a
        scripts/ link made by prepare_workspace.py are removed as links, never followed)."""
        keep = set(self.desc) | {"MANIFEST.tsv", "MD5SUMS.txt"}
        files, links, dirs = rl.walk(self.root)
        gone = []
        for r in links:                                    # the link only; its target is untouched
            p = self.root / r
            try:
                os.unlink(p)
            except OSError:
                os.rmdir(p)                                # Windows directory symlink or junction
            gone.append(r)
        for r in files:
            if r not in keep:
                (self.root / r).unlink()
                gone.append(r)
        for r in sorted(dirs, key=lambda x: x.count("/"), reverse=True):
            p = self.root / r
            if not any(p.iterdir()):
                p.rmdir()
        return gone


def sanitizer(args):
    """Rules from config.yaml (server root, environments), the ssh host, the local root and the user's account names
    (names that the authors list or the contact give are not redacted)."""
    names = {Path.home().name} | {os.environ.get(k, "") for k in ("USERNAME", "USER")}
    names |= {p for p in re.split(r"[\\/]", os.environ.get("TEMP", "")) if "~" in p}
    public = set(re.findall(r"\w+", f"{args.authors or ''} {args.contact or ''}".lower()))
    names = {n for n in names if n and not set(re.findall(r"\w+", n.lower())) <= public}
    named = set()                    # folders read through an environment variable with a local default (COCH_IMAGES)
    for p in S.rglob("*.py"):
        if released_script(p):
            named |= {(path, var) for var, path in re.findall(
                r"os\.environ\.get\(\s*[\"'](\w+)[\"']\s*,\s*[\"']([A-Za-z]:[\\/][^\"']*)[\"']\s*\)",
                p.read_text(encoding="utf-8", errors="replace"))}
    return rl.Sanitizer(CFG["paths"]["root"], CFG["envs"]["python"], CFG["envs"]["rscript"],
                        local_roots=[str(ROOT)], hosts=[args.server] if args.server else [], names=sorted(names),
                        named_roots=sorted(named))


def release_config(text: str) -> str:
    """config.yaml for the release layout: paths relative to the project root (COCH_ROOT overrides paths.root)."""
    subs = [(r"^# Shared configuration.*$", "# Shared configuration of all analysis scripts. Paths are relative to the "
             "project root (the release folder);\n# the environment variable COCH_ROOT overrides paths.root."),
            (r"^(\s*root:)\s*\S+.*$", r"\1 ."),
            (r"^(\s*h5ad:)\s*\S+.*$", r"\1 data/processed/coca_circadian.h5ad"),
            (r"^\s*#[^\n]*\n(\s*manuscript:)\s*\S+.*$", r"  # LaTeX sources of the manuscript\n\1 manuscript")]
    for pat, rep in subs:
        text, n = re.subn(pat, rep, text, count=1, flags=re.M)
        assert n == 1, f"config.yaml: no match for {pat}"
    return text


def stage_code(tree, san, idx, lv):
    purposes = dict(zip(idx.script, idx.purpose))
    groups = dict(zip(idx.script, idx.group))
    n_sub, changes = 0, []
    for p in sorted(S.rglob("*")):
        if not released_script(p):
            continue
        r = p.relative_to(S).as_posix()
        dest = f"code/scripts/{r}"
        head = rl.public_text(san.apply(header(p), "text")[0], SUBS)
        if r == "setup/chain_tail.sh":
            head = re.sub(r"[ \t]*->[ \t]*95\b", "", head)
        if "/" not in r and p.name in purposes:
            desc = f"Analysis script ({groups[p.name]}): {purposes[p.name]}"
        elif r.startswith("lib/"):
            desc = f"Shared library: {head}" if head else "Shared library"
        elif r.startswith("setup/"):
            desc = f"Environment setup or pipeline chain: {head}" if head else "Environment setup or pipeline chain"
        elif r.startswith("resources/"):
            desc = "Curated resource read by the analyses"
        elif r == "config.yaml":
            desc = "Shared configuration: paths (relative to the project root), design, thresholds"
        else:
            desc = "Analysis code"
        if p.suffix.lower() in rl.TEXT or p.suffix.lower() == ".r":
            text = orig = p.read_text(encoding="utf-8")
            if r == "config.yaml":
                text = release_config(text)
            if r == "setup/chain_tail.sh":                  # the excluded internal-log step
                text = re.sub(r"[ \t]*->[ \t]*95\b", "", text)
                text = re.sub(r"[ \t]*\b95_memory_update\.py\b", "", text)
                text = re.sub(r"[ \t]*\(CLAUDE\.md\)", "", text)
            text, n = san.apply(text, rl.flavour_of(p))
            n_sub += n
            if text != orig:                               # audit trail of every changed line (not released)
                a, b = orig.replace("\r\n", "\n").split("\n"), text.split("\n")
                for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                    if op != "equal":
                        changes.append(dict(path=dest, line=i1 + 1, before="\n".join(a[i1:i2])[:500],
                                            after="\n".join(b[j1:j2])[:500]))
            tree.text(dest, text, desc)
        else:
            tree.copy(p, dest, desc)
    # environments
    sv = pd.read_csv(RESULTS / "00_env" / "software_versions.csv")
    tree.text("code/env/software_versions_server.csv", sv.to_csv(index=False),
              "Software versions of the server analyses (language, package, version)")
    tree.text("code/env/software_versions_local.csv", lv.to_csv(index=False),
              "Software versions of the local imaging, flow, figure, table and manuscript scripts")
    specs = rl.env_specs(sv, lv)
    for name, text in specs.items():
        tree.text(f"code/env/{name}", text, "Environment specification derived from the recorded versions")
    rp = rl.r_packages(sv)
    tree.text("code/env/r_packages.csv", rp.to_csv(index=False), "R packages: version, source and install route")
    tree.text("code/env/README.md", env_readme(sv, lv, rp), "Environments: how to recreate them")
    tree.text("code/docs/scripts_index.csv", idx.to_csv(index=False),
              "Every released script: language, group, purpose, registered outputs")
    return n_sub, pd.DataFrame(changes, columns=["path", "line", "before", "after"])


def env_readme(sv, lv, rp):
    py = sv[sv.language == "python"]
    r = sv[sv.language == "R"]
    return f"""# Software environments

The analyses ran in two places:

- **Analysis server (Linux)**: Python {py.loc[py.package == 'python', 'version'].iat[0]} and
  R {r.loc[r.package == 'R', 'version'].iat[0]}, for everything from quality control to rhythm statistics,
  composition models, external data and the evidence register. GPU training (scVI / scANVI) used CUDA.
- **Workstation**: Python {lv.loc[lv.package == 'python', 'version'].iat[0]} for imaging, flow cytometry, figures,
  Supplementary Tables, Source Data and the manuscript checks (scripts 900-937 that state a local run).

| File | Content |
| --- | --- |
| `software_versions_server.csv` | every package imported by the server analyses, with the version used |
| `software_versions_local.csv` | the workstation packages, with the version used |
| `environment-server-python.yml` | conda environment pinned to the server Python versions |
| `requirements-trajectory-overlay.txt` | separate overlay (own SciPy) for the trajectory checks of script 630 |
| `environment-server-r.yml` | conda environment pinned to the server R versions (CRAN and Bioconductor packages) |
| `r_packages.csv` | R packages with source and install route; {int((rp.source == 'GitHub').sum())} packages come from GitHub |
| `environment-local-python.yml` | conda environment pinned to the workstation versions |

```bash
conda env create -f code/env/environment-server-python.yml
conda env create -f code/env/environment-server-r.yml
conda env create -f code/env/environment-local-python.yml
pip install --target envs/traj_pkgs -r code/env/requirements-trajectory-overlay.txt
```

Install the GitHub R packages with `remotes::install_github()` as listed in `r_packages.csv`. The shell chains in
`code/scripts/setup/` call the interpreters named by `COCH_PYTHON` and `COCH_RSCRIPT` (default: `python` and `Rscript`
on the PATH) and look for environments in `COCH_ENVS` (default: `envs/` in the project root). The original
installation scripts are kept in `code/scripts/setup/` (`env_python.sh`, `env_r.sh`, `env_r_deps.sh`, `env_r3.sh`,
`r_extra_pkgs.R`, `github_clone.sh`); they used package mirrors that you may replace with the default channels.
"""


# ------------------------------------------------------------------------------------------------------- data
def ssh(host, cmd, timeout=600):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, cmd], capture_output=True,
                       text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"ssh command failed ({r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout


def obtain_h5ad(args, dst: Path):
    """Copy the processed atlas into the release unless the release copy has the source's MD5; (md5, status)."""
    spec = args.h5ad or (f"{args.server}:{CFG['paths']['root']}/{CFG['paths']['h5ad']}" if args.server else None)
    if spec and ":" in spec and not re.match(r"^[A-Za-z]:[\\/]", spec):
        host, path = spec.split(":", 1)
        src_md5 = ssh(host, f"md5sum '{path}'").split()[0]
        if dst.exists() and rl.md5sum(dst) == src_md5:
            return src_md5, "release copy has the MD5 of the source"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        log(f"copying the atlas from {host} ...")
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", spec, str(tmp)], check=True, timeout=7200)
        assert rl.md5sum(tmp) == src_md5, "atlas copy: MD5 differs from the source"
        os.replace(tmp, dst)
        return src_md5, "copied from the source (MD5 equal)"
    if spec:
        src = Path(spec)
        rl.copy_file(src, dst)
        m = rl.md5sum(dst)
        assert m == rl.md5sum(src), "atlas copy: MD5 differs from the source"
        return m, "copied from a local source (MD5 equal)"
    assert dst.exists(), "no atlas: give --h5ad or --server"
    return rl.md5sum(dst), "kept from the previous run (not compared with the source)"


def doc_for(col):
    for key, d in OBS_DOC:
        if key == col or (key.endswith("*") and col.startswith(key[:-1])):
            return d
    return ""


def stage_atlas(tree, args, checks):
    rel = "data/processed/coca_circadian.h5ad"
    dst = tree.root / rel
    m, status = obtain_h5ad(args, dst)
    checks.append(("atlas h5ad", m, "pass" if "MD5 equal" in status or "has the MD5" in status else "warn", status))
    tree.desc[rel] = "Processed single-cell atlas (AnnData): raw counts and annotations of the 12 samples"
    sch = rl.h5ad_schema(dst)
    obs = rl.h5ad_frame(dst, "obs", ["batch_id", "tissue", "day_time", "platform", "cell_type", "cell_class",
                                     "cell_type_abbr"])
    var = rl.h5ad_frame(dst, "var")
    st = sample_table(obs)
    n_obs, n_var = sch["X"]["shape"]
    plat = obs.groupby(["tissue", "platform"], observed=True).size().reset_index()
    plat_txt = "; ".join(f"{t}: {PLATFORMS.get(p, p)} (`{p}`)" for t, p, _ in plat.itertuples(index=False))
    ocols = []
    for c in sch["obs"]["columns"]:
        lv = c["levels"]
        if c["type"].startswith("categorical"):
            vals = ", ".join(map(str, lv[:6])) + (f", ... ({c['n_levels']} levels)" if c["n_levels"] > 6 else "")
        elif c["type"] == "boolean":
            vals = "; ".join(lv)
        elif c["type"] == "string":
            vals = ", ".join(map(str, lv[:3]))
        elif lv[0] == lv[2]:
            vals = f"all {lv[0]}"
        else:
            vals = f"min {lv[0]}, median {lv[1]}, max {lv[2]}"
        d = doc_for(c["column"])
        if c["column"] == "platform":
            d += f"; {plat_txt}"
        ocols.append(dict(column=f"`{c['column']}`", type=c["type"], values=vals, description=d))
    land = [c for c in ocols if re.match(r"`landmark_mem_\d+`", c["column"])]
    if len(land) > 3:                                   # one row for the numbered landmark memberships
        first = ocols.index(land[0])
        ocols = [c for c in ocols if c not in land]
        ocols.insert(first, dict(column=f"`landmark_mem_1` ... `landmark_mem_{len(land)}`", type=land[0]["type"],
                                 values="0 to 1", description=land[0]["description"]))
    samples = pd.DataFrame({"sample (`batch_id`)": st.sample_id,
                            "tissue": st.tissue.map({"bm": "temporal-bone marrow"}).fillna(st.tissue),
                            "platform": st.platform.map(lambda p: PLATFORMS.get(p, p)),
                            "clock time": st.clock_time.map(lambda v: f"{v:02.0f}:00"),
                            "ZT": st.zt.map(lambda v: f"ZT{v:.0f}"), "cells": st.n_cells.map(lambda v: f"{v:,}")})
    ct = (obs.groupby(["cell_class", "cell_type", "cell_type_abbr", "tissue"], observed=True).size()
          .unstack("tissue", fill_value=0).reset_index())
    ct.columns = [str(c) for c in ct.columns]
    for c in ("cochlea", "bone marrow"):
        if c in ct:
            ct[c] = ct[c].map(lambda v: f"{v:,}")
    ct = ct.rename(columns={"cell_class": "class", "cell_type": "cell type", "cell_type_abbr": "abbreviation",
                            "bone marrow": "marrow"})
    missing = [g for g in ("Arntl", "Xist", "Ddx3y") if g not in set(var.index.astype(str))]
    other = []
    for k in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
        items = ", ".join(f"`{n}` {tuple(s)}" if s else f"`{n}`" for n, s, _ in sch[k]) or "empty"
        other.append(f"- `{k}`: {items}")
    x = sch["X"]
    text = f"""# Processed single-cell atlas

`coca_circadian.h5ad` is an AnnData file (HDF5; written with anndata 0.13, readable with anndata 0.11 or later) with
**{n_obs:,} cells x {n_var:,} genes** from the 12 samples of this study: the cochlea and the adjacent temporal-bone marrow
at six times of day (ZT3, ZT7, ZT11, ZT15, ZT19 and ZT23; lights on at 08:00, so clock time = ZT + 8 h).

| File | Size | MD5 |
| --- | --- | --- |
| `coca_circadian.h5ad` | {rl.human(dst.stat().st_size)} ({dst.stat().st_size:,} bytes) | `{m}` |

## Expression matrix (`X`)

Raw UMI counts ({'integer values' if x['integer'] else 'values'} stored as {x['dtype']}), a sparse
{x['storage'].replace('_matrix', '').upper()} matrix with {x['nnz']:,} non-zero entries; no normalised layer is stored.
Genes (`var` index, {n_var:,}) are MGI symbols of the
protein-coding and mitochondrially encoded genes of the GRCm39 autosomes and mitochondrial chromosome; sex-chromosome
genes are excluded{', and ' + ', '.join(f'*{g}*' for g in missing if g == 'Arntl') + ' is absent' if 'Arntl' in missing else ''}.
`var` has no further columns.

## Samples

Each sample pools three mice per tissue and time point. Library chemistry: {plat_txt}.

{rl.md_table(samples)}

## Cell annotations (`obs`)

{rl.md_table(pd.DataFrame(ocols))}

`cell_type`, `cell_class` and `cell_type_abbr` are the reference annotation of the atlas and were used throughout the
analyses without re-annotation. Quality control, integration and annotation were done on these 12 libraries together
with 30 libraries of other experiments; this file holds the 12 libraries of this study only (see Methods and
Supplementary Information of the article).

## Cell types

{rl.md_table(ct)}

## Other slots

{chr(10).join(other)}

`obsm['X_umap']` is the joint UMAP shown in the article.

## Reading the file

```python
import anndata as ad

adata = ad.read_h5ad("data/processed/coca_circadian.h5ad")
adata.obs["zt"] = (adata.obs["day_time"].astype(float) - {LIGHTS_ON:g}) % 24      # Zeitgeber time (h)
cochlea = adata[adata.obs["tissue"] == "cochlea"]
```

In R, the file can be read with `zellkonverter::readH5AD()` (Bioconductor) or the `anndata` package.

## Provenance

Cochlear libraries: 10x Genomics Chromium Single Cell 3' v4; marrow libraries: MobiSeq v2; Illumina NovaSeq 6000.
Count matrices: Cell Ranger 10.0.0 (10x) and MobiVision 2.0 (MobiSeq) against GRCm39. Raw reads: {args.geo or PH['geo']}.
"""
    tree.text("data/processed/README.md", text, "Schema of the processed atlas (read from the file)")
    return dict(n_cells=n_obs, n_genes=n_var, n_samples=len(st), n_types=obs["cell_type"].nunique(),
                n_classes=obs["cell_class"].nunique(), md5=m, platforms=plat_txt)


def count_lines(p: Path) -> int:
    n = 0
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            n += b.count(b"\n")
    return n


def stage_flow(tree, checks):
    files = sorted(p for p in FLOW_DIR.iterdir() if p.is_file())
    desc = {"events.csv": "Flow cytometry: every event (scatter, two fluorescence channels, fixed gate flags)",
            "tube_values.csv": "Flow cytometry: per tube and sample (biological tubes and controls): markers, gate counts",
            "sample_values.csv": "Flow cytometry: per mouse and tissue: cell counts, viability and readouts",
            "channels.csv": "Flow cytometry: markers and fluorochromes of tubes T1-T8",
            "gates.csv": "Flow cytometry: fixed gate thresholds", "README.txt": "Flow cytometry: notes of the export"}
    rows = []
    for p in files:
        tree.copy(p, f"data/flow/{p.name}", desc.get(p.name, "Flow cytometry export"))
        if p.suffix == ".csv":
            cols = pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns
            rows.append(dict(file=f"`{p.name}`", rows=f"{count_lines(p) - 1:,}", columns=len(cols),
                             content=desc.get(p.name, "").split(": ", 1)[-1]))
    tree.copy(FLOW_ZIP, f"data/flow/{FLOW_ZIP.name}", "Flow cytometry: the original export archive (same files)")
    with zipfile.ZipFile(FLOW_ZIP) as z:
        same = {i.filename: i.file_size for i in z.infolist() if not i.is_dir()} == {p.name: p.stat().st_size
                                                                                    for p in files}
    checks.append(("flow export = original archive", f"{len(files)} files", "pass" if same else "warn",
                   "names and sizes of the flow files match the members of the original archive"))
    sv = pd.read_csv(FLOW_DIR / "sample_values.csv", encoding="utf-8-sig")
    ch = pd.read_csv(FLOW_DIR / "channels.csv", encoding="utf-8-sig")
    ga = pd.read_csv(FLOW_DIR / "gates.csv", encoding="utf-8-sig")
    mice = sv.drop_duplicates("mouse_id")
    zts = sorted(sv.zt_group.astype(str).unique(), key=lambda z: float(re.sub(r"\D", "", z) or 0))
    design = (mice.assign(zt_group=pd.Categorical(mice.zt_group.astype(str), zts))
              .groupby(["cohort", "zt_group"], observed=True).size().rename("mice").reset_index()
              .rename(columns={"zt_group": "time"}))
    sts = [int(re.search(r"_(\d+)_", p.name).group(1))
           for p in sorted((RESULTS / "r9_suppl").glob("Supplementary_Table_*flow*.csv"))]
    st_txt = ("in Supplementary Table" + ("s " if len(sts) > 1 else " ") + " and ".join(map(str, sts))) if sts else \
        "in the Supplementary Tables"
    who = (f"{len(mice)} mice ({', '.join(sorted(mice.strain.astype(str).unique()))}, "
           f"{', '.join(sorted(mice.sex.astype(str).unique()))}, "
           f"{', '.join(sorted(mice.age_weeks.astype(str).unique()))} weeks)")
    text = f"""# Flow cytometry: pre-specified day-night validation cohort

Two-colour flow cytometry (FITC and PE) of the cochlea and the temporal-bone marrow of {who}, collected at
{' and '.join(zts)} in {sv.cohort.nunique()} cohorts. Readouts, gates and tests were fixed before the data were
analysed; the protocol and gating are described in the Methods and Supplementary Information of the article.

{rl.md_table(design)}

## Files

{rl.md_table(pd.DataFrame(rows))}

- `README.txt`: notes that came with the export (how to join and plot the tables).
- `{FLOW_ZIP.name}`: the original export; it holds the same {len(files)} files.

Join `events.csv` to `tube_values.csv` on (`record_id`, `tube`) for tissue, time group and markers; `record_id` is the
mouse and tissue (for example `M001_CO` cochlea, `M001_BM` marrow), `record_type` separates biological tubes from
controls. Gate columns are 0 / 1 flags of the fixed gates below.

### Tubes

{rl.md_table(ch)}

### Gate thresholds

Fluorescence gates in bead-relative units; scatter gates (FSC, SSC) in instrument units; singlet gates as FSC-H / FSC-A.

{rl.md_table(ga)}

## Analyses

`code/scripts/932_flow_validation.py` (readouts, pre-specified tests, figure panels) and
`code/scripts/933_flow_controls.py` (controls) write `results/r9_flow/`; per-mouse readouts and all tests are
{st_txt}.
"""
    tree.text("data/flow/README.md", text, "Flow cytometry: description of the export")
    return dict(mice=len(mice), zts=" and ".join(zts), events=rows[[r["file"] for r in rows].index("`events.csv`")]["rows"]
                if any(r["file"] == "`events.csv`" for r in rows) else "")


_SI = None


def si_key(key) -> str:
    """Panel-index ID of a Supplementary figure -> published key ('SROB' -> 'S6'), from SI_NUMBER of 909 (read with ast,
    without running 909); main and Extended Data keys, and unknown IDs, are returned unchanged."""
    global _SI
    if _SI is None:
        import ast
        _SI = {}
        src = S / "909_source_data.py"
        if src.exists():
            for node in ast.parse(src.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SI_NUMBER" for t in node.targets):
                    _SI = {str(k): int(v) for k, v in ast.literal_eval(node.value).items()}
    key = str(key)
    return f"S{_SI[key]}" if key in _SI else key


def load_panel_index(published=None) -> pd.DataFrame:
    """panel_index.csv with figure keys in the published numbering; rows of figures that are not published (retired
    or internal) are dropped when the published keys are given."""
    pi = pd.read_csv(RESULTS / "r9_figures" / "panel_index.csv")
    pi["figure_internal"] = pi["figure"].astype(str)
    pi["figure"] = pi["figure_internal"].map(si_key)
    return pi[pi.figure.isin(published)].copy() if published is not None else pi


def fig_key(label: str) -> str:
    """'Fig. 4' -> '4'; 'Extended Data Fig. 5' -> 'ED5'; 'Supplementary Fig. 2' -> 'S2'."""
    num = label.replace("~", " ").split()[-1]
    return ("ED" if label.startswith("Extended") else "S" if label.startswith("Supplementary") else "") + num


def key_label(key: str, panel: str = "") -> str:
    key = str(key)
    if key.startswith("ED"):
        return f"Extended Data Fig. {key[2:]}{panel}"
    if key.startswith("S"):
        return f"Supplementary Fig. {key[1:]}{panel}"
    return f"Fig. {key}{panel}"


def panel_labels(pi: pd.DataFrame, needle: str) -> str:
    hit = pi[pi["files"].astype(str).str.contains(re.escape(needle), regex=True)]
    out = []
    for f, g in hit.groupby("figure", sort=False):
        out.append(key_label(f, "".join(sorted(g["panel"].astype(str).unique()))))
    return "; ".join(out)


def image_sources(tree, pi):
    """figure panel -> archive member for every displayed or quantified image recorded in the figure tables."""
    zips = {}

    def members(archive):
        if archive not in zips:
            with zipfile.ZipFile(tree.root / archive) as z:
                zips[archive] = {i.filename: i for i in z.infolist()}
        return zips[archive]

    def locate(path):
        p = str(path).strip().replace("\\", "/")
        for folder, archive, inner in WORKSPACE_MAP:
            if p.startswith(folder):
                return archive, inner + p[len(folder):]
            short = folder.split("/", 2)[-1]                      # without data/photos/ (pubclean form)
            if p.startswith(short):
                return archive, inner + p[len(short):]
        return "", ""

    rows = []
    r9, pb = RESULTS / "r9_figures", RESULTS / "r10_parabiosis"
    log_t = r9 / "image_panels_log.csv"
    if log_t.exists():
        for r in pd.read_csv(log_t).itertuples():
            files = [s.strip() for s in str(r.source_files).split(";") if s.strip()]
            sums = [s.strip() for s in str(getattr(r, "source_md5", "")).split(";")]
            for i, f in enumerate(files):
                a, m = locate(f)
                rows.append(dict(figure_panels=panel_labels(pi, r.panel), use="displayed", source_table=log_t.name,
                                 item=r.panel, archive=a, member=m, md5=sums[i] if i < len(sums) else ""))
    fld = r9 / "f4_parabiont_fields.csv"
    if fld.exists():
        lab = panel_labels(pi, fld.name)
        for r in pd.read_csv(fld).itertuples():
            files = [s.strip() for s in str(r.source_files).split(";") if s.strip()]
            sums = [s.strip() for s in str(r.source_md5).split(";")]
            for i, f in enumerate(files):
                a, m = locate(f)
                rows.append(dict(figure_panels=lab, use="quantified", source_table=fld.name, item=r.field, archive=a,
                                 member=m, md5=sums[i] if i < len(sums) else ""))
    img = r9 / "sfig_parabiosis_images.csv"
    if img.exists():
        for r in pd.read_csv(img).itertuples():
            acq = str(r.acquisition)
            is_path = "/" in acq or bool(Path(acq).suffix)          # else a field label without its file path
            rows.append(dict(figure_panels=key_label(si_key(r.figure), str(r.panel)), use="displayed", source_table=img.name,
                             item=r.image_file, archive="data/imaging/Parabiosis_Mice_ALL.zip" if is_path else "",
                             member="Parabiosis_Mice/" + acq if is_path else f"(field label {acq})", md5=""))
    for name in ("parabiosis_tissue_fields.csv", "parabiosis_perilymph_fields.csv", "parabiosis_noise_sections.csv",
                 "parabiosis_celltype_fields.csv"):
        t = pb / name
        if t.exists():
            lab = panel_labels(pi, name) or panel_labels(pi, name.replace("parabiosis_", "pb_")) or \
                "parabiosis survey statistics"
            tab = pd.read_csv(t)
            col = "path" if "path" in tab else "field"
            for v in tab[col].astype(str).drop_duplicates():
                is_path = "/" in v
                rows.append(dict(figure_panels=lab, use="quantified", source_table=name, item=Path(v).name,
                                 archive="data/imaging/Parabiosis_Mice_ALL.zip" if is_path else "",
                                 member="Parabiosis_Mice/" + v if is_path else f"(field label {v})", md5=""))
    df = pd.DataFrame(rows).drop_duplicates()
    found, verified, bad = [], [], []
    for r in df.itertuples():
        mem = members(r.archive) if r.archive else {}
        info = mem.get(r.member)
        if not r.archive:
            found.append("label only")                     # the figure table gives no file path
        elif info is None and r.member:                   # a folder of exported planes
            n = sum(1 for k in mem if k.startswith(r.member.rstrip("/") + "/") and not k.endswith("/"))
            found.append(f"folder ({n} files)" if n else "")
        else:
            found.append("file" if info is not None else "")
        ok = ""
        if info is not None and r.md5 and re.fullmatch(r"[0-9a-f]{32}", r.md5) and info.file_size < 200e6:
            with zipfile.ZipFile(tree.root / r.archive) as z:
                h = rl.hashlib.md5(z.read(r.member)).hexdigest()
            ok = "yes" if h == r.md5 else "no"
            if ok == "no":
                bad.append(r.member)
        verified.append(ok)
    df["member_found"] = found
    df["md5_matches"] = verified
    return df, bad


def _found(df):
    return df.member_found.astype(str) != ""


def _labels(df):
    return df.member_found.astype(str) == "label only"


def stage_imaging(tree, pi, checks):
    for src, desc in IMAGE_ARCHIVES:
        tree.copy(src, f"data/imaging/{src.name}", desc)
    df, bad = image_sources(tree, pi)
    tree.text("data/imaging/figure_image_sources.csv", df.to_csv(index=False),
              "Imaging: figure panel -> archive member of every displayed or quantified image")
    paths = df[~_labels(df)]
    checks.append(("image sources found in the archives", f"{int(_found(paths).sum())}/{len(paths)}",
                   "pass" if _found(paths).all() else "fail", "; ".join(paths.loc[~_found(paths), "member"].head(5))))
    if _labels(df).any():
        checks.append(("image sources named by field label only (no file path in the figure table)",
                       int(_labels(df).sum()), "note", "; ".join(sorted(set(df.loc[_labels(df), "member"]))[:6])))
    n_md5 = int((df.md5_matches != "").sum())
    checks.append(("image sources MD5 = figure tables", f"{int((df.md5_matches == 'yes').sum())}/{n_md5}",
                   "pass" if not bad else "fail", "; ".join(bad[:5])))
    # Brain_spleen.zip against the nested organ archive of the parabiosis collection
    same_nested = ""
    with zipfile.ZipFile(tree.root / "data/imaging/Parabiosis_Mice_ALL.zip") as z:
        nested = [n for n in z.namelist() if n.endswith("Brain_and_Liver.zip")]
        if nested:
            import io
            inner = zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
            a = {i.filename: i.CRC for i in inner.infolist() if not i.is_dir()}
            with zipfile.ZipFile(tree.root / "data/imaging/Brain_spleen.zip") as bz:
                b = {i.filename: i.CRC for i in bz.infolist() if not i.is_dir()}
            same_nested = "identical" if a == b else "different"
    # acquisition summary of the parabiosis survey
    acq = RESULTS / "r10_parabiosis" / "parabiosis_acquisitions.csv"
    acq_txt = ""
    if acq.exists():
        a = pd.read_csv(acq)
        inc = a[a["included"].astype(str).str.lower() == "true"]
        px = inc["um_per_px"].astype(float)
        chans = inc["channels"].astype(str).value_counts().head(4)
        acq_txt = (f"The survey read {len(a)} acquisitions of the collection and analysed {len(inc)} "
                   f"(pixel sizes {px.min():.3f}-{px.max():.3f} µm; most frequent channel sets: "
                   + "; ".join(f"{c.replace(';', ', ')} ({n})" for c, n in chans.items())
                   + "); `results/r10_parabiosis/parabiosis_acquisitions.csv` lists every acquisition with its "
                   "metadata and the reason for exclusion.")
    lg = pd.read_csv(RESULTS / "r9_figures" / "image_panels_log.csv") if (RESULTS / "r9_figures" /
                                                                          "image_panels_log.csv").exists() else None
    px_txt = ""
    if lg is not None:
        px = (lg.dropna(subset=["um_per_px"]).assign(compartment=lambda d: d.compartment.str.replace("_", " "))
              .groupby(["kind", "compartment"])["um_per_px"].first())
        px_txt = "; ".join(f"{k} of {c}, {v:.3g} µm" for (k, c), v in px.items())
    meth = (MANUSCRIPT / "methods.tex").read_text(encoding="utf-8")
    m = re.search(r"light-sheet fluorescence microscopy\s*\(((?:[^()]|\([^()]*\))*)\)", meth)
    lsm = "" if not m else (" (" + rl.tex_to_text(m.group(1)) + ")" if "authorinput" not in m.group(1) else
                           " ([AUTHOR: light-sheet microscope])")
    label_txt = ""
    if _labels(df).any():
        label_txt = ("\nRows marked `(field label ...)` name an Olympus field of the 54-day parabiont by its label only; "
                     "`code/scripts/931_image_quant.py` and `938_parabiosis_cell_types.py` map the labels to their TIFF "
                     "folders (`FUT4-Neutrophils-English.zip`, `Parabiotic_Mouse_Section_Staining-Immune_Cell_Types/"
                     "cell-type-bone-cochlea/`).\n")
    use = (df.assign(archive=df.archive.str.replace("data/imaging/", "", regex=False).replace("", "(field label only)"))
           .groupby(["figure_panels", "use", "archive"]).size().rename("files").reset_index()
           .rename(columns={"figure_panels": "figure panels"}))
    arch = []
    for src, desc in IMAGE_ARCHIVES:
        p = tree.root / "data/imaging" / src.name
        with zipfile.ZipFile(p) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
        ext = pd.Series([Path(i.filename).suffix.lower().lstrip(".") or "other" for i in infos]).replace(
            {"tiff": "tif", "jpeg": "jpg"}).value_counts()
        arch.append(dict(file=f"`{src.name}`", size=rl.human(p.stat().st_size), files=f"{len(infos):,}",
                         unpacked=rl.human(sum(i.file_size for i in infos)),
                         formats=", ".join(f"{e.upper()} {n:,}" for e, n in ext.items()), content=desc))
    text = f"""# Imaging data

Original image archives of the study, unchanged. Each archive keeps the instrument files (with their embedded
acquisition metadata), the exported planes and the acquisition reports; folder and file names were translated into
English where needed, without changing the image data.

{rl.md_table(pd.DataFrame(arch))}

## Instruments and acquisition

- **ZEISS LSM 900** confocal microscope: the CZI series of the parabiosis survey (`Parabiosis_Mice_ALL.zip`; lectin-
  labelled vessels or FUT4, CD45, CD19 or CD14 immunostaining, GFP of the Lyz2-EGFP partner, DAPI) and the
  three-dimensional imaging of cleared cochleae as tiled z-stacks (lectin-labelled vessels, neutrophils).
- **Light-sheet fluorescence microscopy**{lsm}: whole-organ views of hydrogel-embedded cochleae. Volumes were processed
  in Fiji and rendered in Imaris; `cochlea_3d.zip` holds the rendered exports (8-bit PNG views, `painted/` the renders
  with the manually traced marrow-cochlea vessels). The raw three-dimensional volumes are not part of this release.
- **Olympus FV3000** confocal microscope with a 60x objective: the cochlear-fluid preparations (FUT4, CD45, DAPI) and the
  54-day parabiont sections (OIR acquisitions and their exported 16-bit TIFF planes, `FUT4-Neutrophils-English.zip`).
- Stereo-seq cell-type maps (Cell2location estimates) are included as rendered images
  (`cochlea_3d.zip`, `slide_03/spatial_transcriptomics/`); the Stereo-seq sequencing data are not part of this release.
- Pixel sizes of the displayed fields: {px_txt or 'see the acquisition reports'}. {acq_txt}

Channel assignments are given by the file names (for example `CD45-555-FUT4-488-DAPI`: CD45 with a 555-nm dye, FUT4
with a 488-nm dye) and by the acquisition reports (`*_Info.txt`, `*.txt`). Exported planes carry the channel in the
suffix (`_C1` ... `_C4` for Olympus exports, `_c1`, `_c2`, ... for ZEISS exports; `_z` for z planes).

## Naming and contents

- `Parabiosis_Mice_ALL.zip` contains `Parabiosis_Mice/INDEX.md`, an index of every file (batches by date, markers,
  pairing durations such as `5D`, `7d`, `14D`, `31d`, `54d`, region labels such as `bone`, `apex`, `SL`, and the
  `NOISE` label of the noise-exposed pair).
- `FUT4-Neutrophils-English.zip` contains `FUT4-Neutrophils/README-English.txt`; `WT-Celltype/` holds the wild-type
  marrow, perilymph and endolymph fields, `Parabiotic_Mouse_Section_Staining-Immune_Cell_Types/` the 54-day parabiont
  sections, and `Lymph_Fluid/` further fluid preparations. KFBF files are slide-scanner files (proprietary format).
- `cochlea_3d.zip`: `painted/` annotated vessel renders, `slide_NN/` exported views grouped as they were assembled
  (`color_images/`, `legends/`, `spatial_transcriptomics/`).
- `Brain_spleen.zip` holds the brain and liver projections; {'its files are identical to the nested `Brain_and_Liver.zip` of the parabiosis archive' if same_nested == 'identical' else 'compare with the nested `Brain_and_Liver.zip` of the parabiosis archive'}.

CZI files open in ZEISS ZEN or Fiji (Bio-Formats), OIR files in Olympus software or Fiji (Bio-Formats).

## Use in the figures

`figure_image_sources.csv` lists, for every image displayed in a figure or quantified for it, the figure panels, the
table that records it (in `results/`), the archive and the path inside the archive, and whether the archived file has
the MD5 recorded when the figure was made. Summary:

{rl.md_table(use)}
{label_txt}
To re-run the imaging scripts ({script_range('93')}), extract the archives with `python code/prepare_workspace.py --images`, which
restores the folders these scripts read.
"""
    tree.text("data/imaging/README.md", text, "Imaging: instruments, contents, naming and use in the figures")
    return dict(n_sources=len(df), same_nested=same_nested)


def st_file(key):
    """(number, path) of the Supplementary Table named Supplementary_Table_<n>_<key>.csv (numbers follow the article)."""
    hits = sorted((RESULTS / "r9_suppl").glob(f"Supplementary_Table_*_{key}.csv"))
    assert hits, f"no Supplementary Table *_{key}.csv in results/r9_suppl"
    return int(re.search(r"Supplementary_Table_(\d+)_", hits[0].name).group(1)), hits[0]


def stage_external(tree, checks):
    n_st, p_st = st_file("public_datasets")
    st3 = pd.read_csv(p_st)
    dl = (S / "40_download_resources.sh").read_text(encoding="utf-8")
    section, sec_of, urls, sections, var = "", {}, [], [], {}
    for ln in dl.split("\n"):
        m = re.match(r"\s*if run (\w+); then", ln)
        if m:
            section = m.group(1)
            sections.append(section)
        m = re.match(r"\s*([A-Z]+)=(https?://\S+)\s*$", ln)
        if m:
            var[m.group(1)] = m.group(2)
        for acc in re.findall(r"\b(GSE\d+)\b", ln) + re.findall(r"platforms/GPL\w+/(GPL\d+)/", ln):
            sec_of.setdefault(acc, section)                  # GPL: platform annotations downloaded as such
        m = re.match(r"\s*get \"?(\S+?)\"? (\S+)", ln)
        if m and section in RESOURCE_SECTIONS:
            url = re.sub(r"\$\{?(\w+)\}?", lambda k: var.get(k.group(1), k.group(0)), m.group(1))
            if url.startswith("http"):
                urls.append((section, url, Path(m.group(2)).name))
    rows = []
    for r in st3.itertuples(index=False):
        d = r._asdict()
        acc = str(d["accession"])
        rows.append(dict(kind="dataset", accession=acc, repository=d["repository"], species=d["species"],
                         tissue=d["tissue"], assay=d["assay"], design=d["design"], use=d["use_in_this_study"],
                         displayed_in=d["displayed_in"], reference=d["reference"], doi_or_url=d["doi_or_url"],
                         in_article="yes", download=f"fetch_public_data.sh ({sec_of[acc]})" if acc in sec_of else
                         "raw reads only (see README)",
                         url=(f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}" if acc.startswith("GSE") else
                              f"https://ngdc.cncb.ac.cn/gsa/browse/{acc}" if acc.startswith("CRA") else "")))
    for acc, sec in sorted(sec_of.items()):
        if acc in set(st3.accession.astype(str)):
            continue
        sp, ti, assay, use = EXTRA_DATASETS.get(acc, ("", "", "", ""))
        use = (f"{use} (exploratory; results tables only, not in the article)" if use else
               "downloaded by the analysis code; not used in the results")
        rows.append(dict(kind="dataset", accession=acc, repository="GEO", species=sp, tissue=ti, assay=assay, design="",
                         use=use, displayed_in="", reference="",
                         doi_or_url=f"https://identifiers.org/geo:{acc}", in_article="no",
                         download=f"fetch_public_data.sh ({sec})",
                         url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}"))
    for sec, url, name in urls:
        rows.append(dict(kind="resource", accession=name, repository=re.sub(r"^https?://([^/]+)/.*$", r"\1", url),
                         species="", tissue="", assay="", design="", use=RESOURCE_SECTIONS[sec], displayed_in="",
                         reference="", doi_or_url="", in_article="Methods", download=f"fetch_public_data.sh ({sec})",
                         url=url))
    for f in ("database/refGene.txt.gz", "bigZips/upstream1000.fa.gz"):
        rows.append(dict(kind="resource", accession=Path(f).name, repository="hgdownload.soe.ucsc.edu", species="mouse",
                         tissue="", assay="", design="", use=RESOURCE_SECTIONS["promoters"], displayed_in="", reference="",
                         doi_or_url="", in_article="Methods", download="fetch_public_data.sh (promoters)",
                         url=f"https://hgdownload.soe.ucsc.edu/goldenPath/mm10/{f}"))
    acc = pd.DataFrame(rows)
    tree.text("data/external/accessions.csv", acc.to_csv(index=False),
              "Public data analysed or used as reference: accession, use, reference and download route")
    # every accession named in the manuscript must be listed
    named = set()
    for f in ("methods.tex", "supplementary.tex", "results.tex", "legends_main.tex", "legends_ed.tex", "legends_si.tex"):
        p = MANUSCRIPT / f
        if p.exists():
            named |= set(re.findall(r"\b(GSE\d{4,}|CRA\d{6}|PRJNA\d+|E-MTAB-\d+)\b", p.read_text(encoding="utf-8")))
    miss = sorted(named - set(acc.accession))
    checks.append(("accessions named in the manuscript are listed", f"{len(named) - len(miss)}/{len(named)}",
                   "pass" if not miss else "warn", ", ".join(miss)))
    fetch = FETCH.replace("@@SECTIONS@@", " ".join(dict.fromkeys(sections)))
    tree.text("data/external/fetch_public_data.sh", fetch, "Downloads the public data into external/ (bash, curl)")
    n_art = int((acc.in_article == "yes").sum())
    text = f"""# Public data

Public data are referenced here, not redistributed. `accessions.csv` lists

- the {n_art} data sets analysed in the article (Supplementary Table {n_st}: accession, species, tissue, assay, design, use,
  figures, reference),
- {int(((acc.kind == 'dataset') & (acc.in_article == 'no')).sum())} further data sets that the analysis code downloads
  for exploratory analyses whose tables are in `results/` but not in the article, and
- {int((acc.kind == 'resource').sum())} reference resources (gene sets, regulons, orthology, motif rankings,
  ligand-target priors, genome annotation and sequences),

with the download route of each (`download`, `url`).

## Download

```bash
bash data/external/fetch_public_data.sh            # everything (about 9 GB)
bash data/external/fetch_public_data.sh geo        # one section; sections: {' '.join(dict.fromkeys(sections))} promoters
```

Files go to `external/` in the project root (`code/scripts/config.yaml`: `paths.external`), in the layout the
scripts read: `external/geo/<accession>/` and `external/resources/`. Downloads resume, and finished files are skipped.

## Independent noise cohort (GSA)

The noise cohort (CRA019820, single-cell; CRA019827, bulk; Genome Sequence Archive, China National Center for
Bioinformation) is available as raw reads. The analyses used STARsolo 2.7.11b matrices (10x Genomics mm10-3.0.0
reference; filtered EmptyDrops-based cell calls) and a STAR gene-count table of the bulk libraries, placed in
`external/ma2025/`; `code/scripts/setup/prep_ma2025.py`, `300_external_sc_ingest.py` and `314_ma2025_bulk.py` describe the
expected layout.
"""
    tree.text("data/external/README.md", text, "Public data: what is listed and how to download it")
    return dict(n_article=n_art, n_listed=len(acc))


FETCH = r"""#!/usr/bin/env bash
# Download the public data analysed in this study (data/external/accessions.csv) into $COCH_ROOT/external, where the
# analysis scripts read them (code/scripts/config.yaml: paths.external). Public data are referenced, not redistributed.
# Usage, from the release folder:   bash data/external/fetch_public_data.sh [section]
#   sections (default: all): @@SECTIONS@@ promoters
# About 9 GB; reruns resume and skip finished files.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export COCH_ROOT=${COCH_ROOT:-$(cd "$HERE/../.." && pwd)}
SECTION=${1:-all}
if [ "$SECTION" != promoters ]; then
  bash "$COCH_ROOT/code/scripts/40_download_resources.sh" "$SECTION"
fi
if [ "$SECTION" = all ] || [ "$SECTION" = promoters ]; then        # promoter motifs (44_promoter_motifs.py)
  RES=$COCH_ROOT/external/resources
  mkdir -p "$RES"
  for f in database/refGene.txt.gz bigZips/upstream1000.fa.gz; do
    out="$RES/$(basename "$f")"
    [ -s "$out" ] || curl -fsSL --retry 5 -C - -o "$out" "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/$f"
  done
fi
cat <<'EOF'
Not downloaded: the independent noise cohort (GSA CRA019820, single-cell; CRA019827, bulk) is public as raw reads at
https://ngdc.cncb.ac.cn/gsa/ . Quantify them with STARsolo 2.7.11b (10x Genomics mm10-3.0.0 reference) and STAR
GeneCounts and place the matrices in external/ma2025/ (see code/scripts/setup/prep_ma2025.py).
EOF
"""


def stage_raw_reads(tree, args):
    if not args.fastq:
        return 0
    src = Path(args.fastq)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    base = src.parent if src.is_file() else src
    rows = []
    for p in files:
        rel = f"data/raw_reads/{p.relative_to(base).as_posix()}"
        tree.copy(p, rel, "Raw sequencing reads")
        rows.append(dict(file=f"`{p.relative_to(base).as_posix()}`", size=rl.human(p.stat().st_size)))
    tree.text("data/raw_reads/README.md", f"""# Raw sequencing reads

Raw reads of the 12 libraries of the atlas (also deposited in GEO: {args.geo or PH['geo']}). MD5 checksums are in
`MANIFEST.tsv` and `MD5SUMS.txt` at the release root.

{rl.md_table(pd.DataFrame(rows))}
""", "Raw sequencing reads: file list")
    return len(files)


# ----------------------------------------------------------------------------------------------------- results
def stage_results(tree, san):
    man = pd.read_csv(RESULTS / "results_manifest.csv")
    desc = dict(zip(man.path, man.description.astype(str)))
    excluded = {f"results/{MOD}/{n}" for n in RELEASE_TABLES} | set(RESULTS_EXCLUDE)
    lits = san.literals() + [b":/", b":\\"]
    red = []
    for p in sorted(RESULTS.rglob("*")):
        if not p.is_file():
            continue
        r = p.relative_to(ROOT).as_posix()
        if r in excluded or r == "results/results_manifest.csv" or p.name.startswith("_"):
            continue
        d = rl.public_text(san.apply(desc.get(r, "Analysis output table"), "text")[0], SUBS) or "Analysis output table"
        data = p.read_bytes()
        low = data.lower()
        if any(x in low for x in lits):
            text, n = san.apply(data.decode("utf-8"), "data")
            if n:
                tree.text(r, text, d)
                red.append(dict(path=r, replacements=n, md5_original=rl.md5sum(p), md5_released=rl.md5sum(tree.root / r)))
                continue
        tree.copy(p, r, d)
    red = pd.DataFrame(red, columns=["path", "replacements", "md5_original", "md5_released"])
    man = man[~man.path.isin(excluded)].copy()
    new = dict(zip(red.path, red.md5_released))
    man["md5"] = [new.get(p, m) for p, m in zip(man.path, man.md5)]
    text, n = san.apply(man.to_csv(index=False), "data")
    tree.text("results/results_manifest.csv", text, "Catalogue of every result table: path, module, rows, columns, "
              "size, MD5, producing script and description")
    return man, red


# ------------------------------------------------------------------------------------------------ supplementary
def stage_supplementary(tree, checks):
    sup = (MANUSCRIPT / "supplementary.tex").read_text(encoding="utf-8")
    titles, grouped = {}, set()
    for a, b, t in re.findall(r"\\item\[Supplementary Tables? (\d+)(?:--(\d+))?\]\s*(.*?)(?=\\item|\\end\{)", sup, re.S):
        for k in range(int(a), int(b or a) + 1):
            titles[k] = rl.tex_to_text(t)
            if b:
                grouped.add(k)
    rows, pub = [], []
    for p in sorted((RESULTS / "r9_suppl").glob("*.csv")):
        m = re.match(r"Supplementary_Table_(\d+)_", p.name)
        k = int(m.group(1)) if m else None
        own = p.stem.split("_", 3)[-1].replace("_", " ")
        own = own[:1].upper() + own[1:]
        d = (f"Supplementary Table {k}: {own if k in grouped or k not in titles else titles[k]}" if k else
             "Supplementary Tables: data dictionary of every column")
        tree.copy(p, f"supplementary/Supplementary_Tables/{p.name}", d)
        rows.append(dict(file=f"`{p.name}`", content=d.split(": ", 1)[-1]))
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        pub += [dict(table=p.name, column=c, value=v) for c, v in residual(df)]
    sd_dir = MANUSCRIPT / "submission" / "source_data"
    sd = sorted(sd_dir.glob("*.zip")) + sorted(sd_dir.glob("*.xlsx"))
    n_csv = 0
    for p in sd:
        m = re.match(r"Source_Data_(ED_Fig|Supplementary_Fig|Fig)(\d+)$", p.stem)
        fig = ({"ED_Fig": "Extended Data Fig.", "Supplementary_Fig": "Supplementary Fig.", "Fig": "Fig."}[m.group(1)]
               + " " + m.group(2)) if m else p.stem
        tree.copy(p, f"supplementary/Source_Data/{p.name}", f"Source Data of {fig} ({p.suffix[1:]})")
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith(".csv"):
                        n_csv += 1
                        with z.open(n) as fh:
                            df = pd.read_csv(fh, dtype=str, keep_default_na=False)
                        pub += [dict(table=f"{p.name}:{n}", column=c, value=v) for c, v in residual(df)]
        else:
            for sh, df in pd.read_excel(p, sheet_name=None, header=None, dtype=str).items():
                df = df.fillna("")
                pub += [dict(table=f"{p.name}:{sh}", column=str(c), value=v) for c, v in residual(df)]
    pub = pd.DataFrame(pub, columns=["table", "column", "value"])
    checks.append(("pubclean residual (Supplementary Tables, Source Data)", len(pub), "pass" if pub.empty else "fail",
                   f"{len(rows)} table files, {len(sd)} Source Data files ({n_csv} CSV tables)"))
    n_x = sum(1 for p in sd if p.suffix == ".xlsx")
    text = f"""# Supplementary Tables and Source Data

## Supplementary Tables

CSV files with a data dictionary that describes every column (`Supplementary_Tables_data_dictionary.csv`).

{rl.md_table(pd.DataFrame(rows))}

## Source Data

`Source_Data/` holds the Source Data of every figure: {sum(1 for p in sd if p.suffix == '.zip')} zip files (one per
figure: a CSV per panel table, an index of panels and files, and the image panels as displayed){f' and {n_x} Excel workbooks (one per figure, a sheet per panel table)' if n_x else ''}. The same tables are in
`results/r9_source_data/`, and the figure tables they are built from in `results/r9_figures/`.
"""
    tree.text("supplementary/README.md", text, "Supplementary Tables and Source Data: contents")
    return pub, len(sd)


# ----------------------------------------------------------------------------------------------------- figures
def figure_list():
    """[(stem, label, title)] of the main, Extended Data and Supplementary figures, in order."""
    pages = (MANUSCRIPT / "figures_pages.tex").read_text(encoding="utf-8")
    titles = {}
    for f in ("legends_main.tex", "legends_ed.tex", "legends_si.tex"):
        p = MANUSCRIPT / f
        if p.exists():
            for lab, t in re.findall(r"\\figlegend\{([^}]*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", p.read_text(encoding="utf-8")):
                titles[rl.tex_to_text(lab)] = rl.tex_to_text(t)
    out = [(Path(f).stem, rl.tex_to_text(lab)) for f, lab in re.findall(r"\\figpage\{([^}]+)\}\{([^}]+)\}",
                                                                          rl.tex_public(pages))]
    si = rl.tex_public((MANUSCRIPT / "legends_si.tex").read_text(encoding="utf-8"))
    for m in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", si):
        lab = re.search(r"\\figlegend\{([^}]*)\}", si[m.end():])
        out.append((Path(m.group(1)).stem, rl.tex_to_text(lab.group(1)) if lab else f"Supplementary Fig. {len(out)}"))
    return [(s, lab, titles.get(lab, "")) for s, lab in out]


def figure_script(stem, key, pi):
    base = re.sub(r"_v\d+$", "", stem)
    hits = [p.name for p in sorted(S.glob("9[1-3]*.py")) if base + "_v" in p.read_text(encoding="utf-8", errors="ignore")]
    if not hits:
        t = pi.loc[pi.figure.astype(str) == key, "templates_recommended_alternative"].astype(str)
        nums = sorted({n for x in t for n in re.findall(r"\b(9\d\d)\b", x)} | ({"902"} if t.str.contains("CF-").any()
                                                                            else set()))
        hits = [p.name for n in nums for p in sorted(S.glob(f"{n}_*.py"))]
    return hits


def resolve_table(name, man_paths):
    name = name.strip()
    if not name or name.startswith("figures/") or "*" in name:
        return name
    if (RESULTS / name).exists():
        return f"results/{name}"
    if (RESULTS / "r9_figures" / name).exists():
        return f"results/r9_figures/{name}"
    hit = [p for p in man_paths if p.endswith("/" + name)]
    return hit[0] if hit else name


def stage_figures(tree, pi, checks):
    figs = figure_list()
    rows, missing = [], []
    for stem, lab, title in figs:
        have = []
        for ext in FIG_EXT:
            p = ROOT / "figures" / "figures" / f"{stem}.{ext}"
            if p.exists():
                tree.copy(p, f"figures/{p.name}", f"{lab}: {title} ({ext.upper()})")
                have.append(ext)
        if "pdf" not in have:
            missing.append(stem)
        key = fig_key(lab)
        rows.append(dict(label=lab, key=key, title=title, stem=stem, formats=", ".join(have),
                         scripts=figure_script(stem, key, pi)))
    checks.append(("figure files", f"{len(figs) - len(missing)}/{len(figs)}", "pass" if not missing else "fail",
                   ", ".join(missing)))
    n_pan = 0
    for p in sorted((ROOT / "figures" / "panels").glob("*")):
        if p.is_file() and not p.name.startswith("_"):
            tree.copy(p, f"figures/panels/{p.name}", "Image tile as displayed in the figures"
                      + (" (16-bit raw crop)" if p.name.endswith("_raw16.tif") else ""))
            n_pan += 1
    if FIG1A.exists():
        tree.copy(FIG1A, "figures/illustrations/fig1a_workflow.png", "Fig. 1a: study-design illustration (input of 910)")
    fr = pd.DataFrame(rows)
    sd = {p.stem.replace("Source_Data_", ""): p.name for p in (MANUSCRIPT / "submission" / "source_data").glob("*.zip")}
    tab = pd.DataFrame(dict(
        Figure=fr.label, Title=fr.title,
        Files=[f"`figures/{s}.*` ({f})" for s, f in zip(fr.stem, fr.formats)],
        Script=[", ".join(f"`{x}`" for x in sc) for sc in fr.scripts],
        **{"Source Data": [f"`{sd.get(sub_name(lab), '')}`" if sd.get(sub_name(lab)) else "" for lab in fr.label]}))
    text = f"""# Figures

Final figure files of the article as vector PDF and SVG, 600-dpi JPG and PNG (where drawn), named as written by
the figure scripts. `panels/` holds every image tile as displayed ({n_pan} files; `*_raw16.tif` are the 16-bit crops
before display scaling), `illustrations/` the study-design illustration of Fig. 1a.

{rl.md_table(tab)}

Each figure script reads the figure tables in `results/r9_figures/` (written by `code/scripts/900_figure_data.py`);
`code/docs/pipeline.md` lists the tables of every panel, and `supplementary/Source_Data/` holds them per figure.
"""
    tree.text("figures/README.md", text, "Figures: files, titles, scripts and Source Data")
    return fr


def sub_name(label: str) -> str:
    """Figure label -> submission name ('Fig4', 'ED_Fig5', 'Supplementary_Fig2')."""
    num = label.split()[-1]
    return f"ED_Fig{num}" if label.startswith("Extended") else (f"Supplementary_Fig{num}" if
                                                               label.startswith("Supplementary") else f"Fig{num}")


# --------------------------------------------------------------------------------------------------- manuscript
def stage_manuscript(tree, san, figs, checks):
    def leaky(s):
        return bool(rl.INTERNAL_ID.search(s)) or san.apply(s, "text")[1] > 0

    srcs = [p for p in sorted(MANUSCRIPT.glob("*.tex")) if p.name not in EXCLUDE_TEX]
    n = 0
    for p in srcs:
        t = rl.tex_public(p.read_text(encoding="utf-8"), leaky)
        t = re.sub(r"\\graphicspath\{\{[^}]*\}\}", lambda m: "\\graphicspath{{figures/}}", t)
        t, k = san.apply(t, "text")
        n += k
        tree.text(f"manuscript/{p.name}", t, f"LaTeX source: {p.stem.replace('_', ' ')}")
    tree.text("manuscript/coch_nn.bib", rl.bib_public((MANUSCRIPT / "coch_nn.bib").read_text(encoding="utf-8")),
              "Bibliography (BibTeX)")
    tree.text("manuscript/latexmkrc", rl.perl_public((MANUSCRIPT / "latexmkrc").read_text(encoding="utf-8")),
              "latexmk settings (pdflatex, output in build/)")
    used = set()
    for p in srcs:
        t = rl.tex_public(p.read_text(encoding="utf-8"))
        used |= {Path(f).stem for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", t)}
        used |= {Path(f).stem for f in re.findall(r"\\figpage\{([^}]+)\}", t)}
    used = {u for u in used if "#" not in u}                 # macro arguments (\figpage definition)
    miss = []
    for stem in sorted(used):
        p = ROOT / "figures" / "figures" / f"{stem}.pdf"
        if p.exists():
            tree.copy(p, f"manuscript/figures/{p.name}", "Figure PDF included by the LaTeX sources")
        else:
            miss.append(stem)
    checks.append(("figures included by the LaTeX sources", f"{len(used) - len(miss)}/{len(used)}",
                   "pass" if not miss else "fail", ", ".join(miss)))
    for doc, d in BUILD_DOCS.items():
        p = MANUSCRIPT / "build" / f"{doc}.pdf"
        if p.exists():
            tree.copy(p, f"manuscript/build/{doc}.pdf", f"Compiled PDF: {d}")
    if (MANUSCRIPT / "build" / "main.bbl").exists():
        tree.copy(MANUSCRIPT / "build" / "main.bbl", "manuscript/build/main.bbl", "Compiled bibliography of main.tex")
    final = MANUSCRIPT / "submission" / "latex_final"
    n_final = 0
    if final.exists():                                    # sources, bibliography, PDFs and images; no build by-products
        for p in sorted(final.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in LATEX_FINAL_KEEP:
                continue
            r = f"manuscript/latex_final/{p.relative_to(final).as_posix()}"
            if p.suffix == ".tex":
                tree.text(r, san.apply(rl.tex_public(p.read_text(encoding="utf-8"), leaky), "text")[0],
                          "Final single-file LaTeX of the article")
            else:
                tree.copy(p, r, "Final LaTeX package of the article" + (" (compiled PDF)" if p.suffix == ".pdf" and
                                                                          p.parent == final else ""))
            n_final += 1
    text = f"""# Manuscript sources

LaTeX sources of the article (pdflatex), the bibliography and the figure PDFs they include. Editorial comments were
removed from the sources; the typeset text is unchanged.

| File | Content |
| --- | --- |
| `main.tex` | article: title, abstract, main text, Methods, references, legends, figure pages (inputs the section files) |
| `supplementary.tex` | Supplementary Information (Supplementary Figures with legends, Supplementary Notes, Supplementary Table legends) |
| `reporting_summary.tex` | draft answers of the Nature Portfolio Reporting Summary |
| `macros.tex` | shared macros |
| `coch_nn.bib` | bibliography |
| `figures/` | the figure PDFs included by `main.tex` and `supplementary.tex` |
| `build/` | compiled PDFs and `main.bbl` |{chr(10) + '| `latex_final/` | single-file LaTeX of the article as submitted |' if n_final else ''}

Build (TeX Live or MiKTeX with the `nature` package, which provides `naturemag.bst`):

```bash
cd manuscript
latexmk main.tex            # and: latexmk supplementary.tex, latexmk reporting_summary.tex
```

`latexmkrc` writes the output to `build/`. Text marked `[AUTHOR: ...]` was still open when this release was built.
"""
    tree.text("manuscript/README.md", text, "Manuscript sources: contents and build")
    return n


# ----------------------------------------------------------------------------------------------- documentation
def pipeline_doc(idx, man, figs, pi):
    man = man.copy()
    man["module"] = [p.split("/")[1] + "/" if p.count("/") >= 2 else "" for p in man.path]
    first = {}
    for s in idx.script:
        m = re.match(r"(\d+)", s)
        first[s] = int(m.group(1)) if m else 999
    out = ["# Pipeline: modules, scripts and outputs", "",
           "Generated from `results/results_manifest.csv` (every registered table with its producing script) and "
           "`results/r9_figures/panel_index.csv` (figure panels and their tables). Scripts are in `code/scripts/`; run "
           "them from the project root (see the README). Purposes come from the script headers.", "",
           "## Script groups", ""]
    g = idx.assign(num=idx.script.map(first)).sort_values("num", kind="stable")
    grp = g.groupby("group", sort=False).agg(scripts=("script", "count"), first=("script", "first"),
                                             last=("script", "last")).reset_index()
    out += [rl.md_table(grp.rename(columns={"group": "Group", "scripts": "Scripts", "first": "First", "last": "Last"})), ""]
    out += ["## Scripts and their outputs", ""]
    by_script = man.groupby("script")
    for row in g.itertuples():
        out.append(f"### `{row.script}`")
        out.append("")
        out.append(row.purpose or "(no description)")
        if row.script in by_script.groups:
            t = by_script.get_group(row.script)
            mods = ", ".join(f"`results/{m}`" for m in dict.fromkeys(t.module))
            names = [Path(p).name for p in t.path]
            more = f" and {len(names) - 12} more" if len(names) > 12 else ""
            out.append("")
            out.append(f"Outputs ({len(names)}) in {mods}: " + ", ".join(f"`{n}`" for n in names[:12]) + more + ".")
        out.append("")
    out += ["## Figures, panels and tables", ""]
    man_paths = list(man.path)
    for stem, lab, title in figs:
        key = fig_key(lab)
        sub = pi[pi.figure.astype(str) == key]
        out += [f"### {lab}: {title}", "", f"File: `figures/{stem}.*`; script: "
                + ", ".join(f"`{s}`" for s in figure_script(stem, key, pi)), ""]
        if len(sub):
            rows = []
            for r in sub.itertuples():
                files = [resolve_table(x, man_paths) for x in str(r.files).split(";") if x.strip()]
                rows.append(dict(Panel=r.panel, Content=rl.public_text(r.message, SUBS),
                                 Tables=", ".join(f"`{f}`" for f in files)))
            out += [rl.md_table(pd.DataFrame(rows)), ""]
    return "\n".join(out) + "\n"


def pt(s, san=None) -> str:
    """public_text that keeps the case of the first letter."""
    s = str(s)
    out = rl.public_text(san.apply(s, "text")[0] if san else s, SUBS)
    return out[:1].lower() + out[1:] if s[:1].islower() else out


def glossary(san):
    reg = pd.read_csv(RESULTS / "r2_audit" / "positive_findings_register.csv")
    reg["category"] = reg.category.map(pt)
    reg["prefix"] = reg.finding_id.str[0]
    pref = reg.groupby("prefix")["category"].agg(lambda s: ", ".join(dict.fromkeys(s.astype(str))))
    counts = reg.groupby("prefix").size()
    ptab = pd.DataFrame(dict(Prefix=[f"`{p}`" for p in pref.index], Findings=[int(counts[p]) for p in pref.index],
                             Category=list(pref.values)))
    ftab = pd.DataFrame(dict(ID=reg.finding_id, Category=reg.category, Tissue=reg.tissue,
                             Statement=[pt(s, san) for s in reg.statement],
                             Verdict=reg.verdict))
    R = CFG["rhythm"]
    n_ct, p_ct = st_file("cell_types")
    n_ev, _ = st_file("evidence_grading")
    ct = pd.read_csv(p_ct)
    return f"""# Glossary of identifiers in the result tables

Result tables are written for analysis, so some use short identifiers. This glossary explains them.

## Time, tissues and samples

- **ZT** (Zeitgeber time): hours after lights-on (lights on at 08:00, 12 h light : 12 h dark). **Clock time** = ZT + 8 h.
  Samples were taken at ZT3, 7, 11, 15, 19 and 23. Columns named `zt`, `peak_zt` or `*_zt` are ZT; `clock`,
  `peak_clock` or `day_time` are clock time.
- **Tissues**: `cochlea`; `bm` or `bone marrow` = the temporal-bone marrow next to the cochlea.
- **Samples**: `sample_id` / `batch_id` = one pooled library per tissue and time point (12 in total).
- **Cell labels**: `cell_class` (12 classes), `cell_type` (35 subtypes) and `cell_type_abbr` (abbreviations), the
  reference annotation of the atlas (Supplementary Table {n_ct} and below).

## Rhythm tiers

- **Tier 1**: limma cosinor BH q < {R['q_tier1']}, permutation q < {R['perm_fdr_tier1']}, peak-to-trough fold change >=
  {R['fc_threshold']}, at least one independent method with P < 0.05, leave-one-time-point-out phase SD <
  {R['loto_phase_sd_h']} h, and not driven by a single sample.
- **Tier 2**: limma BH q < {R['q_tier2']} and fold change >= {R['fc_threshold']} (not Tier 1).
- **Tier C / exploratory**: nominal P < 0.05 without either tier.

## Evidence register

Every positive finding of the analyses has an identifier (letter + two digits) and a verdict from pre-specified
evidence dimensions (`results/r2_audit/positive_findings_register.csv`, `validation_matrix.csv`; Supplementary Table
{n_ev} of the article):

- **robust**: all core dimensions support, at least two independent dimensions support and at most one is against;
- **supported**: no core dimension against, at least one core and one independent dimension support, and independent
  support outweighs opposition;
- **not supported**: no core support and independent opposition at least as strong as support;
- **suggestive**: everything else; **retired**: withdrawn finding kept for the record.

{rl.md_table(ptab)}

{rl.md_table(ftab)}

## Hypotheses, grades and templates

- **H1-H12**: the pre-specified B-lineage hypotheses (scripts 510-570; decision: supported, partial or not supported).
- **M1-M9**: the pre-specified myeloid hypotheses (scripts 610-680; same decision categories).
- **E0-E2**: exploratory grades of the pre-specified claims (680): E2 exploratory support, E1 qualified exploratory
  support, E0 no exploratory support.
- **T1-T3**: timing templates of the independent noise cohort: T1 our cochlear fits, T2 the independent sham-cochlea
  series (primary), T3 our structural cell types.
- **Flow cytometry**: tubes T1-T8 (markers in `data/flow/channels.csv`); readouts P1 (intermediate neutrophils), P2
  (classical monocytes), P3 (B220-high IgD+ B cells).

## Internal references

Some descriptions and notes cite the authors' internal analysis log as `D<number>` (decisions), `I<number>`
(interpretations) or `O<number>` (open items). The log is not part of this release; these references carry no
information needed to use the tables. A number in parentheses, such as `(77)`, names the producing script
(`code/scripts/77_*.py`); folders named `r2_` to `r10_` hold the validation and follow-up analyses (see
`code/docs/pipeline.md`).

## Cell types

{rl.md_table(ct)}
"""


def results_readme(man, red):
    man = man.copy()
    man["module"] = [p.split("/")[1] + "/" if p.count("/") >= 2 else "(top level)" for p in man.path]
    mods = (man.groupby("module", sort=True)
            .agg(tables=("path", "count"), scripts=("script", lambda s: ", ".join(dict.fromkeys(s))))
            .reset_index())
    mods["scripts"] = mods["scripts"].str.slice(0, 120)
    red_txt = ""
    if len(red):
        names = [f"`{p.split('/', 1)[1]}`" for p in red.path]
        red_txt = ("\n\nIn " + (", ".join(names[:-1]) + " and " if len(names) > 1 else "") + names[-1] + ", machine "
                   "names and server paths were replaced by `<redacted>` or relative paths; `results_manifest.csv` "
                   "gives the MD5 of the released files.")
    if RESULTS_EXCLUDE:
        red_txt += ("\n\nNot released: " + "; ".join(f"`{p.split('/', 1)[1]}` (a copy of the authors' internal list of "
                                                      f"data sets; see `{v}`)" for p, v in RESULTS_EXCLUDE.items()) + ".")
    n_ev, _ = st_file("evidence_grading")
    return f"""# Analysis results

Every table written by the analysis scripts ({len(man):,} CSV files). `results_manifest.csv` catalogues them: path,
module (folder), rows, columns (names), size, MD5, producing script and a description. `GLOSSARY.md` explains the
identifiers used in the tables (time, tissues, rhythm tiers, evidence register, hypotheses).{red_txt}

Start points:

- `key_findings.csv`: the key numbers of the analyses with their source tables;
- `r3_conclusions.csv`: claims, verdicts and the tables behind them;
- `r2_audit/positive_findings_register.csv`: the evidence register (Supplementary Table {n_ev});
- `r9_figures/`: the tables behind every figure panel (`panel_index.csv` maps panels to tables);
- `r9_suppl/`, `r9_source_data/`: Supplementary Tables and Source Data as released with the article.

## Folders

{rl.md_table(mods.rename(columns={"module": "Folder", "tables": "Tables", "scripts": "Scripts"}))}
"""


def root_readme(args, info, figs_tab, idx, tree, big, pi):
    title = rl.tex_to_text(re.search(r"\\title\{([^{}]+)\}", (MANUSCRIPT / "main.tex").read_text(encoding="utf-8"))
                           .group(1))
    abstract = rl.tex_to_text((MANUSCRIPT / "abstract.tex").read_text(encoding="utf-8"))
    jname = f"{JOURNAL.get('name', 'journal')} {JOURNAL.get('article_type', 'Article')}"
    sizes = {k: rl.human(tree.size(k)) for k in ("code/", "data/processed/", "data/flow/", "data/imaging/",
                                                  "data/external/", "results/", "supplementary/", "figures/",
                                                  "manuscript/")}
    folders = pd.DataFrame([
        ("`code/`", "analysis scripts (`scripts/`), environments (`env/`), pipeline documentation (`docs/`), workspace "
                    "helper", sizes["code/"]),
        ("`data/processed/`", f"the single-cell atlas (h5ad, {info['atlas']['n_cells']:,} cells) and its schema",
         sizes["data/processed/"]),
        ("`data/flow/`", f"flow-cytometry export of the validation cohort ({info['flow']['mice']} mice)",
         sizes["data/flow/"]),
        ("`data/imaging/`", "original image archives: 3D renders, cochlear fluids, parabiosis survey",
         sizes["data/imaging/"]),
        ("`data/external/`", "accessions of the public data and a download script", sizes["data/external/"]),
        ("`results/`", "every analysis table, its catalogue and a glossary", sizes["results/"]),
        ("`supplementary/`", "Supplementary Tables and Source Data", sizes["supplementary/"]),
        ("`figures/`", "final figure files and image tiles", sizes["figures/"]),
        ("`manuscript/`", "LaTeX sources and compiled PDFs", sizes["manuscript/"])],
        columns=["Folder", "Contents", "Size"])
    if info.get("raw_reads"):
        folders.loc[len(folders)] = ("`data/raw_reads/`", "raw sequencing reads", rl.human(tree.size("data/raw_reads/")))
    g = idx.assign(num=idx.script.map(lambda s: int(re.match(r"(\d+)", s).group(1)) if re.match(r"\d", s) else 999))
    order = (g.sort_values("num", kind="stable").groupby("group", sort=False)
             .agg(first=("script", "first"), last=("script", "last"), n=("script", "count")).reset_index())
    order["Scripts"] = [f"`{a.split('_')[0]}`-`{b.split('_')[0]}` ({n})" for a, b, n in
                        zip(order["first"], order["last"], order["n"])]
    order = order.rename(columns={"group": "Group"})[["Group", "Scripts"]]
    sd = {p.stem.replace("Source_Data_", ""): p.name for p in (MANUSCRIPT / "submission" / "source_data").glob("*.zip")}
    n_tab = {}
    for key, files in zip(pi.figure.astype(str), pi.files.astype(str)):
        n_tab.setdefault(key, set()).update(x.strip() for x in files.split(";")
                                            if x.strip() and not x.strip().startswith("figures/"))
    ftab = pd.DataFrame(dict(
        Figure=[f"{lab}: {t}" for lab, t in zip(figs_tab.label, figs_tab.title)],
        File=[f"`figures/{s}.pdf`" for s in figs_tab.stem],
        Script=[", ".join(f"`{x}`" for x in sc) for sc in figs_tab.scripts],
        Tables=[len(n_tab.get(fig_key(lab), ())) for lab in figs_tab.label],
        **{"Source Data": [f"`{sd[sub_name(lab)]}`" if sub_name(lab) in sd else "" for lab in figs_tab.label]}))
    big_txt = ""
    if big:
        big_txt = ("Files above the per-file limit of the repository (" + ", ".join(f"`{b}`" for b in big)
                   + ") are uploaded in byte parts (`.001`, `.002`, ...); join them first (`cat file.0* > file` on "
                   "Linux or macOS, `copy /b file.001+file.002 file` on Windows). ")
    contact = args.contact or PH["contact"]
    return f"""# {title}

Data, code and analysis results of the article "{title}", a {jname} (citation: {args.article or PH['article']}).
Release v{args.version}, built {time.strftime('%Y-%m-%d')}. Code repository: {args.github or PH['github']}. This
release: {args.doi or PH['doi']}.

## The study

{abstract}

The atlas comprises {info['atlas']['n_cells']:,} cells of {info['atlas']['n_samples']} samples: the cochlea and the
adjacent temporal-bone marrow at six times of day (ZT3, ZT7, ZT11, ZT15, ZT19 and ZT23; 12 h light : 12 h dark, lights on
at 08:00), one pool of three mice per tissue and time point, annotated into {info['atlas']['n_types']} cell types of
{info['atlas']['n_classes']} classes. A pre-specified flow-cytometry cohort ({info['flow']['mice']} mice, {info['flow']['zts']}),
confocal and three-dimensional imaging, and a parabiosis survey complement the atlas, together with
{info['external']['n_article']} public data sets.

## Contents

{rl.md_table(folders)}

Root files: `README.md` (this file), `LICENSE-code`, `LICENSE-data`, `CITATION.cff`, `MANIFEST.tsv` (every file with its
size, MD5 and a description) and `MD5SUMS.txt`.

## Quick start

After downloading, unzip the archives into one folder. {big_txt}Then check every file:

```bash
cd {NAME}_v{args.version}
md5sum -c MD5SUMS.txt                      # Windows: CertUtil -hashfile <file> MD5
```

Explore the atlas in Python (anndata 0.11 or later):

```python
import anndata as ad

adata = ad.read_h5ad("data/processed/coca_circadian.h5ad")     # raw UMI counts and cell annotations
adata.obs["zt"] = (adata.obs["day_time"].astype(float) - 8) % 24  # Zeitgeber time (h); clock time = ZT + 8
print(adata.obs.groupby(["tissue", "zt"]).size())
```

Find a result: `results/results_manifest.csv` describes every table; `results/key_findings.csv` and
`results/r3_conclusions.csv` lead from the claims of the article to the tables behind them; `results/GLOSSARY.md`
explains the identifiers. The data behind every figure panel are in `supplementary/Source_Data/` and
`results/r9_figures/`.

## Reproducing the analyses

The analyses ran on a Linux server (Python and R; 24 CPU cores, about 90 GB memory, one GPU for scVI and scANVI) and,
for imaging, flow cytometry, figures, tables and the manuscript, on a workstation. Paths are relative to the project
root, which is this folder.

1. Environments: `code/env/` (conda specifications pinned to the versions used; see `code/env/README.md`).
2. Workspace: `python code/prepare_workspace.py` links `scripts/` to `code/scripts/` (the scripts and chains expect
   `scripts/` in the project root); add `--flow` and `--images` to unpack the flow export and the image archives to the
   folders the imaging and flow scripts ({script_range('93')}) read (about 60 GB for the images).
3. Public data: `bash data/external/fetch_public_data.sh` downloads them to `external/` (about 9 GB).
4. Run from the project root with `export COCH_ROOT=$PWD`: `python scripts/<script>.py` or
   `bash scripts/setup/rjob.sh scripts/<script>.R`. Every script reads `scripts/config.yaml`; the shell chains in
   `scripts/setup/` read `COCH_ROOT`, `COCH_PYTHON`, `COCH_RSCRIPT` and `COCH_ENVS`.
5. Order: run the groups below in order, the scripts of a group in numeric order; `scripts/setup/chain_*.sh` record the
   order used. After any analysis, `scripts/setup/chain_tail.sh` rebuilds the figure index, evidence register, audits,
   key findings and the results catalogue. Every table is written through `scripts/lib/io.py`, which registers it
   (`results/results_manifest.csv`).
6. Figures, tables and Source Data: `900_figure_data.py` writes the figure tables, {script_range('91', '`')} and {script_range('92', '`')} draw the
   figures, {script_range('93', '`')} the imaging and flow panels, `908` and `909` write the Supplementary Tables and
   Source Data.

The released `results/` let you start at any step: figure scripts, for example, need only `results/` and the
environments.

{rl.md_table(order)}

`code/docs/pipeline.md` lists every script with its outputs and every figure panel with its tables;
`code/docs/scripts_index.csv` gives the purpose of each script.

## Figures, scripts and data

The script in each row draws the figure from its figure tables (`Tables`: their number; most are in
`results/r9_figures/`, and `code/docs/pipeline.md` lists them panel by panel). The Source Data zip of each figure
(`supplementary/Source_Data/`) holds the same tables per panel.

{rl.md_table(ftab)}

## Data

- **Single-cell atlas** (`data/processed/`): AnnData file with raw UMI counts ({info['atlas']['n_genes']:,} genes) and
  the reference annotation; the README there describes every field. Platforms: {info['atlas']['platforms']}.
- **Raw sequencing reads**: {args.geo or PH['geo']}{' (also in `data/raw_reads/`)' if info.get('raw_reads') else ''}.
- **Flow cytometry** (`data/flow/`): every event ({info['flow']['events']} events) with scatter, two fluorescence
  channels and fixed gate flags, per-tube and per-mouse tables, tube panels and gates, and the original export.
- **Imaging** (`data/imaging/`): the original image archives (ZEISS LSM 900 CZI series and 3D renders; Olympus FV3000
  60x images of the cochlear fluids and 54-day parabiont sections; the parabiosis survey with its own index);
  `figure_image_sources.csv` maps each figure panel to the archived files.
- **Public data** (`data/external/`): {info['external']['n_listed']} accessions and resources with their use and a
  download script; public data are not redistributed.

## Licences

- Code (`code/`): Apache License 2.0 (`LICENSE-code`).
- Data, results, figures, images, Supplementary Tables, Source Data and documentation: Creative Commons Attribution 4.0
  International (`LICENSE-data`).
- Manuscript sources (`manuscript/`): the authors' version of the article; [AUTHOR: licence of the manuscript sources].

## Citation

Please cite the article and this release (`CITATION.cff`): {args.article or PH['article']}.

## Contact

{contact}
"""


def license_data():
    return """Creative Commons Attribution 4.0 International (CC BY 4.0)

The data, analysis results, figures, images, Supplementary Tables, Source Data and documentation in this release
(everything except the code in code/, which is licensed under LICENSE-code, and the manuscript sources in manuscript/)
are licensed under the Creative Commons Attribution 4.0 International License.

You are free to share (copy and redistribute the material in any medium or format) and to adapt (remix, transform and
build upon the material) for any purpose, even commercially, provided that you give appropriate credit, provide a link
to the license and indicate if changes were made.

Licence summary: https://creativecommons.org/licenses/by/4.0/
Legal code:      https://creativecommons.org/licenses/by/4.0/legalcode

Attribution: please cite the article and this release (see CITATION.cff).
"""


def citation_cff(args, title, abstract):
    def authors():
        if not args.authors:
            return [f'  - name: "{PH["authors"]}"']
        out = []
        for a in [x.strip() for x in args.authors.split(";") if x.strip()]:
            fam, _, giv = a.partition(",")
            out += [f'  - family-names: "{fam.strip()}"'] + ([f'    given-names: "{giv.strip()}"'] if giv.strip() else [])
        return out

    lines = ["cff-version: 1.2.0", 'message: "If you use these data or this code, please cite the article and this '
             'release."', f'title: "{title}: data, code and analysis results"', "type: dataset",
             f'version: "{args.version}"', f'date-released: "{time.strftime("%Y-%m-%d")}"',
             "license:", "  - Apache-2.0", "  - CC-BY-4.0", "authors:", *authors(),
             f'abstract: "{abstract.replace(chr(34), chr(39))}"',
             "keywords:", *[f'  - "{k}"' for k in ("circadian", "cochlea", "bone marrow", "single-cell RNA sequencing",
                                                   "neutrophils", "inner ear", "immunology")]]
    if args.github:
        lines.append(f'repository-code: "{args.github}"')
    if args.doi:
        lines += ["identifiers:", "  - type: doi", f'    value: "{args.doi}"']
    lines += ["preferred-citation:", "  type: article", f'  title: "{title}"', "  authors:",
              *["  " + ln for ln in authors()], f'  journal: "{JOURNAL.get("name", "")}"',
              f'  notes: "{args.article or PH["article"]}"']
    return "\n".join(lines) + "\n"


PREPARE = r'''"""Prepare this release for re-running the analysis scripts (Python standard library only).

The scripts run from the project root (this release folder) and expect scripts/ there: R scripts source
scripts/lib/cosinor.R and the shell chains call scripts/... The imaging scripts (93x) read folders below data/photos/
and the flow scripts data/flow_cochlea/. This helper
  1. links scripts/ to code/scripts (symbolic link; a directory junction or a copy on Windows),
  2. with --flow: copies the flow export to data/flow_cochlea/ and data/Cochlear_Flow_Plotting_Only.zip,
  3. with --images: extracts the image archives to the folders the imaging scripts read (about 60 GB),
  4. copies the Fig. 1a illustration to data/photos/fig1a_workflow/ (read by 910).
Usage, from the release folder: python code/prepare_workspace.py [--flow] [--images]
"""
import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(os.environ.get("COCH_ROOT", Path(__file__).resolve().parents[1]))
IMAGES = @@IMAGES@@
FLOW_FILES = @@FLOW@@
FLOW_ARCHIVE = "data/flow/Cochlear_Flow_Plotting_Only.zip"
FIG1A = ("figures/illustrations/fig1a_workflow.png", "data/photos/fig1a_workflow/fig1a_workflow_pasted.png")


def link_scripts():
    dst, src = ROOT / "scripts", ROOT / "code" / "scripts"
    if dst.exists():
        print("scripts/ exists")
        return
    try:
        dst.symlink_to(Path("code") / "scripts", target_is_directory=True)
    except OSError:
        if not (os.name == "nt" and subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                                                   capture_output=True).returncode == 0):
            shutil.copytree(src, dst)
    print("scripts/ ->", src)


def extract(archive, inner, target):
    target = (ROOT / target).resolve()
    n = 0
    with zipfile.ZipFile(ROOT / archive) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith(inner):
                continue
            rel = PurePosixPath(info.filename[len(inner):])
            if ".." in rel.parts or rel.is_absolute():
                continue
            out = target / Path(*rel.parts)
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 22)
            n += 1
    print(f"{archive} -> {target}: {n} files extracted")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flow", action="store_true", help="copy the flow export to data/flow_cochlea")
    ap.add_argument("--images", action="store_true", help="extract the image archives (about 60 GB)")
    a = ap.parse_args()
    link_scripts()
    src, dst = (ROOT / FIG1A[0], ROOT / FIG1A[1])
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if a.flow:
        for f in FLOW_FILES:
            d = ROOT / "data" / "flow_cochlea" / f
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "data" / "flow" / f, d)
        shutil.copy2(ROOT / FLOW_ARCHIVE, ROOT / "data" / Path(FLOW_ARCHIVE).name)
        print("flow export -> data/flow_cochlea")
    if a.images:
        for target, archive, inner in IMAGES:
            extract(archive, inner, target)


if __name__ == "__main__":
    main()
'''


def stage_docs(tree, args, info, idx, man, red, figs_tab, figs, pi, san, big):
    title = rl.tex_to_text(re.search(r"\\title\{([^{}]+)\}", (MANUSCRIPT / "main.tex").read_text(encoding="utf-8"))
                           .group(1))
    abstract = rl.tex_to_text((MANUSCRIPT / "abstract.tex").read_text(encoding="utf-8"))
    tree.text("code/docs/pipeline.md", pipeline_doc(idx, man, figs, pi),
              "Pipeline: module -> script -> outputs; figure -> panels -> tables")
    flow_files = sorted(p.name for p in FLOW_DIR.iterdir() if p.is_file())
    prep = PREPARE.replace("@@IMAGES@@", "[\n" + "".join(f"    {t!r},\n" for t in WORKSPACE_MAP) + "]").replace(
        "@@FLOW@@", repr(flow_files))
    tree.text("code/prepare_workspace.py", prep, "Workspace helper: links scripts/, unpacks the image and flow inputs")
    tree.text("results/GLOSSARY.md", glossary(san), "Glossary of the identifiers in the result tables")
    tree.text("results/README.md", results_readme(man, red), "Results: contents and start points")
    tree.text("LICENSE-code", rl.APACHE_2_0, "Licence of the code: Apache License 2.0")
    tree.text("LICENSE-data", license_data(), "Licence of the data and documents: CC BY 4.0")
    tree.text("CITATION.cff", citation_cff(args, title, abstract), "Citation metadata (Citation File Format)")
    tree.text("README.md", root_readme(args, info, figs_tab, idx, tree, big, pi), "Overview of the release")


# ------------------------------------------------------------------------------------------------------- checks
def latex_check(tree, checks, work: Path):
    """Compile the released LaTeX sources and the project sources in throwaway copies (latexmk) and compare the typeset
    text page by page: the release removes comments only, so the text must be identical."""
    try:
        import pymupdf as fitz                             # PyMuPDF >= 1.24
    except ImportError:
        import fitz
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(tree.root / "manuscript", work / "release", ignore=shutil.ignore_patterns("build"))
    proj = work / "project" / "manuscript" / "ni"
    proj.mkdir(parents=True)
    for p in MANUSCRIPT.iterdir():
        if p.is_file() and (p.suffix in (".tex", ".bib") or p.name == "latexmkrc"):
            shutil.copy2(p, proj / p.name)
    figdir = work / "project" / "figures" / "figures"     # the project's graphicspath (../../figures/figures/)
    figdir.mkdir(parents=True)
    for p in (tree.root / "manuscript" / "figures").glob("*.pdf"):
        shutil.copy2(p, figdir / p.name)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(x for x in env["PATH"].split(os.pathsep) if "tensorflow_jni" not in x)

    def build(d, doc):
        r = subprocess.run(["latexmk", "-g", "-interaction=nonstopmode", "-halt-on-error", f"{doc}.tex"], cwd=d,
                           env=env, capture_output=True, text=True)
        lg = (d / "build" / f"{doc}.log").read_text(encoding="utf-8", errors="replace")
        bad = [ln for ln in lg.split("\n") if ln.startswith("!")]
        bad += sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' on page \d+ undefined", lg)))
        pdf = d / "build" / f"{doc}.pdf"
        if r.returncode or not pdf.exists():
            return r.returncode or 1, bad, 0, ""
        with fitz.open(pdf) as f:
            return r.returncode, bad, f.page_count, "".join(pg.get_text() for pg in f)

    for doc in BUILD_DOCS:
        if not (tree.root / "manuscript" / f"{doc}.tex").exists():
            continue
        a, b = build(proj, doc), build(work / "release", doc)
        ok = a[0] == 0 and b[0] == 0 and not b[1] and a[3] == b[3]
        checks.append((f"LaTeX: released {doc}.tex compiles like the project source", f"{b[2]} pages",
                       "pass" if ok else "fail", "; ".join(b[1][:3]) or ("" if a[3] == b[3] else "typeset text differs")))
    shutil.rmtree(work, ignore_errors=True)


def code_checks(tree, checks):
    bad_py, n_py, bad_sh, n_sh = [], 0, [], 0
    for r in sorted(tree.desc):
        p = tree.root / r
        if r.endswith(".py"):
            n_py += 1
            try:
                compile(p.read_text(encoding="utf-8"), r, "exec")
            except SyntaxError as e:
                bad_py.append(f"{r}: {e.msg} (line {e.lineno})")
        elif r.endswith(".sh"):
            n_sh += 1
            res = subprocess.run(["bash", "-n", p.as_posix()], capture_output=True, text=True)
            if res.returncode:
                bad_sh.append(f"{r}: {res.stderr.strip()[:120]}")
    checks.append(("released Python files compile", f"{n_py - len(bad_py)}/{n_py}", "pass" if not bad_py else "fail",
                   "; ".join(bad_py[:5])))
    checks.append(("released shell scripts pass bash -n", f"{n_sh - len(bad_sh)}/{n_sh}",
                   "pass" if not bad_sh else "fail", "; ".join(bad_sh[:5])))


def scan(tree, san, args):
    allow = [m.group(0) for m in re.finditer(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", args.contact or "")]
    pats = rl.leak_patterns(san, allow)
    lits = san.literals()
    docs = {r for r in tree.desc if r.endswith((".md", ".cff")) and not r.startswith(("data/flow/", "code/scripts/"))}
    docs |= {r for r in tree.desc if r.startswith("code/docs/")} | {"LICENSE-data"}
    rows = []
    for r in sorted(tree.desc):
        p = tree.root / r
        suf = p.suffix.lower()
        if r.startswith("data/imaging/") and suf == ".zip" or r == f"data/flow/{FLOW_ZIP.name}":
            continue                                     # original data archives: not opened
        if suf in rl.TEXT or suf == ".r":
            for lab, sev, line, ex in rl.scan_file(p, pats + (rl.DOC_PATTERNS if r in docs else [])):
                rows.append(dict(path=r, line=line, check=lab, severity=sev, excerpt=ex))
        else:
            for x in rl.scan_bytes(p, lits):
                rows.append(dict(path=r, line=0, check="literal in binary file", severity="leak", excerpt=x))
    for rel in ("MANIFEST.tsv",):
        if (tree.root / rel).exists():
            for lab, sev, line, ex in rl.scan_file(tree.root / rel, pats + rl.DOC_PATTERNS[:1]):
                rows.append(dict(path=rel, line=line, check=lab, severity=sev, excerpt=ex))
    return pd.DataFrame(rows, columns=["path", "line", "check", "severity", "excerpt"])


# ----------------------------------------------------------------------------------------------------- archives
def archives(tree, args, upload: Path, limit: int):
    upload.mkdir(parents=True, exist_ok=True)
    prefix = tree.root.name
    files = sorted(tree.desc) + ["MANIFEST.tsv", "MD5SUMS.txt"]
    groups = {g: [] for g, _ in UPLOAD_GROUPS}
    for r in files:
        g = next((g for g, pre in UPLOAD_GROUPS[1:] if r.startswith(pre)), "core")
        groups[g].append((r, (tree.root / r).stat().st_size))
    plan, rows = [], []
    k = 0
    for g, _ in UPLOAD_GROUPS:
        if not groups[g]:
            continue
        k += 1
        vols, big = rl.plan_volumes(groups[g], limit)
        for i, v in enumerate(vols, 1):
            name = f"{prefix}_{k:02d}_{g}" + (f"_part{i}" if len(vols) > 1 else "") + ".zip"
            plan.append(("zip", name, v, g))
        for rel, size in big:
            plan.append(("parts", f"{prefix}_{k:02d}_{g}_{Path(rel).name}", [rel], g))
    produced = set()
    for kind, name, rels, g in plan:
        if kind == "zip":
            dst = upload / name
            if not rl.zip_current(dst, tree.root, rels, prefix):
                log(f"writing {name} ({len(rels)} files)")
                rl.write_zip(dst, tree.root, rels, prefix)
            bad = rl.check_zip(dst, tree.root, rels, prefix, crc=args.crc)
            rows.append(dict(file=name, kind="zip", size_bytes=dst.stat().st_size, md5=rl.md5sum(dst), n_files=len(rels),
                             contents=summarise(rels), joins_to="", check="ok" if not bad else "; ".join(bad[:3])))
            produced.add(name)
        else:
            src = tree.root / rels[0]
            parts = rl.split_parts(src, upload, name, limit)
            joined = rl.md5_of_parts(parts)
            want = tree.md5.get(rels[0]) if hasattr(tree, "md5") else rl.md5sum(src)
            for p in parts:
                rows.append(dict(file=p.name, kind="part", size_bytes=p.stat().st_size, md5=rl.md5sum(p), n_files=1,
                                 contents=f"byte part {p.suffix[1:]} of {len(parts)} of {rels[0]}",
                                 joins_to=f"{prefix}/{rels[0]}",
                                 check="ok" if joined == want else "joined MD5 differs"))
                produced.add(p.name)
    for p in upload.iterdir():                         # leftovers of earlier runs
        if p.is_file() and p.name.startswith(NAME + "_v") and p.name not in produced:
            p.unlink()
    arc = pd.DataFrame(rows)
    return arc


def summarise(rels):
    """Files per top folder ('data/<x>' for data), e.g. 'code (242); figures (165)'."""
    tops = {}
    for r in rels:
        parts = r.split("/")
        key = "root files" if len(parts) == 1 else ("/".join(parts[:2]) if parts[0] == "data" else parts[0])
        tops[key] = tops.get(key, 0) + 1
    return "; ".join(f"{k}/ ({n:,} files)" if k != "root files" else f"{k} ({n})" for k, n in tops.items())


def upload_readme(arc, args, limit, prefix):
    tab = arc[["file", "size_bytes", "md5", "contents"]].copy()
    tab["size"] = tab.size_bytes.map(rl.human)
    tab = tab[["file", "size", "md5", "contents"]]
    parts = arc[arc.kind == "part"]
    join = ""
    for target, g in parts.groupby("joins_to", sort=False):
        names = list(g.file)
        join += (f"\n```bash\ncat {' '.join(names)} > {target}\n```\n\nWindows: `copy /b {'+'.join(names)} "
                 f"{target.replace('/', chr(92))}`\n")
    return f"""# Upload files of {prefix}

{len(arc)} files, each at most {limit / 1e9:g} GB (1 GB = 10^9 bytes); together {rl.human(arc.size_bytes.sum())}.

{rl.md_table(tab)}

To rebuild the release, unzip every `.zip` into one folder (they share the top folder `{prefix}/`), then join the
byte parts:
{join}
and check every file with `md5sum -c MD5SUMS.txt` inside `{prefix}/`. `MD5SUMS.txt` in this folder lists the upload
files themselves.
"""


# ---------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Build the public release (see the module docstring).")
    ap.add_argument("--out", default=str(ROOT / "release"))
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--server", help="ssh host of the analysis server (atlas source; redacted from released text)")
    ap.add_argument("--h5ad", help="atlas source: local path or host:path (default: --server and config paths)")
    ap.add_argument("--fastq", help="file or folder of raw reads to include as data/raw_reads/")
    ap.add_argument("--limit-gb", type=float, default=10.0)
    ap.add_argument("--skip-archives", action="store_true")
    ap.add_argument("--crc", action="store_true", help="CRC-test every upload zip")
    ap.add_argument("--latex-check", action="store_true",
                    help="compile the released and the project LaTeX sources (latexmk) and compare the typeset text")
    ap.add_argument("--stage-submission", action="store_true")
    for k in ("geo", "doi", "github", "article", "authors", "contact"):
        ap.add_argument(f"--{k}")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    tree = Tree(out / f"{NAME}_v{args.version}")
    tree.root.mkdir(parents=True, exist_ok=True)
    san = sanitizer(args)
    checks = []
    idx, lv = index(san)
    log(f"scripts indexed: {len(idx)}")
    n_sub, changes = stage_code(tree, san, idx, lv)
    log(f"code staged ({n_sub} path / host replacements)")
    write_csv(changes, MOD, "release_code_changes.csv", "Public release: every line of the released code that the "
              "sanitiser changed, before and after (audit; 904)", required=["path", "line", "before", "after"])
    if args.stage_submission:
        stage_submission(tree, idx, lv, args, san, checks)
    pi = load_panel_index({fig_key(lab) for _, lab, _ in figure_list()})     # published figure keys only
    info = {"atlas": stage_atlas(tree, args, checks)}
    info["flow"] = stage_flow(tree, checks)
    info["imaging"] = stage_imaging(tree, pi, checks)
    info["external"] = stage_external(tree, checks)
    info["raw_reads"] = stage_raw_reads(tree, args)
    log("data staged")
    man, red = stage_results(tree, san)
    log(f"results staged ({len(red)} tables redacted)")
    pub, n_sd = stage_supplementary(tree, checks)
    figs_tab = stage_figures(tree, pi, checks)
    figs = list(zip(figs_tab.stem, figs_tab.label, figs_tab.title))
    n_tex = stage_manuscript(tree, san, figs, checks)
    limit = int(args.limit_gb * 1e9)
    big = [r for r in tree.desc if (tree.root / r).stat().st_size > limit - (1 << 21)]
    stage_docs(tree, args, info, idx, man, red, figs_tab, figs, pi, san, big)
    gone = tree.prune()
    if gone:
        log(f"removed {len(gone)} files of earlier runs: {gone[:5]}")
    code_checks(tree, checks)
    if args.latex_check:
        latex_check(tree, checks, out / "_latex_check")
        log("LaTeX check done")
    # manifest (first pass), scan, verification (second pass)
    paths = sorted(tree.desc)
    sums = rl.md5_many([tree.root / r for r in paths])
    tree.md5 = {r: sums[tree.root / r] for r in paths}
    files = pd.DataFrame(dict(path=paths, component=[r.split("/")[0] if "/" in r else "root" for r in paths],
                              size_bytes=[(tree.root / r).stat().st_size for r in paths],
                              md5=[tree.md5[r] for r in paths],
                              description=[tree.desc[r] for r in paths]))
    rl.write_manifest(tree.root, files)
    hits = scan(tree, san, args)
    n_leak = int((hits.severity == "leak").sum())
    checks.append(("leak scan (paths, hosts, names, e-mails, credentials, internal IDs in documents)", n_leak,
                   "pass" if n_leak == 0 else "fail",
                   "; ".join(f"{r.path}:{r.line} {r.check}" for r in hits[hits.severity == "leak"].head(8).itertuples())))
    checks.append(("scan notes (internal file names, wording)", int((hits.severity == "note").sum()), "note",
                   "; ".join(f"{r.path}:{r.line} {r.excerpt}" for r in hits[hits.severity == "note"].head(8).itertuples())))
    v = rl.verify_md5sums(tree.root)
    ok = not (v["mismatched"] or v["missing"] or v["unlisted"])
    checks.append(("MD5SUMS verified by a second pass", f"{v['verified']}/{v['listed']}", "pass" if ok else "fail",
                   "; ".join(v["mismatched"][:3] + v["missing"][:3] + [f"unlisted {u}" for u in v["unlisted"][:3]])))
    checks.append(("release size", rl.human(files.size_bytes.sum()), "info", f"{len(files)} files"))
    write_csv(files, MOD, "release_files.csv", "Public release: every file with component, size, MD5 and description "
              "(= MANIFEST.tsv; 904)", required=["path", "size_bytes", "md5", "description"])
    write_csv(hits, MOD, "release_scan.csv", "Public release: leak-scan hits in released files (paths, hosts, names, "
              "e-mails, credentials, internal identifiers in documents); empty = clean (904)",
              required=["path", "check", "severity"])
    write_csv(pub, MOD, "release_pubclean.csv", "Public release: internal markers left in Supplementary Tables and "
              "Source Data (lib/pubclean.residual); empty = clean (904)", required=["table", "column", "value"])
    write_csv(red, MOD, "release_redactions.csv", "Public release: result tables released with machine names or "
              "server paths redacted (replacements, MD5 before and after; 904)", required=["path", "replacements"])
    if not args.skip_archives:
        upload = out / "upload"
        arc = archives(tree, args, upload, limit)
        prefix = tree.root.name
        rl.write_text(upload / "README.md", upload_readme(arc, args, limit, prefix))
        rl.write_text(upload / "MD5SUMS.txt", "".join(f"{m}  {f}\n" for f, m in zip(arc.file, arc.md5)))
        over = arc[arc.size_bytes > limit]
        checks.append(("upload files <= limit", f"{len(arc) - len(over)}/{len(arc)} (largest "
                       f"{rl.human(arc.size_bytes.max())})", "pass" if over.empty else "fail", ", ".join(over.file)))
        badc = arc[arc.check != "ok"]
        checks.append(("upload files checked (zip members, joined parts)", f"{len(arc) - len(badc)}/{len(arc)}",
                       "pass" if badc.empty else "fail", "; ".join(badc.file)))
        write_csv(arc, MOD, "release_archives.csv", "Public release: upload files (zip archives and byte parts) with "
                  "size, MD5, contents and checks (904)", required=["file", "kind", "size_bytes", "md5"])
    ck = pd.DataFrame(checks, columns=["check", "value", "status", "detail"])
    write_csv(ck, MOD, "release_checks.csv", "Public release: build checks (atlas MD5, figure files, image sources, "
              "pubclean, code compilation, leak scan, MD5 verification, upload sizes; 904)",
              required=["check", "value", "status", "detail"])
    for r in ck.itertuples():
        log(f"{r.status:5s} {r.check}: {r.value} {r.detail}")
    log(f"release {tree.root} built in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
