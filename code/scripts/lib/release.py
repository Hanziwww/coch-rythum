"""Helpers of the public-release builder (904; round 10, phase G): sanitised copies of code and LaTeX, leak scans,
hashing, manifests, upload archives, the h5ad schema and environment specifications.

Nothing in this file names the analysis machines. Sanitizer receives the server project root, the Python and R
interpreters of the server environments, the ssh host, the local project root and the user name at run time (from
config.yaml, the command line and the environment) and derives its rules from them, so 904 and this module can be
released like every other script.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import math
import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------- files and hashes
STORE = {".zip", ".gz", ".bz2", ".xz", ".7z", ".png", ".jpg", ".jpeg", ".webp", ".pdf", ".czi", ".oir", ".bam",
         ".cram", ".kfbf"}                       # already compressed: stored, not deflated, in the upload archives
TEXT = {".py", ".r", ".sh", ".yaml", ".yml", ".md", ".txt", ".tex", ".bib", ".cff", ".csv", ".tsv", ".json", ".svg",
        ".bbl", ".cfg", ".toml", ""}            # '' = files without a suffix (LICENSE-code, latexmkrc)


def md5sum(path, buf=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_many(paths, workers=4) -> dict:
    """{path: md5} of many files (threads: hashlib releases the GIL on large blocks)."""
    paths = list(paths)
    with cf.ThreadPoolExecutor(workers) as ex:
        return dict(zip(paths, ex.map(md5sum, paths)))


def same_file(src: Path, dst: Path) -> bool:
    """dst is an unchanged copy of src (size and modification time; copies keep the source time)."""
    if not dst.exists():
        return False
    a, b = src.stat(), dst.stat()
    return a.st_size == b.st_size and abs(a.st_mtime - b.st_mtime) < 2


def copy_file(src, dst) -> bool:
    """Copy src to dst with its modification time unless an unchanged copy is there; True if copied."""
    src, dst = Path(src), Path(dst)
    if same_file(src, dst):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def is_link(path) -> bool:
    """Symbolic link or Windows directory junction (reparse point)."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return os.path.islink(path) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def walk(root: Path):
    """(files, links, dirs) below root as POSIX paths relative to root, without following links or junctions."""
    files, links, dirs = [], [], []
    stack = [Path(root)]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                rel = Path(e.path).relative_to(root).as_posix()
                if is_link(e.path):
                    links.append(rel)
                elif e.is_dir(follow_symlinks=False):
                    dirs.append(rel)
                    stack.append(Path(e.path))
                else:
                    files.append(rel)
    return files, links, dirs


def write_text(path, text: str) -> bool:
    """UTF-8 text with LF line endings; an identical file is left untouched (keeps its time). True if written."""
    path = Path(path)
    data = text.replace("\r\n", "\n").encode("utf-8")
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def human(n: float) -> str:
    """Byte count as text (decimal units: 1 GB = 10^9 bytes)."""
    for unit in ("bytes", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "bytes" else f"{n:,.1f} {unit}"
        n /= 1000


def md_table(df: pd.DataFrame) -> str:
    """Markdown table (pipes escaped, newlines flattened)."""
    def cell(v):
        s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        return s.replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# ------------------------------------------------------------------------------------------------ sanitising text
class Sanitizer:
    """Rewrites machine-specific paths, hosts and names in released text.

    Flavours: 'sh' (shell scripts: COCH_ROOT, COCH_ENVS, COCH_PYTHON and COCH_RSCRIPT), 'py' and 'R' (paths relative to
    the project root, the working directory of every script), 'text' (documents, comments, configuration) and 'data'
    (result tables: values redacted). Rules, in order:
      server project root   'cd <root>' -> cd "$COCH_ROOT" (sh); quoted root -> Sys.getenv("COCH_ROOT", ".") (R) or
                            "." (py); '<root>/x' -> 'x' (py, R, text) or '$COCH_ROOT/x' (sh) or '<project root>/x'
      other server roots    of the project (storage of uploaded public data) -> 'external/'
      environments          server python / Rscript -> ${COCH_PYTHON:-python} / ${COCH_RSCRIPT:-Rscript} (sh) or the
                            bare command; environment root -> ${COCH_ENVS:-$COCH_ROOT/envs} (sh) or 'envs'; conda base
                            binaries -> the command on PATH
      network proxy         'source /etc/network_*' lines and prefixes removed
      provider, host        cloud-provider words and the ssh host -> 'the analysis server' ('<redacted>' in tables)
      local root, drives    the local project root -> /path/to/<project>; folders that scripts read through an
                            environment variable with a default on a local drive (named roots) -> the folder name as
                            the default in code and $VAR elsewhere; parentheticals that hold a drive path are dropped
                            (docstrings and comments); other drive paths -> '<local path>'
      user name             -> '<user>'
    """

    def __init__(self, server_root, python, rscript, local_roots=(), hosts=(), names=(), named_roots=()):
        P = PurePosixPath
        self.server_root = str(server_root).rstrip("/")
        self.python, self.rscript = str(python), str(rscript)
        self.env_root = str(P(self.python).parents[2])               # <envs>/<env>/bin/python -> <envs>
        self.project = P(self.server_root).name
        words = {re.split(r"[-_.]", p)[0].lower() for p in P(self.server_root).parts[1:-1]}
        self.provider = sorted(w for w in words if len(w) >= 4 and w not in ("root", "home", "data", "work"))
        self.hosts = sorted({h for h in hosts if h and len(h) >= 3})
        self.names = sorted({n for n in names if n and len(n.replace(" ", "")) >= 4})
        roots = set()
        for r in local_roots:
            r = str(r).rstrip("/\\")
            if r:
                roots |= {r.replace("\\", "/"), r.replace("/", "\\"), r.replace("/", "\\").replace("\\", "\\\\")}
        self.local_roots = sorted(roots, key=len, reverse=True)
        self.named_roots = []                                       # (path variant, environment variable, default)
        for path, var in named_roots:
            path = str(path).rstrip("/\\")
            for v in {path.replace("\\", "/"), path.replace("/", "\\")}:
                self.named_roots.append((v, var, re.split(r"[\\/]", path)[-1] or var.lower()))
        self.rules = self._rules()

    def _rules(self):
        R = []

        def add(pat, flags=0, **rep):
            R.append((re.compile(pat, flags), rep))

        esc = re.escape
        sr, env, proj = esc(self.server_root), esc(self.env_root), esc(self.project)
        other = rf"(?<![\w/.~-])/root/[\w.-]+/{proj}(?![\w.-])"
        add(rf"\bcd {sr}(?![\w/.-])", sh='cd "${COCH_ROOT:?set COCH_ROOT to the project root}"', default="cd $COCH_ROOT")
        add(rf"^(\s*)ROOT={sr}\s*$", re.M, sh=r"\1ROOT=${COCH_ROOT:-.}")
        add(rf'"{sr}"', R='Sys.getenv("COCH_ROOT", ".")', py='"."', sh='"$COCH_ROOT"', default='"<project root>"')
        add(rf"{sr}/", sh="$COCH_ROOT/", data="<project root>/", default="")
        add(rf"{sr}(?![\w.-])", sh="$COCH_ROOT", py=".", R="$COCH_ROOT", data="<project root>",
            default="the project root")
        add(other + "/", sh="$COCH_ROOT/external/", default="external/")
        add(other, sh="$COCH_ROOT/external", default="external")
        add(esc(self.python) + r"(?![\w.-])", sh="${COCH_PYTHON:-python}", default="python")
        add(esc(self.rscript) + r"(?![\w.-])", sh="${COCH_RSCRIPT:-Rscript}", default="Rscript")
        add(rf'"{env}/([^"\n]*)"', R=r'file.path(Sys.getenv("COCH_ENVS", "envs"), "\1")')
        add(rf"{env}/", sh="${COCH_ENVS:-$COCH_ROOT/envs}/", default="envs/")
        add(rf"{env}(?![\w.-])", sh="${COCH_ENVS:-$COCH_ROOT/envs}", default="envs")
        add(r"(?<![\w/.~-])/root/(?:mini|ana)conda3/bin/", default="")
        add(r"^([ \t]*)source /etc/network_\w+[^\n;]*$", re.M,
            default=r"\1: # site-specific network proxy (not needed elsewhere)")
        add(r"source /etc/network_\w+\s*>\s*/dev/null(?:\s*2>&1)?\s*;\s*", default="")
        for p in self.provider:
            add(rf"\b{esc(p)}(?:[-_]\w+)+", re.I, data="<redacted>", default="the analysis server")
            add(rf"\b{esc(p)}\b", re.I, default="analysis")
        for h in self.hosts:
            add(rf"(?<![\w.-]){esc(h)}(?![\w.-])", re.I, data="<redacted>", default="the analysis server")
        for lr in self.local_roots:
            add(esc(lr), re.I, data="<local project root>", default="/path/to/" + self.project)
        for path, var, base in self.named_roots:
            add(rf'(["\']){esc(path)}\1', re.I, py=rf"\g<1>{base}\g<1>", R=rf"\g<1>{base}\g<1>")
            add(esc(path) + r"(?![\w.-])", re.I, sh="${" + var + "}", default="$" + var)
        # drive paths: upper-case drive letters only, so that regular expressions such as r"z:\d+" stay untouched
        add(r"[ \t]*\([^()]{0,300}?(?<!\w)[A-Z]:[\\/][^()]{0,300}\)", default="", data=None)
        add(r"(?<![\w/])[A-Z]:[\\/]+[^\s\"'`,;()]*", default="<local path>")
        for n in self.names:
            add(r"\s*".join(map(esc, n.split())), re.I, default="<user>")
        return R

    def apply(self, text: str, flavour: str):
        """(sanitised text, number of replacements)."""
        n = 0
        for rx, rep in self.rules:
            r = rep.get(flavour, rep.get("default"))
            if r is None:
                continue
            text, k = rx.subn(r, text)
            n += k
        return text, n

    def literals(self) -> list:
        """Lower-case byte strings that must not occur in any released file (raw scan of binary files)."""
        lits = {b"/root/"} | {p.encode() for p in self.provider} | {h.lower().encode() for h in self.hosts}
        lits |= {r.lower().encode() for r in self.local_roots if "\\\\" not in r}
        lits |= {n.lower().encode() for n in self.names}
        return sorted(lits)


FLAVOUR = {".sh": "sh", ".py": "py", ".r": "R"}


def flavour_of(path) -> str:
    return FLAVOUR.get(Path(path).suffix.lower(), "text")


# internal identifiers and wording that must not appear in documents (README, docs, manifest descriptions)
INTERNAL_ID = re.compile(r"\b(?:D\d{2,3}|I\d{3}|O\d{2})\b")
_ID_PAREN = re.compile(r"[ \t]*\([^()]*?\b(?:D|I|O)\d{2,3}\b[^()]*\)")


def public_text(s, subs=()) -> str:
    """Document text without internal log identifiers, analysis-round labels or project-internal wording."""
    s = str(s)
    for pat, rep in subs:                                   # lib/pubclean.SUBS (released-table rules)
        s = re.sub(pat, rep, s)
    s = _ID_PAREN.sub("", s)                                # '(D82; 930)', '(D82-D88)'
    s = re.sub(r"\b(?:D|I|O)\d{2,3}\b(?:\s*[-\u2013]\s*(?:D|I|O)?\d{2,3}\b)?[ \t]*[:;,]?[ \t]*", "", s)
    s = re.sub(r"(?i)\bround[- ]?\d+\b(?:,?\s*(?:phase [A-G]|WP\s?\d+|step [A-Z]))?[ \t]*[,:;]?[ \t]*(?:[-\u2013\u2014][ \t]*)?",
               "", s)
    s = re.sub(r"(?i)\bphase [A-G]\b[ \t]*[,:;]?[ \t]*(?:[-\u2013\u2014][ \t]*)?", "", s)
    s = re.sub(r"[ \t]*\([^()]*(?:\bCLAUDE\.md\b|(?<![\w/])memory/)[^()]*\)", "", s)   # pointers to internal notes
    s = re.sub(r"(?<![\w/])memory/[\w.]+|\bCLAUDE\.md\b", "the authors' notes", s)
    s = re.sub(r"\bthe user's\b", "the authors'", s)
    s = re.sub(r"\bThe user's\b", "The authors'", s)
    s = re.sub(r"\b(?:the|The) user\b", "the authors", s)
    s = re.sub(r"\bheadline\b", "key", s)
    s = re.sub(r"\(\s*[,;]?\s*\)", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip(" ;,:\u2014\u2013-")
    return s[:1].upper() + s[1:]


# -------------------------------------------------------------------------------------------------------- LaTeX
def _comment_start(line: str) -> int:
    """Index of the first comment '%' (one preceded by an even number of backslashes), or -1."""
    for m in re.finditer("%", line):
        k, i = 0, m.start() - 1
        while i >= 0 and line[i] == "\\":
            k, i = k + 1, i - 1
        if k % 2 == 0:
            return m.start()
    return -1


def tex_public(text: str, leaky=lambda s: False) -> str:
    """LaTeX source without comments: full-line comments are dropped except the '% audit:' fences read by 905 (kept
    bare when their note is leaky); trailing comments keep their '%' (it still joins the lines). Lines with \\url,
    \\href or \\verb are left as they are. The text that LaTeX typesets is unchanged."""
    out = []
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.lstrip()
        if s.startswith("%"):
            if s.startswith("% audit:"):
                out.append(line if not leaky(line) else line[:len(line) - len(s)] + s.split("(")[0].rstrip())
            continue
        i = _comment_start(line)
        if i >= 0 and not re.search(r"\\(?:url|href|verb)\b", line):
            line = line[:i + 1]
        out.append(line)
    return "\n".join(out)


def bib_public(text: str) -> str:
    """BibTeX file without comment lines (text outside entries starting with '%')."""
    return "\n".join(ln for ln in text.replace("\r\n", "\n").split("\n") if not ln.lstrip().startswith("%"))


def perl_public(text: str) -> str:
    """latexmkrc without comment lines and trailing comments."""
    out = []
    for ln in text.replace("\r\n", "\n").split("\n"):
        if ln.lstrip().startswith("#"):
            continue
        out.append(re.sub(r"\s+#[^'\"]*$", "", ln))
    return "\n".join(out).strip("\n") + "\n"


def tex_to_text(s: str) -> str:
    """Plain text of a short LaTeX snippet (titles, abstract)."""
    s = "\n".join(ln for ln in s.split("\n") if not ln.lstrip().startswith("%"))
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = s.replace("---", "\u2014").replace("--", "\u2013").replace("~", " ").replace("\\%", "%").replace("\\&", "&")
    s = re.sub(r"\$?\\times\$?", "\u00d7", s)
    s = re.sub(r"\\(?:cite|citep|citet|ref|label)\{[^{}]*\}", "", s)
    for _ in range(3):
        s = re.sub(r"\\(?:textit|emph|textbf|textrm|mathrm|text|textsc|mbox)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\,", " ").replace("$", "")
    s = re.sub(r"\\[A-Za-z]+\*?\s?", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------------------------------- leak scan
def leak_patterns(san: Sanitizer, allow_emails=()):
    """(label, severity, regex, literals): what must not appear in released text. The literals pre-filter files."""
    P = [
        ("server path", "leak", r"(?<![\w.~-])/root/[\w.-]+", [b"/root/"]),
        ("home path", "leak", r"(?<![\w.~-])/home/[A-Za-z][\w.-]*", [b"/home/"]),
        ("local drive path", "leak", r"(?<![\w/])[A-Z]:[\\/]{1,2}[\w .~-]", [b":/", b":\\"]),
        ("e-mail address", "leak", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|edu|gov|"
                                   r"info|io|ai|bio|[a-z]{2})\b", [b"@"]),     # TLDs: not R slots such as obj@meta.data
        ("credential", "leak", r"(?i)\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|passw(?:or)?d|access[_-]?token|"
                               r"auth[_-]?token|bearer)\b[\"']?\s*[:=]\s*[\"']?[^\s\"'<>]{6,}",
         [b"key", b"secret", b"passw", b"token", b"bearer"]),
        ("key material", "leak", r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bhf_[A-Za-z0-9]{30,}|"
                                 r"\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9_-]{30,}|\bxox[abp]-[A-Za-z0-9-]{10,}",
         [b"-----begin", b"gh", b"hf_", b"akia", b"sk-", b"xox"]),
        ("network proxy", "leak", r"/etc/network_\w+", [b"/etc/network_"]),
        ("internal file", "note", r"(?i)\bCLAUDE\.md\b|\bauthor_inputs\.md\b|(?<![\w/])memory/[\w.]+",
         [b"claude.md", b"author_inputs", b"memory/"]),
    ]
    P += [("cloud provider", "leak", rf"(?i)\b{re.escape(p)}", [p.encode()]) for p in san.provider]
    P += [("ssh host", "leak", rf"(?i)(?<![\w.-]){re.escape(h)}(?![\w.-])", [h.lower().encode()]) for h in san.hosts]
    P += [("user name", "leak", r"(?i)" + r"\s*".join(map(re.escape, n.split())), [n.split()[0].lower().encode()])
          for n in san.names]
    allow = {e.lower() for e in allow_emails}
    out = []
    for lab, sev, rx, lits in P:
        out.append((lab, sev, re.compile(rx), lits, allow if lab == "e-mail address" else set()))
    return out


DOC_PATTERNS = [("internal ID", "leak", INTERNAL_ID, [], set()),
                ("internal wording", "note", re.compile(r"\b(?:the user|headline)\b|\bround[- ]\d+\b", re.I), [], set())]


def scan_file(path, patterns, max_hits=20) -> list:
    """[(label, severity, line, excerpt)] of pattern hits in a text file."""
    data = Path(path).read_bytes()
    low = data.lower()
    text, out = None, []
    for lab, sev, rx, lits, allow in patterns:
        if lits and not any(x in low for x in lits):
            continue
        if text is None:
            text = data.decode("utf-8", errors="replace")
        k = 0
        for m in rx.finditer(text):
            if m.group(0).lower() in allow:
                continue
            out.append((lab, sev, text.count("\n", 0, m.start()) + 1, m.group(0)[:80]))
            k += 1
            if k >= max_hits:
                break
    return out


def scan_bytes(path, literals) -> list:
    """Literal byte strings found anywhere in a (binary) file."""
    low = Path(path).read_bytes().lower()
    return [x.decode() for x in literals if x in low]


# ------------------------------------------------------------------------------------------------ h5ad (AnnData)
def _decode(a):
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        return np.array([x.decode("utf-8") if isinstance(x, bytes) else x for x in a], dtype=object)
    return a


def _column(g, name):
    import h5py
    item = g[name]
    if isinstance(item, h5py.Group):
        enc = item.attrs.get("encoding-type", "")
        if enc == "categorical":
            cats = _decode(item["categories"][:])
            codes = item["codes"][:]
            vals = np.where(codes >= 0, cats[np.clip(codes, 0, max(len(cats) - 1, 0))] if len(cats) else None, None)
            return pd.Categorical(vals, categories=list(cats)), "categorical"
        if enc.startswith("nullable"):
            vals = _decode(item["values"][:]).astype(object)
            if "mask" in item:
                vals[item["mask"][:]] = None
            return vals, enc
        return None, enc or "group"
    if "categories" in item.attrs:                        # AnnData < 0.8: categories by reference
        cats = _decode(g.file[item.attrs["categories"]][:])
        codes = item[:]
        return pd.Categorical(np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], None),
                              categories=list(cats)), "categorical"
    return _decode(item[:]), str(item.dtype)


def h5ad_frame(path, key="obs", columns=None) -> pd.DataFrame:
    """obs or var of an h5ad file as a DataFrame (h5py only; categoricals decoded)."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f[key]
        order = [c.decode() if isinstance(c, bytes) else str(c) for c in g.attrs.get("column-order", [])]
        cols = [c for c in order if columns is None or c in columns]
        data = {c: _column(g, c)[0] for c in cols}
        idx_name = g.attrs.get("_index", "_index")
        idx = _column(g, idx_name)[0] if idx_name in g else None
    df = pd.DataFrame(data, index=pd.Index(idx, name=None) if idx is not None else None)
    return df


def h5ad_schema(path) -> dict:
    """Structure of an h5ad file: X, obs / var columns (type, values), layers, obsm, varm, obsp, varp, uns."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        out["encoding"] = f.attrs.get("encoding-type", ""), f.attrs.get("encoding-version", "")
        X = f["X"]
        if isinstance(X, h5py.Group):
            data = X["data"]
            sample = data[: min(data.shape[0], 5_000_000)]
            out["X"] = dict(storage=X.attrs.get("encoding-type", "sparse"), shape=tuple(int(v) for v in X.attrs["shape"]),
                            dtype=str(data.dtype), nnz=int(data.shape[0]),
                            integer=bool(np.all(np.mod(sample, 1) == 0)), max_sampled=float(sample.max()))
        else:
            sample = X[: min(X.shape[0], 1000)]
            out["X"] = dict(storage="dense", shape=tuple(X.shape), dtype=str(X.dtype), nnz=None,
                            integer=bool(np.all(np.mod(sample, 1) == 0)), max_sampled=float(np.max(sample)))
        for key in ("obs", "var"):
            g = f[key]
            order = [c.decode() if isinstance(c, bytes) else str(c) for c in g.attrs.get("column-order", [])]
            cols = []
            for c in order:
                vals, kind = _column(g, c)
                if isinstance(vals, pd.Categorical):
                    cats = list(vals.categories)
                    cols.append(dict(column=c, type=f"categorical ({len(cats)} level{'' if len(cats) == 1 else 's'})",
                                     n_levels=len(cats), levels=cats))
                else:
                    v = pd.Series(vals)
                    if v.dtype == bool or kind == "bool":
                        cols.append(dict(column=c, type="boolean", n_levels=2,
                                         levels=[f"True {int(v.sum()):,}", f"False {int((~v.astype(bool)).sum()):,}"]))
                    elif np.issubdtype(v.dtype, np.number):
                        cols.append(dict(column=c, type=str(v.dtype), n_levels=None,
                                         levels=[f"{v.min():.4g}", f"{v.median():.4g}", f"{v.max():.4g}"]))
                    else:
                        cols.append(dict(column=c, type="string", n_levels=int(v.nunique()), levels=list(v.unique()[:5])))
            idx_name = g.attrs.get("_index", "_index")
            n = None
            if idx_name in g:
                iv = g[idx_name]
                n = (iv["values"].shape[0] if isinstance(iv, h5py.Group) else iv.shape[0])
            out[key] = dict(columns=cols, n=n)
        for key in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
            items = []
            if key in f:
                for k, v in f[key].items():
                    shape = tuple(v.shape) if isinstance(v, h5py.Dataset) else tuple(v.attrs.get("shape", ()))
                    items.append((k, shape, v.attrs.get("encoding-type", "")))
            out[key] = items
    return out


# ------------------------------------------------------------------------------------------- environment specs
R_BIOC = {"limma", "edgeR", "variancePartition", "rain", "speckle", "sccomp", "BiocParallel", "SingleCellExperiment",
          "DropletUtils", "scuttle", "BiocNeighbors", "RcisTarget", "miloR", "DESeq2", "fgsea", "goseq", "qvalue",
          "tricycle", "UCell", "muscat", "scran", "scater", "sva", "TOAST", "ComplexHeatmap", "SummarizedExperiment",
          "scDblFinder"}
R_GITHUB = {"dryR": "naef-lab/dryR", "diffCircadian": "diffCircadian/diffCircadian",
            "compareRhythms": "bharathananth/compareRhythms", "nichenetr": "saeyslab/nichenetr",
            "multinichenetr": "saeyslab/multinichenetr", "InstaPrism": "humengying0907/InstaPrism",
            "BayesPrism": "Danko-Lab/BayesPrism", "MuSiC": "xuranw/MuSiC", "tauFisher": "micnngo/tauFisher"}


def r_packages(sv: pd.DataFrame) -> pd.DataFrame:
    """R packages with their source and install route (Bioconductor / CRAN via conda, or GitHub)."""
    rows = []
    for p, v in sv.loc[(sv.language == "R") & (sv.package != "R"), ["package", "version"]].itertuples(index=False):
        if p in R_GITHUB:
            src, how = "GitHub", f'remotes::install_github("{R_GITHUB[p]}")'
        elif p in R_BIOC:
            src, how = "Bioconductor", f"conda: bioconductor-{p.lower()}={v}"
        else:
            src, how = "CRAN", f"conda: r-{p.lower()}={v}"
        rows.append(dict(package=p, version=v, source=src, install=how))
    return pd.DataFrame(rows)


def env_specs(sv: pd.DataFrame, local: pd.DataFrame) -> dict:
    """Environment files derived from the recorded package versions: {file name: text}."""
    py = sv[sv.language == "python"].set_index("package")["version"]
    ov = sv[sv.language.str.startswith("python overlay")]
    r = sv[sv.language == "R"].set_index("package")["version"]
    out = {}
    pins, notes = [], []
    for p, v in py.items():
        if p == "python":
            continue
        if "+" in str(v):
            notes.append(f"# {p} was the {v.split('+')[1]} build ({v}); install the build that matches your hardware.")
            v = v.split("+")[0]
        pins.append(f"      - {p}=={v}")
    out["environment-server-python.yml"] = "\n".join(
        ["# Python environment of the server analyses (single-cell, statistics, integration), pinned to the versions used",
         "# (software_versions_server.csv). Create with: conda env create -f environment-server-python.yml",
         *notes, "name: coch-sc", "channels:", "  - conda-forge", "dependencies:", f"  - python={py.get('python', '3.12')}",
         "  - pip", "  - pip:", *pins]) + "\n"
    out["requirements-trajectory-overlay.txt"] = "\n".join(
        ["# Separate package overlay for the trajectory checks of the myeloid hypotheses (630_myeloid_trajectories.py),",
         "# which add envs/traj_pkgs to the Python path (it needs its own SciPy). Install with:",
         "#   pip install --target envs/traj_pkgs -r requirements-trajectory-overlay.txt",
         *[f"{p}=={v}" for p, v in ov[["package", "version"]].itertuples(index=False)]]) + "\n"
    rp = r_packages(sv)
    conda = [f"  - {row.install.split(': ', 1)[1]}" for row in rp.itertuples() if row.install.startswith("conda:")]
    out["environment-server-r.yml"] = "\n".join(
        ["# R environment of the rhythm statistics, compositional models, communication and deconvolution analyses,",
         "# pinned to the versions used (software_versions_server.csv). Create with:",
         "#   conda env create -f environment-server-r.yml",
         "# then install the GitHub packages listed in r_packages.csv (remotes::install_github). If a pinned version is not",
         "# available from conda, install that version from CRAN or Bioconductor.",
         "name: coch-rcirc", "channels:", "  - conda-forge", "  - bioconda", "dependencies:",
         f"  - r-base={r.get('R', '4.5')}", "  - r-remotes", "  - r-yaml", "  - cmdstan", *conda]) + "\n"
    lp = local.set_index("package")["version"]
    lpins, lnotes = [], []
    for p, v in lp.items():
        if p == "python" or str(v) == "not installed":
            continue
        if "+" in str(v):
            lnotes.append(f"# {p} was the {v.split('+')[1]} build ({v}); install the build that matches your hardware.")
            v = v.split("+")[0]
        lpins.append(f"      - {p}=={v}")
    out["environment-local-python.yml"] = "\n".join(
        ["# Python environment of the imaging, flow-cytometry, figure, table and manuscript scripts (9xx scripts that",
         "# state a local run), run on a workstation; versions from software_versions_local.csv. Create with:",
         "#   conda env create -f environment-local-python.yml", *lnotes,
         "name: coch-local", "channels:", "  - conda-forge", "dependencies:", f"  - python={lp.get('python', '3.10')}",
         "  - pip", "  - pip:", *lpins]) + "\n"
    return out


# --------------------------------------------------------------------------------------------- manifest and md5
def write_manifest(root: Path, df: pd.DataFrame) -> None:
    """MANIFEST.tsv (path, size_bytes, md5, description) and MD5SUMS.txt (md5sum -c format) at the release root."""
    df = df.sort_values("path")
    clean = df["description"].astype(str).str.replace(r"[\t\r\n]+", " ", regex=True)
    tsv = "path\tsize_bytes\tmd5\tdescription\n" + "".join(
        f"{p}\t{s}\t{m}\t{d}\n" for p, s, m, d in zip(df["path"], df["size_bytes"], df["md5"], clean))
    write_text(root / "MANIFEST.tsv", tsv)
    write_text(root / "MD5SUMS.txt", "".join(f"{m}  {p}\n" for p, m in zip(df["path"], df["md5"])))


def verify_md5sums(root: Path, workers=4) -> dict:
    """Second pass: re-hash every file listed in MD5SUMS.txt and compare; also report files not listed."""
    listed = {}
    for ln in (root / "MD5SUMS.txt").read_text(encoding="utf-8").splitlines():
        m, p = ln.split("  ", 1)
        listed[p] = m
    files, links, _ = walk(root)
    on_disk = (set(files) | set(links)) - {"MD5SUMS.txt", "MANIFEST.tsv"}
    present = [p for p in listed if (root / p).is_file()]
    got = md5_many([root / p for p in present], workers)
    bad = [p for p in present if got[root / p] != listed[p]]
    return dict(listed=len(listed), verified=len(present) - len(bad), mismatched=bad,
                missing=sorted(set(listed) - set(present)), unlisted=sorted(on_disk - set(listed)))


# ----------------------------------------------------------------------------------------------- upload archives
def plan_volumes(items, limit, overhead=1024, ratio=1.001):
    """Greedy split of (rel, size) items, in order, into zip volumes whose worst-case size stays under limit; items that
    alone exceed the limit are returned apart (they are split into byte parts)."""
    vols, cur, tot, big = [], [], 0.0, []
    for rel, size in items:
        need = size * ratio + overhead + 2 * len(rel)
        if need > limit - (1 << 20):
            big.append((rel, size))
            continue
        if cur and tot + need > limit - (1 << 20):
            vols.append(cur)
            cur, tot = [], 0.0
        cur.append(rel)
        tot += need
    if cur:
        vols.append(cur)
    return vols, big


def zip_current(dst: Path, root: Path, rels, prefix) -> bool:
    """dst holds exactly these files (names, sizes) and is newer than each of them."""
    if not dst.exists():
        return False
    try:
        with zipfile.ZipFile(dst) as z:
            have = {i.filename: i.file_size for i in z.infolist()}
    except zipfile.BadZipFile:
        return False
    want = {f"{prefix}/{r}": (root / r).stat().st_size for r in rels}
    newest = max((root / r).stat().st_mtime for r in rels)
    return have == want and dst.stat().st_mtime >= newest


def write_zip(dst: Path, root: Path, rels, prefix, level=6) -> None:
    """Zip root/<rel> as <prefix>/<rel> (already-compressed formats stored, others deflated; ZIP64)."""
    tmp = dst.with_name(dst.name + ".part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for rel in rels:
            p = root / rel
            stored = p.suffix.lower() in STORE
            z.write(p, arcname=f"{prefix}/{rel}", compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                    compresslevel=None if stored else level)
    os.replace(tmp, dst)


def check_zip(dst: Path, root: Path, rels, prefix, crc=False) -> list:
    """Problems of a written archive: member names and sizes against the files (and CRCs with crc=True)."""
    want = {f"{prefix}/{r}": (root / r).stat().st_size for r in rels}
    with zipfile.ZipFile(dst) as z:
        have = {i.filename: i.file_size for i in z.infolist()}
        bad = [f"missing {n}" for n in sorted(set(want) - set(have))] + [f"extra {n}" for n in sorted(set(have) - set(want))]
        bad += [f"size {n}" for n in want if n in have and have[n] != want[n]]
        if crc and not bad:
            first = z.testzip()
            if first:
                bad.append(f"crc {first}")
    return bad


def split_parts(src: Path, dst_dir: Path, name: str, limit: int) -> list:
    """Byte parts <name>.001, .002, ... of src, each <= limit (equal sizes); existing parts that are newer than src and
    of the right sizes are kept."""
    size = src.stat().st_size
    n = math.ceil(size / limit)
    part = math.ceil(size / n)
    sizes = [min(part, size - i * part) for i in range(n)]
    dsts = [dst_dir / f"{name}.{i + 1:03d}" for i in range(n)]
    if all(d.exists() and d.stat().st_size == s and d.stat().st_mtime >= src.stat().st_mtime for d, s in zip(dsts, sizes)):
        return dsts
    with open(src, "rb") as fh:
        for d, s in zip(dsts, sizes):
            tmp = d.with_name(d.name + ".part")
            with open(tmp, "wb") as o:
                left = s
                while left:
                    b = fh.read(min(1 << 24, left))
                    o.write(b)
                    left -= len(b)
            os.replace(tmp, d)
    return dsts


def md5_of_parts(parts) -> str:
    """MD5 of the concatenation of byte parts (= the reassembled file)."""
    h = hashlib.md5()
    for p in parts:
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()

# ------------------------------------------------------------------------------------------- licence text
# Apache License 2.0, verbatim (https://www.apache.org/licenses/LICENSE-2.0.txt; MD5 3b83ef96387f14655fc854ddc3c6bd57)
APACHE_2_0 = """
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
