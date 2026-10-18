"""Round 9 (D61) — figure QA for composites, after the reference design system:
colour-blind (Machado, Oliveira & Fernandes 2009, severity 1.0; protanopia / deuteranopia / tritanopia) and greyscale
simulations of each rendered composite, and the smallest CIE76 colour difference between the palette colours of a
figure, normally and under each simulation (pairs below 10 listed).

Also checks text geometry in the vector PDFs (coca-fig only checks that text stays inside a panel): every pair of text
spans whose tight glyph boxes overlap by more than 0.3 mm in both directions, every span below 5 pt, every span
that extends beyond the page (clipped text) and, since D137, every coloured text span (Nature: black text, colour in
the keys; chroma > 0.25).

Runs locally on the coca-fig outputs (no analysis data). Inputs: one or more composite JPG/PNG/PDF paths, or the default
figures/figures/*composite*.{jpg,pdf}. Outputs: figures/qc/<stem>_<kind>.png, figures/qc/palette_distinguishability.csv
and figures/qc/text_overlaps.csv. Run: python scripts/906_figure_qc.py [file ...]
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "figures" / "qc"
MACHADO = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}
# palettes in use (lib/figdata.py and the Fig. 2 panel options)
PALETTES = {
    "cell classes": {"Sensory": "#CC6677", "Neuronal": "#882255", "Glia": "#AA4499", "Epithelial": "#44AA99",
                     "Stromal": "#999933", "Bone": "#DDCC77", "Endothelial": "#88CCEE", "Progenitor": "#332288",
                     "Cycling": "#222222", "Erythroid": "#7A0403", "Lymphoid": "#117733", "Myeloid": "#E69F00",
                     "not significant": "#D9D9D9"},
    "fig2 grammar": {"structural": "#0E5A4E", "haematopoietic": "#767676", "pool": "#222222",
                     "cochlea": "#0072B2", "marrow": "#D55E00"},
    "fig2 genes": {"Dbp": "#E69F00", "Nr1d1": "#56B4E9", "Per2": "#009E73", "Npas2": "#CC79A7"},
    "fig3 drivers": {"core clock": "#000000", "clock-controlled": "#AA3377", "temperature": "#E69F00",
                     "glucocorticoid": "#F0E442", "handling-linked": "#56B4E9", "composition": "#228833",
                     "sample pattern": "#9A9A9A", "unexplained": "#E3E3E3"},
    "fig3c sets": {"heat shock": "#B2182B", "IEG": "#7F7F7F", "NF-kB": "#2C5C8A"},
    "fig3f species": {"mouse": "#B2182B", "baboon": "#6B4C9A"},
    "fig5i ageing": {"timer score": "#000000", "Sell": "#009E73", "Cxcr4/Cxcr2": "#CC79A7"},
    "fig6 groups": {"control": "#1A1A1A", "day noise": "#E69F00", "night noise": "#6B4C9A"},
    "fig6 statuses": {"supporting": "#1A1A1A", "inconclusive": "#BDBDBD", "opposing": "#CC79A7", "confound": "#E69F00"},
    "fig6 direction": {"up": "#B2182B", "down": "#2166AC"},
    "ed4 rules": {"BH q < 0.05": "#1A1A1A", "BH q < 0.1": "#56B4E9", "+ permutation q": "#E69F00",
                  "Tier-1-like": "#CC79A7"},
    "ed4 pattern": {"none": "#1A1A1A", "half": "#8C8C8C", "full": "#CFCFCF"},
    "ed5 phase anchors": {"ZT0": "#E9C46A", "ZT6": "#D9822B", "ZT12": "#8E5B9E", "ZT18": "#2E4A7D"},
    "ed7 tissues + baboon": {"cochlea": "#0072B2", "marrow": "#D55E00", "baboon": "#6B4C9A", "raw": "#B0B0B0"},
    "ed7 diverging": {"low": "#B35806", "mid": "#F7F7F7", "high": "#542788"},
    "ed10 groups": {"control": "#1A1A1A", "day noise": "#E69F00", "night noise": "#6B4C9A", "CMN": "#7F7F7F"},
    "ed10 estimators": {"T2": "#1A1A1A", "T1": "#7F7F7F", "T1 noise-robust": "#BDBDBD", "T3": "#56B4E9"},
}


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb, kind):
    if kind == "greyscale":
        y = _lin(rgb) @ np.array([0.2126, 0.7152, 0.0722])
        return np.repeat(_srgb(y)[..., None], 3, axis=-1)
    return _srgb(_lin(rgb) @ MACHADO[kind].T)


def _lab(rgb):
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = _lin(rgb) @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def simulate_image(path: Path):
    img = np.asarray(Image.open(path).convert("RGB"), float) / 255
    for kind in ("protanopia", "deuteranopia", "tritanopia", "greyscale"):
        out = QC / f"{path.stem}_{kind}.png"
        Image.fromarray((simulate(img, kind) * 255 + 0.5).astype(np.uint8)).save(out)
        print(f"[qc] {out.relative_to(ROOT)}")


def palette_report():
    rows = []
    for pal, cols in PALETTES.items():
        names = list(cols)
        rgb = np.array([hex_rgb(cols[n]) for n in names])
        for kind in ("normal", "protanopia", "deuteranopia", "tritanopia", "greyscale"):
            lab = _lab(rgb if kind == "normal" else simulate(rgb, kind))
            pairs = sorted((float(np.linalg.norm(lab[i] - lab[j])), names[i], names[j])
                           for i in range(len(names)) for j in range(i + 1, len(names)))
            below = [f"{a}~{b} ({d:.1f})" for d, a, b in pairs if d < 10]
            rows.append({"palette": pal, "vision": kind, "min_delta_e": round(pairs[0][0], 1),
                         "closest_pair": f"{pairs[0][1]}~{pairs[0][2]}", "pairs_below_10": "; ".join(below)})
    out = QC / "palette_distinguishability.csv"
    with open(out, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[qc] {out.relative_to(ROOT)}")
    return rows


PT_MM = 25.4 / 72
TOL_MM = 0.3        # overlaps up to 0.3 mm are tolerated
MIN_PT = 5.0


def _run_collisions(span):
    """Labels on one baseline are merged into one span. With advance-width glyph boxes, a glyph starting more than
    TOL_MM before the previous glyph's advance ends means two text runs overlap (kerning is far smaller)."""
    out, chars = [], [ch for ch in span.get("chars", []) if ch["c"].strip()]
    tol = max(TOL_MM, 0.2 * span["size"] * PT_MM)            # DejaVu kerning pairs such as 'Tr' reach ~0.15 em
    for prev, ch in zip(chars, chars[1:]):
        w = (prev["bbox"][2] - ch["bbox"][0]) * PT_MM
        if w > tol:
            out.append((prev, ch, w))
    return out


def text_issues(pdf: Path):
    """Overlapping text spans (tight glyph boxes) and spans below MIN_PT on every page of a vector PDF."""
    rows = []
    with pymupdf.open(pdf) as doc:
        for pno, page in enumerate(doc):
            for b in page.get_text("rawdict")["blocks"]:                       # advance-width glyph boxes
                for line in b.get("lines", []):
                    if line["dir"] != (1.0, 0.0):
                        continue
                    for s in line["spans"]:
                        text = "".join(ch["c"] for ch in s.get("chars", []))
                        for prev, ch, w in _run_collisions(s):
                            rows.append({"file": pdf.name, "page": pno + 1, "issue": "overlap", "text_a": text,
                                         "text_b": f"glyph '{prev['c']}' x '{ch['c']}'", "size_pt": round(s["size"], 2),
                                         "x_mm": round(ch["bbox"][0] * PT_MM, 1), "y_mm": round(ch["bbox"][1] * PT_MM, 1),
                                         "overlap_w_mm": round(w, 2), "overlap_h_mm": ""})
            spans = []
            pr = page.rect
            for b in page.get_text("dict", flags=pymupdf.TEXT_ACCURATE_BBOXES)["blocks"]:  # tight glyph boxes
                for li, line in enumerate(b.get("lines", [])):
                    for s in line["spans"]:                  # any text, rotated or not, that leaves the page
                        x0, y0, x1, y1 = s["bbox"]
                        out = max(pr.x0 - x0, pr.y0 - y0, x1 - pr.x1, y1 - pr.y1) * PT_MM
                        if s["text"].strip() and out > 0.05:
                            rows.append({"file": pdf.name, "page": pno + 1, "issue": "off_page", "text_a": s["text"],
                                         "text_b": "", "size_pt": round(s["size"], 2), "x_mm": round(x0 * PT_MM, 1),
                                         "y_mm": round(y0 * PT_MM, 1), "overlap_w_mm": round(out, 2), "overlap_h_mm": ""})
                    dx, dy = line["dir"]
                    if min(abs(dx), abs(dy)) > 1e-3:
                        continue    # oblique text (e.g. 45-degree labels): axis-aligned boxes overstate overlap
                    for s in line["spans"]:
                        if s["text"].strip():
                            spans.append((b["number"], li, s))
            for _, _, s in spans:                    # D137: text is black / grey / white; colour lives in the keys
                r_, g_, b_ = ((s.get("color", 0) >> 16) & 255) / 255, ((s.get("color", 0) >> 8) & 255) / 255,                     (s.get("color", 0) & 255) / 255
                mx, mn = max(r_, g_, b_), min(r_, g_, b_)
                if mx - mn > 0.25 and s["text"].strip():
                    rows.append({"file": pdf.name, "page": pno + 1, "issue": "coloured_text", "text_a": s["text"],
                                 "text_b": f"#{s.get('color', 0):06X}", "size_pt": round(s["size"], 2),
                                 "x_mm": round(s["bbox"][0] * PT_MM, 1), "y_mm": round(s["bbox"][1] * PT_MM, 1),
                                 "overlap_w_mm": "", "overlap_h_mm": ""})
                if s["size"] < MIN_PT - 1e-3:
                    rows.append({"file": pdf.name, "page": pno + 1, "issue": "below_5pt", "text_a": s["text"], "text_b": "",
                                 "size_pt": round(s["size"], 2), "x_mm": round(s["bbox"][0] * PT_MM, 1),
                                 "y_mm": round(s["bbox"][1] * PT_MM, 1), "overlap_w_mm": "", "overlap_h_mm": ""})
            for i in range(len(spans)):
                a = spans[i][2]
                for j in range(i + 1, len(spans)):
                    c = spans[j][2]       # spans merged into one line still count: abutting spans stay below TOL_MM
                    w =(min(a["bbox"][2], c["bbox"][2]) - max(a["bbox"][0], c["bbox"][0])) * PT_MM
                    h = (min(a["bbox"][3], c["bbox"][3]) - max(a["bbox"][1], c["bbox"][1])) * PT_MM
                    if w > TOL_MM and h > TOL_MM:
                        rows.append({"file": pdf.name, "page": pno + 1, "issue": "overlap", "text_a": a["text"],
                                     "text_b": c["text"], "size_pt": round(min(a["size"], c["size"]), 2),
                                     "x_mm": round(max(a["bbox"][0], c["bbox"][0]) * PT_MM, 1),
                                     "y_mm": round(max(a["bbox"][1], c["bbox"][1]) * PT_MM, 1),
                                     "overlap_w_mm": round(w, 2), "overlap_h_mm": round(h, 2)})
    return rows


def write_text_report(rows, out=QC / "text_overlaps.csv"):
    cols = ["file", "page", "issue", "text_a", "text_b", "size_pt", "x_mm", "y_mm", "overlap_w_mm", "overlap_h_mm"]
    with open(out, "w", newline="\n", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[qc] {out.relative_to(ROOT)}: {sum(r['issue'] == 'overlap' for r in rows)} overlaps, "
          f"{sum(r['issue'] == 'below_5pt' for r in rows)} spans below {MIN_PT} pt, "
          f"{sum(r['issue'] == 'off_page' for r in rows)} spans leaving the page, "
          f"{sum(r['issue'] == 'coloured_text' for r in rows)} coloured text spans")


if __name__ == "__main__":
    QC.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "figures" / "figures"
    paths = [Path(p) for p in sys.argv[1:]] or sorted(fig_dir.glob("*composite*.jpg")) + sorted(fig_dir.glob("*composite*.pdf"))
    text_rows = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            text_rows += text_issues(p)
        else:
            simulate_image(p)
    write_text_report(text_rows)
    for r in palette_report():
        if r["vision"] != "normal" or r["pairs_below_10"]:
            print(r)
