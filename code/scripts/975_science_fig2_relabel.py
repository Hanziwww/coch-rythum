"""Round 10 (D144) — Science-style copy of the Nature Immunology Fig. 2 (fig2_local_clocks_composite_v5, assembled with
coca-fig, D63, so not drawn by our figure code): the vector PDF is edited in place - the six panel letters a-f become
A-F and figure words in British spelling become American (lib.figstyle.american) - by removing those text spans
(redaction of text only; drawings and images untouched) and setting the new text at the same baseline origin, size and
colour in the same font (DejaVu Sans, embedded from matplotlib). The empty second page of the source is dropped.
Output: figures/science/fig2_local_clocks_composite_v5.{pdf,png,jpg} (used in the Supplementary Materials).
Run: COCH_ROOT=/path/to/coch-rythum python scripts/975_science_fig2_relabel.py
"""
import sys
from pathlib import Path

import matplotlib
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.figstyle import SCIENCE_DIR, american  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "figures" / "figures" / "fig2_local_clocks_composite_v5.pdf"
NAME = "fig2_local_clocks_composite_v5"
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = {"DejaVuSans-Bold": ("dvb", TTF / "DejaVuSans-Bold.ttf"), "DejaVuSans": ("dv", TTF / "DejaVuSans.ttf"),
         "DejaVuSans-Oblique": ("dvi", TTF / "DejaVuSans-Oblique.ttf")}


def main():
    doc = pymupdf.open(SRC)
    doc.select([0])
    page = doc[0]
    spans = [s for b in page.get_text("dict")["blocks"] for ln in b.get("lines", []) for s in ln["spans"]]
    edits = []
    for s in spans:
        t = s["text"]
        letter = (len(t) == 1 and t in "abcdef" and s["font"] == "DejaVuSans-Bold" and abs(s["size"] - 8.0) < 0.1)
        new = t.upper() if letter else american(t)
        if new != t:
            edits.append((s, new))
    assert sum(1 for s, n in edits if len(n) == 1) == 6, "expected six panel letters"
    before = sorted(sp["text"] for sp in spans)
    for s, _ in edits:                                 # the word's own line only (x-height band), so neighbours stay
        x0, _, x1, _ = s["bbox"]
        oy = s["origin"][1]
        page.add_redact_annot(pymupdf.Rect(x0 + 0.1, oy - 0.62 * s["size"], x1 - 0.1, oy - 0.08 * s["size"]))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    for s, new in edits:
        alias, path = FONTS[s["font"]]
        c = s["color"]
        rgb = ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
        page.insert_text(pymupdf.Point(*s["origin"]), new, fontsize=s["size"], fontname=alias, fontfile=str(path),
                         color=rgb)
        print(f"{s['text']!r} -> {new!r}")
    after = sorted(sp["text"] for b in page.get_text("dict")["blocks"] for ln in b.get("lines", [])
                   for sp in ln["spans"])
    expect = sorted([t for t in before if t not in {e[0]["text"] for e in edits}] + [n for _, n in edits])
    assert after == expect, (sorted(set(expect) - set(after)), sorted(set(after) - set(expect)))
    SCIENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = SCIENCE_DIR / f"{NAME}.pdf"
    doc.save(out, garbage=4, deflate=True)
    pix = page.get_pixmap(dpi=200)
    pix.save(SCIENCE_DIR / f"{NAME}.png")
    page.get_pixmap(dpi=600).pil_save(SCIENCE_DIR / f"{NAME}.jpg", quality=95)
    print(f"[fig] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
