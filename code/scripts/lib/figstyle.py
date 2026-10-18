"""Round 9 (D67) — custom figure canvas at Nature specifications for figures built in code rather than from coca-fig
templates: 183 mm wide, at most 170 mm tall, text 5-7 pt (DejaVu Sans, as in the coca-fig Fig. 2), panel letters 8 pt
bold, axes placed in millimetres from the top-left corner, point clouds rasterised at 600 dpi with all text kept as text.

Export: PDF (TrueType fonts, no creation date), SVG (text as text), 600-dpi JPG and a PNG preview.

D144 (Science version): with the environment variable COCH_FIG_STYLE=science every figure is drawn in Science style -
panel letters upper case (a composite can assign its own letters through Canvas.relabel), figure text in American
spelling (applied to every text object and fixed tick label at save) - and written to figures/science/ under its
usual name, so the Nature Immunology files stay untouched.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter, NullLocator  # noqa: E402
import numpy as np  # noqa: E402

MM = 1 / 25.4
FULL, MAX_H = 180.0, 170.0                             # Nature Neuroscience: <= 180 mm wide (D95)
FONT = "DejaVu Sans"
PT, PT_SMALL, PT_LETTER = 6.0, 5.0, 8.0
LW, LW_THIN = 0.5, 0.35
TEXT, TEXT2, AXIS, GRID = "#1A1A1A", "#4D4D4D", "#333333", "#D9D9D9"
DPI = 600
STYLE = os.environ.get("COCH_FIG_STYLE", "nature").strip().lower()      # 'nature' (default) or 'science' (D144)
SCIENCE_DIR = Path(__file__).resolve().parents[2] / "figures" / "science"
# British (Oxford) -> American spelling of figure text in Science style; whole words only, case kept for the first letter
_US_WORDS = {
    "haematopoietic": "hematopoietic", "haematopoiesis": "hematopoiesis", "haemoglobin": "hemoglobin",
    "haematology": "hematology", "programme": "program", "programmes": "programs", "colour": "color",
    "colours": "colors", "coloured": "colored", "grey": "gray", "neighbourhood": "neighborhood",
    "neighbourhoods": "neighborhoods", "labelled": "labeled", "labelling": "labeling", "modelled": "modeled",
    "modelling": "modeling", "signalling": "signaling", "behaviour": "behavior", "behaviours": "behaviors",
    "centre": "center", "centres": "centers", "centred": "centered", "ageing": "aging", "analysed": "analyzed",
    "analyse": "analyze", "analysing": "analyzing", "catalogue": "catalog", "orthologue": "ortholog",
    "orthologues": "orthologs", "favour": "favor", "fibre": "fiber", "fibres": "fibers", "litre": "liter",
    "anaesthesia": "anesthesia", "anaesthetized": "anesthetized", "foetal": "fetal", "tumour": "tumor",
    "artefact": "artifact", "artefacts": "artifacts", "travelling": "traveling", "whilst": "while",
    "acknowledgement": "acknowledgment", "judgement": "judgment", "sceptical": "skeptical", "licence": "license",
    "defence": "defense", "towards": "toward", "per cent": "percent"}
_US_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in _US_WORDS), key=len, reverse=True)) + r")\b",
                    re.IGNORECASE)


# in-figure references to panel letters (Science letters are upper case); '(h)' elsewhere means hours, so only these
_PANEL_REFS = [("flow-replicated (c)", "flow-replicated (C)"), ("flow cohort (c)", "flow cohort (C)"),
               ("flow cohort (h)", "flow cohort (H)")]


def american(s: str) -> str:
    """American spelling of a figure string (Science style); the case of the first letter is kept."""
    if not s or "$" in s:
        return s
    for old, new in _PANEL_REFS:
        s = s.replace(old, new)

    def sub(m):
        w = m.group(0)
        us = _US_WORDS[w.lower()]
        return us[0].upper() + us[1:] if w[0].isupper() else us
    return _US_RE.sub(sub, s)


def setup() -> None:
    plt.rcParams.update({
        "font.family": FONT, "font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT, "xtick.labelsize": PT_SMALL,
        "ytick.labelsize": PT_SMALL, "legend.fontsize": PT_SMALL, "axes.linewidth": LW, "xtick.major.width": LW,
        "ytick.major.width": LW, "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 1.0,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.minor.size": 1.1, "ytick.minor.size": 1.1,
        "xtick.minor.width": LW_THIN, "ytick.minor.width": LW_THIN, "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "ytick.major.pad": 1.0, "axes.labelpad": 1.5, "axes.edgecolor": AXIS, "xtick.color": AXIS, "ytick.color": AXIS,
        "text.color": TEXT, "axes.labelcolor": TEXT, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": DPI, "figure.dpi": 150, "lines.linewidth": LW, "patch.linewidth": LW, "hatch.linewidth": LW,
        "mathtext.default": "regular", "axes.unicode_minus": True})


class Canvas:
    """A figure of width x height mm whose axes and text are placed in millimetres from the top-left corner."""

    def __init__(self, width: float = FULL, height: float = MAX_H):
        setup()
        self.w, self.h = float(width), float(height)
        self.fig = plt.figure(figsize=(self.w * MM, self.h * MM))
        self.letters: list[str] = []
        self.relabel: list | None = None              # Science composites: the letters to draw, in drawing order

    def axes(self, left: float, top: float, width: float, height: float, **kw):
        return self.fig.add_axes((left / self.w, 1 - (top + height) / self.h, width / self.w, height / self.h), **kw)

    def letter(self, x: float, y: float, letter: str, title: str | None = None):
        if self.relabel is not None:                  # composite (D144): next assigned letter; None drops the letter
            letter = self.relabel.pop(0)
            if letter is None:
                if title:
                    self.fig.text(x / self.w, 1 - (y + 0.55) / self.h, title, fontsize=PT, va="top", ha="left",
                                  fontweight="bold", color=TEXT)
                return
        elif STYLE == "science":
            letter = letter.upper()
        self.letters.append(letter)
        self.fig.text(x / self.w, 1 - y / self.h, letter, fontweight="bold", fontsize=PT_LETTER, va="top", ha="left",
                      gid=f"letter_{letter}")
        if title:
            self.fig.text((x + 3.0) / self.w, 1 - (y + 0.55) / self.h, title, fontsize=PT, va="top", ha="left",
                          fontweight="bold", color=TEXT)

    def text(self, x: float, y: float, s: str, size: float = PT, **kw):
        return self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=size, **kw)

    def mark(self, x: float, y: float, s: float = 6.0, marker: str = "o", fc: str = TEXT, ec: str | None = None,
             lw: float = 0.5) -> None:
        """A legend glyph centred at (x, y) mm; s is the scatter size (points squared) so glyphs match the panels."""
        self.fig.add_artist(Line2D([x / self.w], [1 - y / self.h], transform=self.fig.transFigure, marker=marker,
                                   markersize=s ** 0.5, markerfacecolor=fc, markeredgecolor=ec or fc,
                                   markeredgewidth=lw, linestyle="none"))

    def box(self, x: float, y: float, w: float, h: float, fc: str, ec: str = "none", lw: float = 0.0) -> None:
        """A rectangle with its top-left corner at (x, y) mm."""
        self.fig.patches.append(Rectangle((x / self.w, 1 - (y + h) / self.h), w / self.w, h / self.h,
                                          transform=self.fig.transFigure, facecolor=fc, edgecolor=ec, linewidth=lw))

    def line(self, xs, ys, colour: str = TEXT, lw: float = LW, **kw) -> None:
        self.fig.add_artist(Line2D([x / self.w for x in xs], [1 - y / self.h for y in ys], transform=self.fig.transFigure,
                                   color=colour, linewidth=lw, **kw))

    def key(self, x: float, y: float, items, size: float = PT_SMALL, gap: float = 2.4, pad: float = 0.8,
            colour: str = TEXT) -> float:
        """One legend row starting at x mm, centred on y mm. items: (glyphs, label) pairs, glyphs a list of dicts
        {"kind": "mark" | "box" | "vline" | "hline", ...}. Labels are placed by their measured width, so a row never
        overlaps itself. Returns the right end of the row (mm)."""
        for glyphs, label in items:
            for g in glyphs:
                kind = g.get("kind", "mark")
                if kind == "mark":
                    d = g.get("s", 6.0) ** 0.5 * 25.4 / 72
                    self.mark(x + d / 2, y + g.get("dy", 0.0), g.get("s", 6.0), g.get("marker", "o"),
                              g.get("fc", TEXT), g.get("ec"), g.get("lw", 0.5))
                    x += d + g.get("space", 0.5)
                elif kind == "box":
                    b = g.get("size", 1.3)
                    self.box(x, y - b / 2, b, b, g.get("fc", TEXT), g.get("ec", "none"), g.get("lw", 0.0))
                    x += b + 0.5
                elif kind == "vline":
                    hh = g.get("h", 2.0)
                    self.line([x + 0.3, x + 0.3], [y - hh / 2, y + hh / 2], g.get("colour", TEXT), g.get("lw", 1.0))
                    x += 0.6 + 0.5
                elif kind == "hline":
                    ww = g.get("w", 2.4)
                    self.line([x, x + ww], [y, y], g.get("colour", TEXT), g.get("lw", 0.9), linestyle=g.get("ls", "-"))
                    if "mark_fc" in g:                       # line with a centred marker (fit + points glyph)
                        self.mark(x + ww / 2, y, g.get("s", 9.0), g.get("marker", "o"), g["mark_fc"], g.get("colour"),
                                  g.get("mlw", 0.5))
                    x += ww + 0.5
            if label:
                self.text(x + pad - 0.5, y, label, size=size, va="center", ha="left", color=colour)
                x += pad - 0.5 + text_width(label, size)
            x += gap
        return x - gap

    def save(self, out_dir: Path, name: str) -> list[Path]:
        if STYLE == "science":                         # D144: Science style, written beside the Nature files
            out_dir = SCIENCE_DIR
            self.americanize()
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext, kw in [("pdf", {"metadata": {"CreationDate": None}}), ("svg", {"metadata": {"Date": None}}),
                        ("jpg", {"pil_kwargs": {"quality": 95}}), ("png", {"dpi": 200})]:
            p = out_dir / f"{name}.{ext}"
            self.fig.savefig(p, dpi=kw.pop("dpi", DPI), facecolor="white", **kw)
            paths.append(p)
        return paths


    def americanize(self) -> None:
        """American spelling of every text object and every fixed tick label of the figure (Science style)."""
        from matplotlib.text import Text
        from matplotlib.ticker import FixedFormatter
        for ax in self.fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                for fmt in (axis.get_major_formatter(), axis.get_minor_formatter()):
                    if isinstance(fmt, FixedFormatter):
                        fmt.seq = [american(str(v)) for v in fmt.seq]
        for t in self.fig.findobj(Text):
            s = t.get_text()
            us = american(s)
            if us != s:
                t.set_text(us)


_T2P = TextToPath()


P_SYM, R_SYM = r"$\mathit{P}$", r"$\mathit{r}$"    # statistical symbols in figure text (italic; Nature style, D99)
Q_SYM = r"$\mathit{q}$"                             # FDR q-value, italic like P (D102)


def text_width(s: str, size: float = PT, weight: str = "normal", style: str = "normal") -> float:
    """Advance width of s in mm at the given size (DejaVu Sans, as rendered; mathtext such as P_SYM measured as drawn)."""
    w, _, _ = _T2P.get_text_width_height_descent(s, FontProperties(family=FONT, size=size, weight=weight, style=style),
                                                 ismath="$" in s)
    return w * 25.4 / 72


def clean(ax, left: bool = True, bottom: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.set_yticks([])
    if not bottom:
        ax.set_xticks([])


def blank(ax) -> None:
    ax.set_axis_off()


def halo(width: float = 1.2, colour: str = "white"):
    return [patheffects.withStroke(linewidth=width, foreground=colour)]


# ---- shared helpers for the time-of-day figures (Figs 3-6)
LIGHT, DARK = "#FFFFFF", "#4D4D4D"


def ld_bar(cv: Canvas, x: float, y: float, w: float, h: float = 0.9, vertical: bool = False) -> None:
    """Light (ZT0-12, open) / dark (ZT12-24, filled) bar along a 0-24 h axis of length w mm. Vertical bars run upwards
    (ZT0 at the bottom), with (x, y) the top-left corner of a bar of width h."""
    if vertical:
        cv.box(x, y + w / 2, h, w / 2, LIGHT, AXIS, 0.35)
        cv.box(x, y, h, w / 2, DARK, AXIS, 0.35)
    else:
        cv.box(x, y, w / 2, h, LIGHT, AXIS, 0.35)
        cv.box(x + w / 2, y, w / 2, h, DARK, AXIS, 0.35)


def zt_axis(ax, ticks=(0, 6, 12, 18, 24), bar: bool = True) -> None:
    """0-24 h x-axis; with bar=True the LD bar replaces the bottom spine and the tick labels sit below it."""
    ax.set_xlim(0, 24)
    ax.set_xticks(list(ticks), [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    if bar:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=4.2, labelsize=PT_SMALL)


def fits(x_end: float, right: float, what: str) -> None:
    assert x_end <= right + 1e-6, f"{what}: ends at {x_end:.1f} mm, beyond {right:.1f} mm"


def fmt_p(p: float) -> str:
    """p for figure text: two significant figures below 0.1 with trailing zeros kept (0.0025, 0.050, 0.067; text uses the
    same rule, phase F), two decimals above, '<0.001' below 0.001."""
    return "<0.001" if p < 0.001 else f"{p:#.2g}" if p < 0.1 else f"{p:.2f}"


# ---- shared helpers for the cell-type matrices of the Extended Data figures
def header_height(abbrs, size: float = PT_SMALL) -> float:
    """Height (mm) from a panel's top edge to the top of a matrix carrying a type_header, clearing the panel title."""
    return 4.2 + max(text_width(a, size) for a in abbrs) + 2.0


def type_header(cv: Canvas, abbrs, colours, x: float, top: float, q: float, size: float = PT_SMALL) -> None:
    """Class-colour squares and rotated cell-type abbreviations above a matrix whose columns start at x (pitch q mm)."""
    bw = min(q * 0.7, 1.6)
    for i, (a, c) in enumerate(zip(abbrs, colours)):
        cv.box(x + (i + 0.5) * q - bw / 2, top - 1.6, bw, 1.0, c)
        cv.text(x + (i + 0.5) * q, top - 2.0, a, size=size, ha="center", va="bottom", rotation=90)


# ---- D137: axes with more information (Nature figure guide: ticks with units, minor ticks on continuous axes, every
# point shown with its centre and interval; derived axes beside the raw data)
def minor(ax, x: bool = True, y: bool = True, n: int = 2) -> None:
    """Unlabelled minor ticks: n subdivisions between major ticks on linear axes, 2-9 x each decade on log axes."""
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if not on:
            continue
        if axis.get_scale() == "log":
            axis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10)), numticks=20))
            axis.set_minor_formatter(NullFormatter())
        else:
            axis.set_minor_locator(AutoMinorLocator(n))


def marginal_axes(cv: Canvas, left: float, top: float, w: float, h: float, top_h: float = 5.0, right_w: float = 5.0,
                  gap: float = 0.6):
    """Main axes (left, top + top_h + gap, w, h) with a marginal axis above (shared x) and to the right (shared y);
    the marginals carry no spines or ticks."""
    main = cv.axes(left, top + top_h + gap, w, h)
    mx = cv.axes(left, top, w, top_h, sharex=main)
    my = cv.axes(left + w + gap, top + top_h + gap, right_w, h, sharey=main)
    for a in (mx, my):
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return main, mx, my


def density(ax, values, colour: str, orient: str = "x", grid=None, bw: float | None = None, log: bool = False,
            alpha: float = 0.35, lw: float = LW_THIN, scale: float = 1.0) -> None:
    """Kernel density of values drawn as a filled curve along x (orient 'x', curve upwards) or y (orient 'y', curve to
    the right); log=True estimates on log10(values) and draws on the original scale; heights normalised to the max."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    if len(v) < 3 or np.ptp(v) == 0:
        return
    t = np.log10(v) if log else v
    k = gaussian_kde(t, bw_method=bw)
    g = np.linspace(t.min() - 0.5 * t.std(), t.max() + 0.5 * t.std(), 200) if grid is None else (
        np.log10(grid) if log else np.asarray(grid, float))
    f = k(g)
    f = scale * f / f.max()
    gx = 10 ** g if log else g
    if orient == "x":
        ax.fill_between(gx, 0, f, color=colour, alpha=alpha, lw=0)
        ax.plot(gx, f, color=colour, lw=lw)
    else:
        ax.fill_betweenx(gx, 0, f, color=colour, alpha=alpha, lw=0)
        ax.plot(f, gx, color=colour, lw=lw)


def interval(ax, pos: float, est: float, lo: float, hi: float, colour: str = TEXT, orient: str = "v",
             ms: float = 9.0, lw: float = 0.8, marker: str = "o", fc: str | None = None, z: float = 4) -> None:
    """A point estimate with its interval at position pos (orient 'v': vertical interval at x = pos)."""
    if orient == "v":
        ax.plot([pos, pos], [lo, hi], color=colour, lw=lw, solid_capstyle="butt", zorder=z)
        ax.scatter([pos], [est], s=ms, marker=marker, facecolor=fc or colour, edgecolor=colour, linewidth=0.5,
                   zorder=z + 1)
    else:
        ax.plot([lo, hi], [pos, pos], color=colour, lw=lw, solid_capstyle="butt", zorder=z)
        ax.scatter([est], [pos], s=ms, marker=marker, facecolor=fc or colour, edgecolor=colour, linewidth=0.5,
                   zorder=z + 1)


def half_violin(ax, pos: float, values, colour: str, width: float = 0.35, side: str = "right", log: bool = False,
                alpha: float = 0.3, orient: str = "v") -> None:
    """Bootstrap (or posterior) distribution as a half violin beside position pos (estimation plots, Ho et al. 2019)."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    if len(v) < 10 or np.ptp(v) == 0:
        return
    t = np.log10(v) if log else v
    g = np.linspace(np.percentile(t, 0.5), np.percentile(t, 99.5), 160)
    f = gaussian_kde(t)(g)
    f = width * f / f.max()
    gy = 10 ** g if log else g
    sgn = 1 if side == "right" else -1
    if orient == "v":
        ax.fill_betweenx(gy, pos, pos + sgn * f, color=colour, alpha=alpha, lw=0, zorder=2)
    else:
        ax.fill_between(gy, pos, pos + sgn * f, color=colour, alpha=alpha, lw=0, zorder=2)


def null_hist(ax, null, observed: float, colour: str = TEXT2, obs_colour: str = TEXT, bins: int = 30) -> None:
    """Permutation null as a histogram with the observed value as a vertical line (null-distribution insets)."""
    ax.hist(np.asarray(null, float), bins=bins, color=colour, alpha=0.55, lw=0)
    ax.axvline(observed, color=obs_colour, lw=0.8)


def ref_line(ax, value: float, orient: str = "h", colour: str = GRID, lw: float = LW_THIN, ls=(0, (2, 1.5))) -> None:
    """A reference line (1 for ratios, 0 for differences, chance) behind the data."""
    (ax.axhline if orient == "h" else ax.axvline)(value, color=colour, lw=lw, ls=ls, zorder=0)
