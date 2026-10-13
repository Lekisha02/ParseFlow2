"""Character-level reading of a PDF page's native text layer (PyMuPDF "rawdict"), normalised into
line segments that keep font, size, baseline and geometry for every character.

Everything downstream (layout, tables, math, header/footer detection) works on these segments, so the
original glyph positions are never thrown away. Coordinates are PDF points (top-left origin).
"""
import re
import statistics
import unicodedata
from dataclasses import dataclass, field

import pymupdf

SUPERSCRIPT, ITALIC, SERIFED, MONO, BOLD = 1, 2, 4, 8, 16

# rawdict flags: expand ligatures (no PRESERVE_LIGATURES), keep whitespace, clip to the visible page,
# and skip images (we read those separately). Dehyphenation is done later, with evidence.
RAW_FLAGS = (pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)

_LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}
_MONO_FONT = re.compile(r"(courier|mono|consol|menlo|cmtt|lmmono|inconsolata|code)", re.I)
_BOLD_FONT = re.compile(r"(bold|black|heavy|semibold|demi|cmbx|cmb\d|-bd|\.b$)", re.I)
_ITAL_FONT = re.compile(r"(italic|oblique|cmti|cmmi|-it|\.i$)", re.I)


@dataclass
class Char:
    c: str
    x0: float
    y0: float
    x1: float
    y1: float
    ox: float          # origin (baseline start) x
    oy: float          # baseline y
    size: float
    font: str
    flags: int


@dataclass
class Segment:
    """A run of characters on one baseline with no large horizontal gap."""
    chars: list
    bbox: tuple
    dir: tuple = (1.0, 0.0)
    gaps: list = field(default_factory=list)   # x positions of wide intra-segment gaps (possible cell boundaries)
    _text: str = None

    @property
    def text(self):
        if self._text is None:
            self._text = _chars_to_text(self.chars)
        return self._text

    @property
    def size(self):
        s = [c.size for c in self.chars if not c.c.isspace()]
        return statistics.median(s) if s else 0.0

    @property
    def baseline(self):
        b = [c.oy for c in self.chars if not c.c.isspace()]
        return statistics.median(b) if b else self.bbox[3]

    def _frac(self, pred):
        cs = [c for c in self.chars if c.c.strip()]
        return sum(1 for c in cs if pred(c)) / len(cs) if cs else 0.0

    @property
    def bold(self):
        return self._frac(lambda c: bool(c.flags & BOLD) or bool(_BOLD_FONT.search(c.font))) >= 0.6

    @property
    def italic(self):
        return self._frac(lambda c: bool(c.flags & ITALIC) or bool(_ITAL_FONT.search(c.font))) >= 0.6

    @property
    def mono(self):
        return self._frac(lambda c: bool(c.flags & MONO) or bool(_MONO_FONT.search(c.font))) >= 0.6

    @property
    def horizontal(self):
        return abs(self.dir[1]) < 0.05 and self.dir[0] > 0

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


def _space_threshold(chars):
    """Gap (in em) above which a missing space is synthesised. Lines that already carry space glyphs use a
    conservative threshold; lines without any (some producers position every word) use the line's own gap
    distribution: letter gaps cluster near zero, word gaps sit well above them."""
    gaps = sorted((b.x0 - a.x1) / max(a.size, b.size, 1) for a, b in zip(chars, chars[1:]) if not a.c.isspace() and not b.c.isspace())
    if len(gaps) < 4:
        return 0.2
    med = gaps[len(gaps) // 2]
    return max(0.1, min(0.25, med + 0.1))


def _chars_to_text(chars):
    out, prev = [], None
    thr = _space_threshold(chars) if chars else 0.25
    for ch in chars:
        if prev is not None and not ch.c.isspace() and not prev.c.isspace():
            gap = ch.x0 - prev.x1
            if gap > thr * max(ch.size, prev.size, 1):
                out.append(" ")
        out.append(ch.c)
        prev = ch
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u200e", "\u200f"}


def _clean_char(c):
    if c in _ZERO_WIDTH:
        return ""
    if c in _LIG:
        return _LIG[c]
    if c in (" ", " ", " ", " ", " ", " ", " "):
        return " "
    if c == "­":   # soft hyphen: a line-break hyphen inserted by the typesetter
        return "-"
    return c


def _line_box(chars):
    return (min(c.x0 for c in chars), min(c.y0 for c in chars), max(c.x1 for c in chars), max(c.y1 for c in chars))


def read_segments(page, split_gap_em=1.6, cell_gap_em=0.9, ocr_layer=False):
    """Return list[Segment] for the page. Invisible text (alpha 0) is skipped and counted separately.

    Lines are split into segments at horizontal gaps wider than split_gap_em * font size; narrower but
    still wide gaps (> cell_gap_em) are remembered in Segment.gaps for table detection.
    ocr_layer=True reads an invisible OCR text layer: its placeholder font says nothing about style, so
    bold/italic/monospace flags are cleared.
    """
    d = page.get_text("rawdict", flags=RAW_FLAGS)
    segs, invisible = [], 0
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            chars = []
            for sp in ln.get("spans", []):
                if sp.get("alpha", 255) == 0:
                    invisible += len(sp.get("chars", []))
                    continue
                size, font, flags = float(sp.get("size", 0) or 0), sp.get("font", ""), int(sp.get("flags", 0))
                if ocr_layer:
                    font, flags = "ocr-layer", 0
                for ch in sp.get("chars", []):
                    c = _clean_char(ch["c"])
                    if not c:
                        continue
                    x0, y0, x1, y1 = ch["bbox"]
                    ox, oy = ch["origin"]
                    if len(c) > 1:      # expanded ligature: split the glyph box evenly
                        w = (x1 - x0) / len(c)
                        for k, cc in enumerate(c):
                            chars.append(Char(cc, x0 + k*w, y0, x0 + (k+1)*w, y1, ox + k*w, oy, size, font, flags))
                    else:
                        chars.append(Char(c, x0, y0, x1, y1, ox, oy, size, font, flags))
            if not any(c.c.strip() for c in chars):
                continue
            dr = tuple(ln.get("dir", (1.0, 0.0)))
            horiz = abs(dr[1]) < 0.05 and dr[0] > 0
            if horiz:
                chars = compose_accents(_dedupe(chars))
                for run in _monotonic_runs(chars):
                    for part, gaps in _split_on_gaps(run, split_gap_em, cell_gap_em):
                        segs.append(Segment(part, _line_box(part), dr, gaps))
            else:
                segs.append(Segment(chars, _line_box(chars), dr))
    return segs, invisible


def _monotonic_runs(chars):
    """Keep the PDF's own character order (it follows the text flow). Start a new run where the pen jumps
    backwards or to another baseline, instead of sorting by x (which scrambles overlapping lines)."""
    runs, cur = [], []
    for c in chars:
        if cur:
            p = cur[-1]
            sz = max(c.size, p.size, 1)
            if c.x0 < p.x0 - 0.5 * sz or abs(c.oy - p.oy) > 0.6 * sz and not c.c.isspace() and not p.c.isspace() and c.size >= 0.8 * p.size and p.size >= 0.8 * c.size:
                runs.append(cur); cur = []
        cur.append(c)
    if cur:
        runs.append(cur)
    return runs


_SPACING_ACCENT = {"\u00b4": "\u0301", "`": "\u0300", "\u02c6": "\u0302", "\u02dc": "\u0303", "\u00a8": "\u0308", "\u02c7": "\u030c",
                   "\u02d8": "\u0306", "\u00b8": "\u0327", "\u02da": "\u030a", "\u00af": "\u0304", "\u02d9": "\u0307", "\u02dd": "\u030b",
                   "\u02db": "\u0328", "^": "\u0302", "~": "\u0303"}


def compose_accents(chars):
    """Typesetters often draw an accent as its own glyph over the letter ('a' + '´'). Merge such pairs into the
    precomposed character when the accent sits over (or under, for cedilla/ogonek) a neighbouring letter."""
    out = list(chars)
    i = 0
    while i < len(out):
        c = out[i]
        comb = _SPACING_ACCENT.get(c.c)
        if comb:
            cx = (c.x0 + c.x1) / 2
            best, best_d = None, None
            for j in (i - 1, i + 1):
                if 0 <= j < len(out) and out[j].c.isalpha():
                    l = out[j]
                    pad = 0.15 * max(l.size, 1)
                    if l.x0 - pad <= cx <= l.x1 + pad:
                        d = abs(cx - (l.x0 + l.x1) / 2)
                        if best is None or d < best_d:
                            best, best_d = j, d
            if best is not None:
                l = out[best]
                below = comb in ("\u0327", "\u0328")
                vertical_ok = (c.oy <= l.oy + 0.1 * l.size) if not below else True
                if c.c not in "^~" or vertical_ok and c.y1 < l.oy - 0.3 * l.size:
                    if vertical_ok:
                        new = unicodedata.normalize("NFC", l.c + comb)
                        out[best] = Char(new, l.x0, min(l.y0, c.y0), l.x1, l.y1, l.ox, l.oy, l.size, l.font, l.flags)
                        del out[i]
                        continue
        i += 1
    return out


def _dedupe(chars):
    """Drop duplicate glyphs drawn twice at (almost) the same spot (fake bold / shadow text)."""
    out = []
    for c in chars:
        if out and c.c == out[-1].c and not c.c.isspace() and abs(c.ox - out[-1].ox) < 0.15 * max(c.size, 1) and abs(c.oy - out[-1].oy) < 0.15 * max(c.size, 1):
            continue
        out.append(c)
    return out


def _split_on_gaps(chars, split_em, cell_em):
    """Split a baseline-sorted char run where the gap between consecutive *visible* glyphs is wide."""
    parts, cur, gaps, last = [], [], [], None
    for c in chars:
        if c.c.isspace():
            if cur:
                cur.append(c)
            continue
        if last is not None:
            gap = c.x0 - last.x1
            sz = max(c.size, last.size, 1)
            if gap > split_em * sz:
                while cur and cur[-1].c.isspace():
                    cur.pop()
                parts.append((cur, gaps))
                cur, gaps = [], []
            elif gap > cell_em * sz:
                gaps.append((last.x1 + c.x0) / 2)
        cur.append(c)
        last = c
    while cur and cur[-1].c.isspace():
        cur.pop()
    if cur:
        parts.append((cur, gaps))
    return [(p, g) for p, g in parts if p]


# ---------------------------------------------------------------------------------------------
# Text-layer quality: is the encoded text trustworthy, or is it a broken font map / bad hidden OCR?
# ---------------------------------------------------------------------------------------------
_WORD = re.compile(r"[^\W\d_]+", re.U)


def text_quality(text):
    """Return (score 0..1, details). Low scores mean the text layer is probably garbage
    (unmapped glyphs, private-use codes, implausible words). No dictionary needed."""
    t = text or ""
    vis = [c for c in t if not c.isspace()]
    if not vis:
        return 0.0, dict(chars=0)
    bad = sum(1 for c in vis if c == "�" or 0xE000 <= ord(c) <= 0xF8FF or unicodedata.category(c) in ("Cc", "Co", "Cn"))
    cid = len(re.findall(r"\(cid:\d+\)", t))
    words = _WORD.findall(t)
    latin = [w for w in words if all(ord(ch) < 0x250 for ch in w) and len(w) >= 3]
    implausible = 0
    for w in latin:
        lw = w.lower()
        no_vowel = not re.search(r"[aeiouy]", lw)
        weird_case = re.search(r"[a-z][A-Z]{2,}|[A-Z][a-z]+[A-Z][a-z]*[A-Z]", w) is not None
        repeats = re.search(r"(.)\1\1\1", lw) is not None
        if no_vowel or weird_case or repeats:
            implausible += 1
    junk = (bad + 5 * cid) / len(vis)
    imp = implausible / len(latin) if latin else 0.0
    score = max(0.0, 1.0 - 3.0 * junk - 1.2 * max(0.0, imp - 0.08))
    return round(score, 3), dict(chars=len(vis), bad_glyphs=bad, cid=cid, words=len(latin), implausible=implausible)
