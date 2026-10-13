"""Table detection and structure reconstruction (local, deterministic).

Two detectors feed one structured model:

* ruled  - PyMuPDF ``find_tables()`` (vector ruling lines). Cell geometry is mapped onto the grid of row/column
           edges so merged cells get real rowspan/colspan instead of being flattened.
* aligned - borderless / booktabs tables found from text geometry: consecutive text rows that split into
           several horizontally-aligned fragments whose column positions agree. Multi-column prose (two long
           fragments per row) and code are rejected. Horizontal rules, when present, mark the header block.

Every table is a ``Table`` with cell bboxes, row/col indices, spans, header flags, method, confidence and
warnings. Text is never invented: an empty cell is "" (``filled=False``); a cell the detector could not read
is reported in warnings.

Output: a GitHub-style Markdown table when the grid is rectangular with a single header row, otherwise an
HTML table with <th>, rowspan and colspan.
"""
import html
import re
import statistics
from dataclasses import dataclass, field

from .schema import overlap_frac

NUMERIC = re.compile(r"^[\s(\[]*[-+−±~<>≈]?\s*[$€£¥]?\s*\d[\d,.\s]*%?[)\]]*\s*[a-zA-Z%]{0,3}\*{0,3}$")


@dataclass
class Cell:
    row: int
    col: int
    text: str
    bbox: tuple
    rowspan: int = 1
    colspan: int = 1
    header: bool = False


@dataclass
class Table:
    bbox: tuple
    n_rows: int
    n_cols: int
    cells: list
    method: str
    confidence: float
    header_rows: int = 1
    warnings: list = field(default_factory=list)

    @property
    def has_spans(self):
        return any(c.rowspan > 1 or c.colspan > 1 for c in self.cells)

    def grid(self, fill_spans=False):
        g = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            for r in range(c.row, min(self.n_rows, c.row + c.rowspan)):
                for k in range(c.col, min(self.n_cols, c.col + c.colspan)):
                    if (r, k) == (c.row, c.col) or fill_spans:
                        g[r][k] = c.text
        return g

    def rows(self):
        """v1-compatible list-of-lists (span text repeated in covered positions is NOT done: blanks stay blank)."""
        return self.grid(False)

    def to_dict(self):
        return dict(n_rows=self.n_rows, n_cols=self.n_cols, header_rows=self.header_rows, method=self.method,
                    cells=[dict(row=c.row, col=c.col, rowspan=c.rowspan, colspan=c.colspan, text=c.text,
                                bbox=[round(v, 1) for v in c.bbox], header=c.header, filled=bool(c.text)) for c in self.cells])


# ---------------------------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------------------------
def md_cell(t):
    t = re.sub(r"\s*\n\s*", " ", t or "").strip()
    return t.replace("\\", "\\\\").replace("|", "\\|")


def to_markdown(tb):
    g = tb.grid(False)
    if not g:
        return ""
    lines = ["| " + " | ".join(md_cell(c) for c in g[0]) + " |", "|" + "|".join(["---"] * tb.n_cols) + "|"]
    for r in g[1:]:
        lines.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return "\n".join(lines)


def to_html(tb):
    by_row = {}
    for c in tb.cells:
        by_row.setdefault(c.row, []).append(c)
    out = ["<table>"]
    hdr = [r for r in range(tb.n_rows) if r < tb.header_rows]
    if hdr:
        out.append("<thead>")
    for r in range(tb.n_rows):
        if r == tb.header_rows and hdr:
            out.append("</thead><tbody>")
        elif r == 0 and not hdr:
            out.append("<tbody>")
        cells = sorted(by_row.get(r, []), key=lambda c: c.col)
        tds = []
        for c in cells:
            tag = "th" if c.header else "td"
            attrs = (f' rowspan="{c.rowspan}"' if c.rowspan > 1 else "") + (f' colspan="{c.colspan}"' if c.colspan > 1 else "")
            txt = html.escape(re.sub(r"\s*\n\s*", " ", c.text or "").strip(), quote=False)
            tds.append(f"<{tag}{attrs}>{txt}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def serialize(tb):
    if tb.has_spans or tb.header_rows != 1:
        return to_html(tb)
    return to_markdown(tb)


def from_rows(rows, method="model", confidence=0.5, header_rows=1):
    """Build a Table from a plain list-of-lists (e.g. a vision model's rows). No geometry, no spans."""
    rows = [list(r) for r in rows if r]
    nc = max((len(r) for r in rows), default=0)
    cells = []
    for i, r in enumerate(rows):
        for j in range(nc):
            cells.append(Cell(i, j, str(r[j]).strip() if j < len(r) and r[j] is not None else "", (0, 0, 0, 0), header=i < header_rows))
    return Table((0, 0, 0, 0), len(rows), nc, cells, method, confidence, header_rows)


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------
def rules_from_drawings(drawings, page_w, page_h):
    """Horizontal and vertical ruling segments from page.get_drawings(): list of (x0, y0, x1, y1)."""
    hs, vs = [], []
    for d in drawings or []:
        for it in d.get("items", []):
            op = it[0]
            if op == "l":
                p, q = it[1], it[2]
                if abs(p.y - q.y) < 1.0 and abs(p.x - q.x) > 6:
                    hs.append((min(p.x, q.x), p.y, max(p.x, q.x), p.y))
                elif abs(p.x - q.x) < 1.0 and abs(p.y - q.y) > 6:
                    vs.append((p.x, min(p.y, q.y), p.x, max(p.y, q.y)))
            elif op == "re":
                r = it[1]
                w, h = r.x1 - r.x0, r.y1 - r.y0
                if h <= 2.0 and w > 6:
                    hs.append((r.x0, (r.y0 + r.y1) / 2, r.x1, (r.y0 + r.y1) / 2))
                elif w <= 2.0 and h > 6:
                    vs.append(((r.x0 + r.x1) / 2, r.y0, (r.x0 + r.x1) / 2, r.y1))
    big = 0.98
    hs = [h for h in hs if (h[2] - h[0]) < big * page_w]
    return hs, vs


def fragments(seg, gap_em=0.75):
    """Split a text segment into cell fragments at wide gaps. Returns list of (x0, x1, text, chars)."""
    from .textlayer import _chars_to_text
    chars = [c for c in seg.chars]
    out, cur, last = [], [], None
    for c in chars:
        if c.c.isspace():
            if cur:
                cur.append(c)
            continue
        if last is not None and c.x0 - last.x1 > gap_em * max(c.size, last.size, 1):
            out.append(cur); cur = []
        cur.append(c); last = c
    if cur:
        out.append(cur)
    res = []
    for f in out:
        vis = [c for c in f if not c.c.isspace()]
        if vis:
            res.append((min(c.x0 for c in vis), max(c.x1 for c in vis), _chars_to_text(f), vis))
    return res


@dataclass
class _Row:
    y0: float
    y1: float
    base: float
    frags: list      # (x0, x1, text, chars)
    segs: list


def _rows(segs):
    """Group horizontal segments into visual rows by baseline."""
    rows = []
    for s in sorted([s for s in segs if s.horizontal], key=lambda s: s.baseline):
        sz = max(s.size, 1)
        if rows and abs(rows[-1].base - s.baseline) < 0.4 * sz:
            r = rows[-1]
            r.y0, r.y1 = min(r.y0, s.bbox[1]), max(r.y1, s.bbox[3])
            r.frags += fragments(s); r.segs.append(s)
        else:
            rows.append(_Row(s.bbox[1], s.bbox[3], s.baseline, fragments(s), [s]))
    for r in rows:
        # leader dots / stray bullets between cells are layout, not cells
        r.frags = [f for f in r.frags if re.search(r"[\w%$€£¥]", f[2])]
        r.frags.sort(key=lambda f: f[0])
    return [r for r in rows if r.frags]


# ---------------------------------------------------------------------------------------------
# Aligned (borderless / booktabs) detector
# ---------------------------------------------------------------------------------------------
def detect_aligned(segs, hrules, body_size, page_w, exclude=()):
    segs = [s for s in segs if not any(overlap_frac(s.bbox, e) > 0.5 for e in exclude)]
    rows = _rows(segs)
    tables, used = [], set()
    i = 0
    while i < len(rows):
        if len(rows[i].frags) < 2 or _proselike(rows[i], body_size):
            i += 1; continue
        j = i
        multi = 1
        sizes0 = [statistics.median([s.size for s in rows[i].segs])]
        while j + 1 < len(rows):
            nxt, cur = rows[j + 1], rows[j]
            lh = max(1.0, cur.y1 - cur.y0)
            gap = nxt.y0 - cur.y1
            if gap > max(2.2 * lh, 2.0 * body_size) and not (_rule_between(hrules, cur.y1, nxt.y0) and gap < 4.5 * lh):
                break
            nsz = statistics.median([s.size for s in nxt.segs])
            if nsz > 1.2 * statistics.median(sizes0) or nsz < 0.7 * statistics.median(sizes0):
                break           # a heading or footnote in another font size ends the table
            if _proselike(nxt, body_size):
                # a category row ('Translation  Rosetta-stone style ...') is allowed when the table continues after it
                after = rows[j + 2] if j + 2 < len(rows) else None
                if not (after and len(after.frags) >= 3 and not _proselike(after, body_size) and after.y0 - nxt.y1 < 2.2 * lh and multi >= 2):
                    break
            sizes0.append(nsz)
            if len(nxt.frags) >= 2:
                multi += 1
            elif not _short(nxt, body_size):
                break
            j += 1
        block = rows[i:j + 1]
        # trim single-fragment rows at the bottom
        while len(block) > 1 and len(block[-1].frags) < 2:
            block.pop()
        tb = _build_aligned(block, hrules, body_size) if multi >= 3 and len(block) >= 3 else None
        if tb:
            tables.append(tb)
            for r in block:
                for s in r.segs:
                    used.add(id(s))
            i = j + 1
        else:
            i += 1
    return tables, used


def _proselike(row, body):
    if len(row.frags) == 1:
        f = row.frags[0]
        return (f[1] - f[0]) > 22 * body and len(f[2]) > 60
    lens = [len(f[2]) for f in row.frags]
    if len(row.frags) <= 3 and min(lens) > 35:
        return True
    ext = row.frags[-1][1] - row.frags[0][0]
    cover = sum(f[1] - f[0] for f in row.frags) / ext if ext > 0 else 1
    words = sum(len(f[2].split()) for f in row.frags)
    return cover > 0.85 and words >= 8 and sum(lens) > 40


def _short(row, body):
    return all(len(f[2]) <= 40 for f in row.frags)


def _rule_between(hrules, y0, y1):
    return any(y0 - 1 <= h[1] <= y1 + 1 for h in hrules)


def _column_edges(block, body):
    """Columns from x-coverage of fragments; spanning fragments (covering several columns) are ignored first."""
    frs = [f for r in block for f in r.frags]
    if not frs:
        return []
    widths = sorted(f[1] - f[0] for f in frs)
    med = widths[len(widths) // 2]
    narrow = [f for f in frs if (f[1] - f[0]) <= max(3 * med, 6 * body)] or frs
    narrow_ids = {id(f) for f in narrow}
    iv = sorted((f[0], f[1]) for f in narrow)
    cols, cur = [], list(iv[0])
    for a, b in iv[1:]:
        if a <= cur[1] + 0.3 * body:
            cur[1] = max(cur[1], b)
        else:
            cols.append(tuple(cur)); cur = [a, b]
    cols.append(tuple(cur))
    # wide fragments that overlap no column are a real (text) column, not a spanning header
    extra = [(f[0], f[1]) for f in frs if id(f) not in narrow_ids and not any(min(f[1], c[1]) - max(f[0], c[0]) > 0 for c in cols)]
    if extra:
        allc = sorted(cols + extra)
        cols, cur = [], list(allc[0])
        for a, b in allc[1:]:
            if a <= cur[1] + 0.3 * body:
                cur[1] = max(cur[1], b)
            else:
                cols.append(tuple(cur)); cur = [a, b]
        cols.append(tuple(cur))
    return cols


def _split_rivers(block, cols, body):
    """Split a column at a vertical whitespace channel that runs through (nearly) every row of the column -
    e.g. a narrow '#' column whose numbers sit one word-space left of the next column ('1 Easy')."""
    out = []
    for c0, c1 in cols:
        glyphs, rows_in = [], 0
        for r in block:
            cs = [ch for f in r.frags for ch in f[3] if c0 - 0.5 <= (ch.x0 + ch.x1) / 2 <= c1 + 0.5]
            if cs:
                rows_in += 1
                glyphs.append([(ch.x0, ch.x1) for ch in cs])
        if rows_in < 3 or c1 - c0 < 3 * body:
            out.append((c0, c1)); continue
        step = 0.5
        n = int((c1 - c0) / step) + 1
        occ = [0] * n
        for gl in glyphs:
            mark = [False] * n
            for a, b in gl:
                for k in range(max(0, int((a - c0) / step)), min(n, int((b - c0) / step) + 1)):
                    mark[k] = True
            for k in range(n):
                occ[k] += mark[k]
        best, run = None, []
        for k in range(n + 1):
            if k < n and occ[k] <= 0.3 * rows_in:
                run.append(k); continue
            if run and len(run) * step >= 0.25 * body and run[0] > 0 and k < n:
                x = c0 + (run[0] + run[-1]) / 2 * step
                lft = sum(1 for gl in glyphs if any(b <= x for a, b in gl))
                rgt = sum(1 for gl in glyphs if any(a >= x for a, b in gl))
                if lft >= 0.6 * rows_in and rgt >= 0.6 * rows_in and (best is None or len(run) > best[1]):
                    best = (x, len(run))
            run = []
        if best:
            out += [(c0, best[0] - 0.01), (best[0] + 0.01, c1)]
        else:
            out.append((c0, c1))
    return out


def _resplit_frags(block, cols):
    """Cut fragments that straddle a column boundary created by river splitting."""
    from .textlayer import _chars_to_text
    bounds_all = sorted(b[1] for b in cols[:-1])
    for r in block:
        nf = []
        for f in r.frags:
            cut = [b for b in bounds_all if f[0] < b < f[1]]
            # only cut where the fragment really has whitespace at the boundary (a category label that runs
            # across both columns stays whole and becomes a spanning cell)
            vis0 = [c for c in f[3] if c.c.strip()]
            cut = [b for b in cut if any(a.x1 <= b <= c.x0 and c.x0 - a.x1 > 0.15 * max(a.size, 1) for a, c in zip(vis0, vis0[1:]))]
            if not cut:
                nf.append(f); continue
            pieces, cur = [], []
            for ch in f[3]:
                if cur and cut and (ch.x0 + ch.x1) / 2 > cut[0]:
                    pieces.append(cur); cur = []
                    cut.pop(0)
                cur.append(ch)
            if cur:
                pieces.append(cur)
            for pc in pieces:
                vis = [c for c in pc if c.c.strip()]
                if vis:
                    nf.append((min(c.x0 for c in vis), max(c.x1 for c in vis), _chars_to_text(pc), vis))
        r.frags = nf


def _build_aligned(block, hrules, body):
    cols = _column_edges(block, body)
    if len(cols) < 2:
        return None
    rc = _split_rivers(block, cols, body)
    if len(rc) > len(cols):
        cols = rc
        _resplit_frags(block, cols)
    nrows, ncols = len(block), len(cols)
    # reject multi-column prose: few columns, long fragments
    lens = [len(f[2]) for r in block for f in r.frags]
    if ncols <= 3 and statistics.median(lens) > 28:
        return None
    filled_rows = sum(1 for r in block if len(r.frags) >= 2)
    if filled_rows < 0.5 * nrows:
        return None
    # cells of a real table line up: each fragment starts, ends or centres on its column's edge/centre.
    # Justified or ragged prose split at wide word gaps does not.
    frs = [f for r in block for f in r.frags]
    aligned = 0
    for f in frs:
        col = max(cols, key=lambda c: min(f[1], c[1]) - max(f[0], c[0]))
        tol = 1.2 * body
        if abs(f[0] - col[0]) < tol or abs(f[1] - col[1]) < tol or abs((f[0] + f[1]) / 2 - (col[0] + col[1]) / 2) < tol:
            aligned += 1
    align_ratio = aligned / max(1, len(frs))
    # prose split at word gaps covers almost the whole row; table rows have real whitespace between cells
    cover = []
    for r in block:
        if len(r.frags) >= 2:
            ext = r.frags[-1][1] - r.frags[0][0]
            cover.append(sum(f[1] - f[0] for f in r.frags) / ext if ext > 0 else 1)
    if cover and statistics.median(cover) > 0.82:
        return None
    words_per_cell = statistics.mean(len(f[2].split()) for f in frs)
    if align_ratio < 0.75 or (words_per_cell > 5.5 and ncols <= 3):
        return None
    cells, grid_used = [], set()
    for ri, r in enumerate(block):
        for f in r.frags:
            cover = [k for k, c in enumerate(cols) if min(f[1], c[1]) - max(f[0], c[0]) > 0.3 * min(f[1] - f[0], c[1] - c[0])]
            if not cover:
                cx = (f[0] + f[1]) / 2
                cover = [min(range(ncols), key=lambda k: abs((cols[k][0] + cols[k][1]) / 2 - cx))]
            c0, c1 = min(cover), max(cover)
            if any((ri, k) in grid_used for k in range(c0, c1 + 1)):
                # two fragments landed in one column: append text to the existing cell
                ex = next((c for c in cells if c.row == ri and c.col <= c0 < c.col + c.colspan), None)
                if ex:
                    ex.text = (ex.text + " " + f[2]).strip()
                    ex.bbox = (min(ex.bbox[0], f[0]), ex.bbox[1], max(ex.bbox[2], f[1]), ex.bbox[3])
                continue
            for k in range(c0, c1 + 1):
                grid_used.add((ri, k))
            cells.append(Cell(ri, c0, f[2], (f[0], r.y0, f[1], r.y1), 1, c1 - c0 + 1))
    for ri in range(nrows):
        for k in range(ncols):
            if (ri, k) not in grid_used:
                cells.append(Cell(ri, k, "", (cols[k][0], block[ri].y0, cols[k][1], block[ri].y1)))
    # header rows: rows above the first horizontal rule that lies inside the table; else first row
    y_top, y_bot = block[0].y0, block[-1].y1
    x0, x1 = min(c[0] for c in cols), max(c[1] for c in cols)
    inner = sorted(h[1] for h in hrules if y_top + 2 < h[1] < y_bot - 2 and min(h[2], x1) - max(h[0], x0) > 0.5 * (x1 - x0))
    hdr = 1
    if inner:
        hdr = sum(1 for r in block if r.y1 <= inner[0] + 1) or 1
        if hdr > 4 or hdr >= nrows - 1:
            hdr = 1
    for c in cells:
        c.header = c.row < hdr
    # numeric share boosts confidence
    texts = [c.text for c in cells if c.text]
    num = sum(1 for t in texts if NUMERIC.match(t)) / max(1, len(texts))
    align = filled_rows / nrows * align_ratio
    conf = round(min(0.9, 0.45 + 0.25 * align + 0.2 * num + (0.05 if inner else 0)), 2)
    tb = Table((x0, y_top, x1, y_bot), nrows, ncols, sorted(cells, key=lambda c: (c.row, c.col)), "aligned_text", conf, hdr)
    empties = sum(1 for c in cells if not c.text)
    if empties:
        tb.warnings.append(f"{empties} empty cell(s): left blank, nothing inferred.")
    if any(c.colspan > 1 for c in cells):
        tb.warnings.append("Some cells span several columns (merged headers); emitted as HTML colspan.")
    return tb


# ---------------------------------------------------------------------------------------------
# Ruled tables (PyMuPDF find_tables) -> structured model with spans
# ---------------------------------------------------------------------------------------------
def grid_regions(hrules, vrules, pad=4.0):
    """Regions where vector rules actually form a grid: >= 2 vertical rules overlapping in y, crossed by
    >= 2 horizontal rules. find_tables() is expensive, so it only runs inside these regions."""
    vs = sorted(vrules, key=lambda v: v[1])
    clusters = []
    for v in vs:
        for c in clusters:
            if v[1] <= c["y1"] + pad and v[3] >= c["y0"] - pad:
                c["v"].append(v); c["y0"] = min(c["y0"], v[1]); c["y1"] = max(c["y1"], v[3]); break
        else:
            clusters.append(dict(v=[v], y0=v[1], y1=v[3]))
    out = []
    for c in clusters:
        xs = sorted({round(v[0]) for v in c["v"]})
        if len(xs) < 2:
            continue
        x0, x1 = xs[0], xs[-1]
        cross = [h for h in hrules if c["y0"] - pad <= h[1] <= c["y1"] + pad and h[0] <= x0 + pad and h[2] >= x1 - pad]
        if len(cross) >= 2:
            out.append((x0 - pad, c["y0"] - pad, x1 + pad, c["y1"] + pad))
    return out


def detect_ruled(page, segs, has_vrules, hrules=None, vrules=None):
    if not has_vrules:
        return []
    out = []
    regions = grid_regions(hrules or [], vrules or []) if vrules is not None else [None]
    found = []
    for clip in regions:
        try:
            found += page.find_tables(strategy="lines", clip=clip).tables if clip else page.find_tables(strategy="lines").tables
        except Exception:
            continue
    for t in found:
        cb = [tuple(c) for c in (t.cells or []) if c]
        if len(cb) < 4:
            continue
        xs = _cluster(sorted({round(v, 1) for c in cb for v in (c[0], c[2])}))
        ys = _cluster(sorted({round(v, 1) for c in cb for v in (c[1], c[3])}))
        ncols, nrows = len(xs) - 1, len(ys) - 1
        if ncols < 2 or nrows < 2:
            continue
        cells, seen = [], set()
        for c in sorted(set(cb), key=lambda c: (c[1], c[0])):
            c0, c1 = _idx(xs, c[0]), _idx(xs, c[2])
            r0, r1 = _idx(ys, c[1]), _idx(ys, c[3])
            if c1 <= c0 or r1 <= r0 or (r0, c0) in seen:
                continue
            for r in range(r0, r1):
                for k in range(c0, c1):
                    seen.add((r, k))
            cells.append(Cell(r0, c0, _text_in(segs, c), c, r1 - r0, c1 - c0))
        for r in range(nrows):
            for k in range(ncols):
                if (r, k) not in seen:
                    cells.append(Cell(r, k, "", (xs[k], ys[r], xs[k + 1], ys[r + 1])))
        filled = sum(1 for c in cells if c.text) / max(1, len(cells))
        if filled < 0.25:
            continue
        hdr = 1
        try:
            if t.header and t.header.external:
                hdr = 0
        except Exception:
            pass
        top_span = [c for c in cells if c.row == 0 and c.rowspan > 1]
        if top_span:
            hdr = max(hdr, max(c.rowspan for c in top_span))
        for c in cells:
            c.header = c.row < hdr
        tb = Table(tuple(t.bbox), nrows, ncols, sorted(cells, key=lambda c: (c.row, c.col)), "ruled_grid",
                   round(min(0.95, 0.6 + 0.35 * filled), 2), hdr)
        if any(c.rowspan > 1 or c.colspan > 1 for c in cells):
            tb.warnings.append("Merged cells detected from ruling geometry; emitted as HTML rowspan/colspan.")
        out.append(tb)
    return out


def _cluster(vals, tol=2.0):
    out = []
    for v in vals:
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _idx(edges, v):
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


def _text_in(segs, box):
    from .textlayer import _chars_to_text
    lines = []
    for s in segs:
        cs = [c for c in s.chars if box[0] - 1 <= (c.x0 + c.x1) / 2 <= box[2] + 1 and box[1] - 1 <= (c.y0 + c.y1) / 2 <= box[3] + 1]
        if cs:
            lines.append((s.baseline, cs[0].x0, _chars_to_text(cs)))
    lines.sort()
    return " ".join(t for _, _, t in lines if t).strip()


# ---------------------------------------------------------------------------------------------
# Post-processing shared by both detectors
# ---------------------------------------------------------------------------------------------
_SYMBOL_ONLY = re.compile(r"^\s*[$€£¥%]\s*$")


def _col_intervals(tb):
    iv = {}
    for c in tb.cells:
        if c.colspan == 1 and c.text and c.bbox[2] > c.bbox[0]:
            a, b = iv.get(c.col, (c.bbox[0], c.bbox[2]))
            iv[c.col] = (min(a, c.bbox[0]), max(b, c.bbox[2]))
    for k in range(tb.n_cols):
        if k not in iv:
            xs = [c.bbox for c in tb.cells if c.col == k and c.bbox[2] > c.bbox[0]]
            iv[k] = (min(b[0] for b in xs), max(b[2] for b in xs)) if xs else (0, 0)
    return [iv[k] for k in range(tb.n_cols)]


def merge_symbol_columns(tb):
    """A column holding only currency/percent signs is typography, not data: fold '$' into the value to its right
    ('$178') and '%' into the value to its left. Nothing is dropped."""
    changed = True
    while changed and tb.n_cols > 2:
        changed = False
        for k in range(tb.n_cols):
            col = [c for c in tb.cells if c.col == k and c.colspan == 1]
            filled = [c for c in col if c.text]
            if not filled or not all(_SYMBOL_ONLY.match(c.text) for c in filled):
                continue
            pct = all("%" in c.text for c in filled)
            tgt = k - 1 if pct else k + 1
            if not (0 <= tgt < tb.n_cols):
                continue
            for c in filled:
                other = next((o for o in tb.cells if o.row == c.row and o.col <= tgt < o.col + o.colspan), None)
                if other is not None:
                    other.text = (other.text + c.text.strip()) if pct else (c.text.strip() + other.text)
            # remove column k: cells in it disappear, spans crossing it shrink, later columns shift left
            keep = []
            for c in tb.cells:
                if c.col == k and c.colspan == 1:
                    continue
                if c.col <= k < c.col + c.colspan:
                    c.colspan -= 1
                elif c.col > k:
                    c.col -= 1
                keep.append(c)
            tb.cells = keep
            tb.n_cols -= 1
            changed = True
            break
    return tb


def extend_header(tb, segs, hrules, body):
    """Short, column-aligned rows directly above a table are its header (e.g. 'Year Ended' over date columns).
    A horizontal rule right under a header cell defines how many columns it spans."""
    if tb.n_cols < 2:
        return tb, set()
    cols = _col_intervals(tb)
    x0 = min(c[0] for c in cols if c[1] > c[0]); x1 = max(c[1] for c in cols)
    cand = [s for s in segs if s.horizontal and s.bbox[3] <= tb.bbox[1] + 1 and s.bbox[0] >= x0 - 2 * body and s.bbox[2] <= x1 + 2 * body]
    rows = sorted(_rows(cand), key=lambda r: -r.y1)
    # collinear rule pieces with small gaps (one underline broken at column boundaries) act as one rule
    merged = []
    for h in sorted(hrules, key=lambda h: (round(h[1], 0), h[0])):
        if merged and abs(merged[-1][1] - h[1]) < 0.6 and h[0] - merged[-1][2] < max(1.5 * body, 12.0):
            m = merged[-1]; merged[-1] = (m[0], m[1], max(m[2], h[2]), m[3])
        else:
            merged.append(tuple(h))
    pieces = [tuple(h) for h in hrules]
    hrules = merged
    first_col_end = cols[0][1]
    top, added, used = tb.bbox[1], [], set()
    for r in rows:
        lh = max(1.0, r.y1 - r.y0)
        if top - r.y1 > max(2.2 * lh, 1.8 * body) and not _rule_between(hrules, r.y1, top):
            break
        if _proselike(r, body) or any(len(f[2]) > 45 for f in r.frags):
            break
        if not any(f[1] > first_col_end + 0.5 * body for f in r.frags):
            break       # a header row labels data columns; text confined to the stub column is prose/caption
        added.insert(0, r)
        top = r.y0
        for s in r.segs:
            used.add(id(s))
        if len(added) >= 4:
            break
    # rotated column labels (e.g. 45-degree headers) directly above the table: one header row, each label
    # assigned to the column under its starting point
    rot = [s for s in segs if not s.horizontal and s.bbox[3] <= tb.bbox[1] + 0.8 * body and tb.bbox[1] - s.bbox[3] < 3 * body
           and x0 - 2 * body <= s.chars[0].ox <= x1 + 2 * body]
    if rot:
        ry0, ry1 = min(s.bbox[1] for s in rot), max(s.bbox[3] for s in rot)
        frs = [(s.chars[0].ox, s.chars[0].ox + 0.1, s.text, s.chars) for s in rot]
        r = _Row(ry0, ry1, ry1, sorted(frs, key=lambda f: f[0]), rot)
        added = [a for a in added if a.y1 <= ry0 + 1] + [r] + [a for a in added if a.y1 > ry0 + 1]
        for sg in rot:
            used.add(id(sg))
        # horizontal rows further up (group labels like 'ReACT') become reachable once the rotated band is in
        more = sorted(_rows([sg for sg in cand if sg.bbox[3] <= ry0 + 1 and id(sg) not in used]), key=lambda x: -x.y1)
        top2 = ry0
        for x in more:
            lh = max(1.0, x.y1 - x.y0)
            if top2 - x.y1 > max(2.2 * lh, 1.8 * body) or _proselike(x, body) or any(len(f[2]) > 45 for f in x.frags):
                break
            if any(id(sg) in used for sg in x.segs):
                continue
            added.append(x); top2 = x.y0
            for sg in x.segs:
                used.add(id(sg))
            if len(added) >= 5:
                break
    if not added:
        return tb, set()
    added = sorted(added, key=lambda a: a.y0)
    k = len(added)
    for c in tb.cells:
        c.row += k
        c.header = False
    new = []
    for ri, r in enumerate(added):
        occupied = set()
        for f in r.frags:
            fc = (f[0] + f[1]) / 2
            near = lambda h: r.y1 - 1 <= h[1] <= r.y1 + 0.9 * (r.y1 - r.y0) + 2 and h[0] - 1 <= fc <= h[2] + 1
            rotated_row = all(not sg.horizontal for sg in r.segs)
            tw = max(1.0, x1 - x0)
            rule = None if rotated_row else next((h for h in hrules if near(h) and h[2] - h[0] > (f[1] - f[0]) * 1.2 and (h[2] - h[0]) < 0.9 * tw), None)
            if rule and sum(1 for g in r.frags if rule[0] - 1 <= (g[0] + g[1]) / 2 <= rule[2] + 1) > 1:
                # several labels under one long underline: each label owns only its own piece
                rule = next((h for h in pieces if near(h) and h[2] - h[0] > (f[1] - f[0]) * 1.2), None)
            if rule:
                cover = [j for j, c in enumerate(cols) if rule[0] - 2 <= (c[0] + c[1]) / 2 <= rule[2] + 2]
            else:
                cover = [j for j, c in enumerate(cols) if min(f[1], c[1]) - max(f[0], c[0]) > 0]
            if not cover:
                cover = [min(range(len(cols)), key=lambda j: abs((cols[j][0] + cols[j][1]) / 2 - fc))]
            c0, c1 = min(cover), max(cover)
            if any(j in occupied for j in range(c0, c1 + 1)):
                ex = next((c for c in new if c.row == ri and c.col <= c0 < c.col + c.colspan), None)
                if ex:
                    ex.text = (ex.text + " " + f[2]).strip()
                continue
            occupied.update(range(c0, c1 + 1))
            new.append(Cell(ri, c0, f[2], (f[0], r.y0, f[1], r.y1), 1, c1 - c0 + 1, True))
        for j in range(tb.n_cols):
            if j not in occupied:
                new.append(Cell(ri, j, "", (cols[j][0], r.y0, cols[j][1], r.y1), 1, 1, True))
    tb.cells = sorted(new + tb.cells, key=lambda c: (c.row, c.col))
    tb.n_rows += k
    tb.header_rows = k
    tb.bbox = (min(tb.bbox[0], min(f[0] for r in added for f in r.frags)), added[0].y0, tb.bbox[2], tb.bbox[3])
    tb.warnings.append(f"{k} header row(s) above the ruled/aligned body were attached by column alignment.")
    return tb, used
