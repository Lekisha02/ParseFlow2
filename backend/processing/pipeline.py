"""DETECT -> ANALYZE -> ROUTE -> EXTRACT -> ASSEMBLE -> VERIFY -> OUTPUT. Every processor emits Digital Twin blocks.

Routing is per page and per region, cheapest suitable method first:
  digital page with a trustworthy text layer   -> native text + layout analysis (no rendering needed)
  page whose text layer is broken/implausible  -> adaptive OCR (text layer kept as evidence)
  scanned page with an embedded OCR layer      -> embedded layer when it is plausible, else Tesseract
  scanned page                                 -> adaptive OCR
  tables                                       -> ruled-grid or aligned-text reconstruction
  equations                                    -> LaTeX rebuilt from glyph geometry (digital) / optional local
                                                  recognizer (scans) / OCR text kept + flagged
  figures, handwriting, diagrams               -> vision model when configured, else OCR labels / flagged
Bounding boxes are in page-image pixels (PDF points * Z), the same space as v1 and the frontend viewer.
"""
import json
import os
import re
import statistics
import time
from contextlib import contextmanager

import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi

from . import headers_footers, layout, mathtex, ocr, recognizers, tables, textlayer, verifier, vision
from . import markdown as mdgen
from .schema import Z, make_block, overlap_frac, scale, union, verdict

STAGES = ["DETECT", "ANALYZE", "ROUTE", "EXTRACT", "ASSEMBLE", "VERIFY", "OUTPUT"]
VIS_EXT = {"handwritten_text": "vision_handwriting", "handwritten_code": "vision_handwriting", "paragraph": "vision_handwriting", "heading": "vision_handwriting",
           "list": "vision_handwriting", "equation": "math_vision", "diagram": "vision_diagram", "figure": "vision_diagram", "chart": "vision_chart", "unknown": "vision_diagram", "table": "vision_table"}
PREFIX = {"table": "tbl", "chart": "chart", "diagram": "diag", "equation": "eq", "handwritten_text": "hand", "handwritten_code": "code", "figure": "fig",
          "unknown": "unk", "caption": "cap", "header": "hdr", "footer": "ftr", "page_number": "pgn", "margin": "mrg", "code": "code"}


class PFError(Exception):
    def __init__(self, code, title, message):
        super().__init__(message); self.code, self.title, self.message = code, title, message

    def dict(self):
        return dict(code=self.code, title=self.title, message=self.message)


class Progress:
    def __init__(self):
        self.stages = [dict(name=s, state="pending", detail="") for s in STAGES]
        self.status = "running"; self.error = None; self.timings = {}

    def set(self, name, state, detail=""):
        for s in self.stages:
            if s["name"] == name:
                s["state"], s["detail"] = state, detail or s["detail"]


@contextmanager
def _stage(prog, name, timings):
    prog.set(name, "running")
    t0 = time.perf_counter()
    yield
    timings[name] = round(timings.get(name, 0.0) + time.perf_counter() - t0, 4)


def _env_int(k, d):
    try:
        return int(os.getenv(k, str(d)))
    except ValueError:
        return d


def open_source(path, filename):
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("pdf", "png", "jpg", "jpeg"):
        raise PFError("PF-204", "Unsupported Document", "ParseFlow could not process this file type. Supported: PDF, PNG, JPG/JPEG.")
    if os.path.getsize(path) == 0:
        raise PFError("PF-205", "Empty Document", "The file contains no data.")
    if ext == "pdf":
        try:
            d = pymupdf.open(path)
        except Exception:
            raise PFError("PF-201", "Corrupt Document", "The PDF could not be opened. It may be damaged or not a real PDF.")
        if d.needs_pass:
            raise PFError("PF-202", "Encrypted Document", "The PDF is password protected.")
        if d.page_count == 0:
            raise PFError("PF-205", "Empty Document", "The PDF has no pages.")
        return "pdf", d
    try:
        im = Image.open(path); im.load(); im = im.convert("RGB")
    except Exception:
        raise PFError("PF-201", "Corrupt Document", "The image could not be decoded.")
    if max(im.size) > 3400:
        im.thumbnail((3400, 3400))
    return "image", im


def page_count(path, filename):
    kind, src = open_source(path, filename)
    try:
        return src.page_count if kind == "pdf" else 1
    finally:
        if kind == "pdf":
            src.close()


# ---------------------------------------------------------------------------------------------
# Page analysis helpers
# ---------------------------------------------------------------------------------------------
def analyze_page(page):
    """Cheap page metrics (no rendering): text amount, hidden-text layer, image coverage, text-layer quality."""
    W, H = page.rect.width, page.rect.height
    segs, invisible = textlayer.read_segments(page)
    vis_text = " ".join(s.text for s in segs)
    chars = len(re.sub(r"\s", "", vis_text))
    cov, best_img_zoom = 0.0, 0.0
    try:
        for i in page.get_image_info():
            b = i["bbox"]
            a = max(0, min(b[2], W) - max(b[0], 0)) * max(0, min(b[3], H) - max(b[1], 0))
            cov += a
            bw = max(1e-3, b[2] - b[0])
            if a > 0.3 * W * H and i.get("width"):
                best_img_zoom = max(best_img_zoom, i["width"] / bw)
    except Exception:
        pass
    cov = min(1.0, cov / (W * H))
    q, qd = textlayer.text_quality(vis_text)
    hidden = ""
    if invisible > 20:
        hidden = page.get_text("text")
    return dict(segs=segs, chars=chars, invisible=invisible, image_coverage=round(cov, 2), text_quality=q, quality_detail=qd,
                img_zoom=best_img_zoom, hidden_text=hidden, W=W, H=H)


def route_page(a, ocr_ok):
    """Decide how to read a page. Returns (route, reason)."""
    if a["chars"] >= 30 and a["text_quality"] >= 0.6 and not (a["image_coverage"] > 0.85 and a["chars"] < 80):
        return "digital", "visible text layer passes quality checks"
    if 0 < a["chars"] < 30 and a["text_quality"] >= 0.6 and a["image_coverage"] < 0.3 and a["invisible"] <= 20:
        return "digital", "sparse page: short but trustworthy text layer and no large images"
    if a["chars"] >= 30 and a["text_quality"] < 0.6:
        return ("ocr" if ocr_ok else "digital"), f"visible text layer looks broken (quality {a['text_quality']:.2f})"
    if a["invisible"] > 20:
        hq, _ = textlayer.text_quality(a["hidden_text"])
        if hq >= 0.75 or not ocr_ok:
            return "embedded_ocr", f"scanned page with a plausible embedded OCR layer (quality {hq:.2f})"
        return "ocr", f"scanned page; embedded OCR layer implausible (quality {hq:.2f})"
    return ("ocr" if ocr_ok else "unreadable"), "no usable text layer"


def detect_ink_regions(img, ocr_boxes, max_side=1800):
    """Regions with ink that OCR could not explain (handwriting, diagrams, charts). Ruled lines are removed first.
    Large images are analysed on a downsampled copy (morphology cost grows with pixel count)."""
    k = min(1.0, max_side / max(img.size))
    if k < 1.0:
        small = img.convert("L").resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.BILINEAR)
        res = detect_ink_regions(small, [tuple(v * k for v in b) for b in ocr_boxes], max_side)
        return [tuple(min(lim, v / k) for v, lim in zip(r, (img.width, img.height, img.width, img.height))) for r in res]
    g = np.array(img.convert("L")); thr = min(170, np.median(g) * 0.65)
    mask = g < thr
    mask &= ~ndi.binary_dilation(ndi.binary_opening(mask, structure=np.ones((1, 90))) | ndi.binary_opening(mask, structure=np.ones((90, 1))), iterations=2)
    for (x0, y0, x1, y1) in ocr_boxes:
        mask[max(0, int(y0) - 3):int(y1) + 3, max(0, int(x0) - 3):int(x1) + 3] = False
    f = 4; h, w = (mask.shape[0] // f) * f, (mask.shape[1] // f) * f
    coarse = mask[:h, :w].reshape(h // f, f, w // f, f).max(axis=(1, 3))
    lab, n = ndi.label(ndi.binary_dilation(coarse, structure=np.ones((3, 3)), iterations=10))
    out = []
    for sl in ndi.find_objects(lab):
        y0, y1, x0, x1 = sl[0].start * f, sl[0].stop * f, sl[1].start * f, sl[1].stop * f
        ink = int(mask[y0:y1, x0:x1].sum()); bw, bh = x1 - x0, y1 - y0
        if ink < 400 or bw < 24 or bh < 24 or bw * bh > 0.9 * img.size[0] * img.size[1]:
            continue
        out.append((max(0, x0 - 8), max(0, y0 - 8), min(img.size[0], x1 + 8), min(img.size[1], y1 + 8)))
    return out


def figures(page, avoid, segs, drawings=None):
    """Image and vector-drawing regions large enough to be figures (merged first, filtered after)."""
    W, H = page.rect.width, page.rect.height
    rects, raster = [], []
    try:
        raster = [tuple(i["bbox"]) for i in page.get_image_info()]
        rects += raster
    except Exception:
        pass
    try:
        rects += [tuple(r) for r in (page.cluster_drawings(drawings=drawings) if drawings is not None else page.cluster_drawings())]
    except Exception:
        pass
    rects = [r for r in rects if (r[2] - r[0]) >= 3 and (r[3] - r[1]) >= 3 and not any(overlap_frac(r, a) > 0.5 for a in avoid)]
    if len(rects) > 300:            # extremely busy vector pages (maps, plots): one merged region per cluster pass
        rects = sorted(rects, key=lambda r: (r[1], r[0]))[:300]
    g, merged = 12, True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] - g < b[2] and b[0] - g < a[2] and a[1] - g < b[3] and b[1] - g < a[3]:
                    rects[i] = union(a, b); del rects[j]; merged = True; break
            if merged:
                break
    out = []
    for r in rects:
        w, h = r[2] - r[0], r[3] - r[1]
        if w < 40 or h < 30 or not (0.015 * W * H <= w * h <= 0.92 * W * H):
            continue
        # vector boxes that hold prose (cards, shaded sidebars, framed text) are text containers, not figures
        prose = sum(1 for s in segs if overlap_frac(s.bbox, r) > 0.8 and len(s.text) > 30 and len(s.text.split()) >= 5)
        is_raster = any(overlap_frac(r, ib) > 0.5 for ib in raster)
        if prose >= 3 and not is_raster:
            continue
        out.append(r)
    return out


class OcrLine:
    """An OCR text line shaped like textlayer.Segment (PDF-point coordinates), so layout/table/furniture code is shared."""

    def __init__(self, words, k):
        # words: list of dict(text, conf, box) in image px; k = px per PDF point
        ws = sorted(words, key=lambda w: w["box"][0])
        self.words = ws
        hs = [w["box"][3] - w["box"][1] for w in ws]
        h = statistics.median(hs) / k if hs else 10
        sz = max(4.0, 0.8 * h)
        self.chars = [textlayer.Char(w["text"] + " ", w["box"][0] / k, w["box"][1] / k, w["box"][2] / k, w["box"][3] / k,
                                     w["box"][0] / k, w["box"][3] / k, sz, "ocr", 0) for w in ws]
        self.bbox = (min(c.x0 for c in self.chars), min(c.y0 for c in self.chars), max(c.x1 for c in self.chars), max(c.y1 for c in self.chars))
        self.dir = (1.0, 0.0)
        self.gaps = []
        self.conf = sum(w["conf"] for w in ws) / len(ws)
        self._size = sz
        self.text = " ".join(w["text"] for w in ws)
        self.baseline = statistics.median(c.oy for c in self.chars)
        self.bold = self.italic = self.mono = False
        self.horizontal = True

    @property
    def size(self):
        return self._size

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]


_GOOD = re.compile(r"^[^\W\d_]{3,}[.,;:!?)\]'\"]*$|^\(?\d[\d,.:/%-]*\)?[.,;:]?$", re.U)


def ocr_lines(words, k, min_conf=0.25):
    """Group OCR words into lines and drop noise: texture, photos and blank paper produce short,
    low-confidence, implausible 'words'. A line is kept when it carries real evidence of text."""
    groups = {}
    for w in words:
        if w["conf"] < min_conf and not re.search(r"[A-Za-z]{3,}", w["text"]):
            continue
        if not re.search(r"\w", w["text"]) and w["conf"] < 0.5:
            continue
        groups.setdefault((w["block"], w["par"], w["line"]), []).append(w)
    out = []
    for ws in groups.values():
        good = [w for w in ws if _GOOD.match(w["text"]) and w["conf"] >= 0.6]
        mean = sum(w["conf"] for w in ws) / len(ws)
        plaus = sum(1 for w in ws if _GOOD.match(w["text"])) / len(ws)
        formula = mean >= 0.6 and any(re.search(r"[=<>≤≥≈]", w["text"]) for w in ws) and any(re.search(r"\w", w["text"]) for w in ws)
        if len(good) >= 2 or (len(good) == 1 and (len(ws) <= 2 and mean >= 0.7)) or (mean >= 0.8 and plaus >= 0.5) or formula:
            out.append(OcrLine(ws, k))
    return out


# ---------------------------------------------------------------------------------------------
# Text with inline math
# ---------------------------------------------------------------------------------------------
def _visual_lines(lines):
    """Merge small attached segments (sub/superscripts split off by the PDF) into their host line."""
    lines = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    hosts = []
    for l in lines:
        h = None
        for hl in hosts:
            off = abs(l.baseline - hl["base"])
            script_like = (len(l.text) <= 15 or (l.bbox[2] - l.bbox[0]) < 0.35 * (hl["bbox"][2] - hl["bbox"][0])) and 0.1 * hl["size"] < off < 0.75 * hl["size"]
            if l.size < 0.85 * hl["size"] and l.bbox[1] < hl["bbox"][3] and l.bbox[3] > hl["bbox"][1] and script_like:
                h = hl; break
            if abs(l.baseline - hl["base"]) < 0.35 * max(l.size, hl["size"]):
                h = hl; break
        if h is None:
            hosts.append(dict(chars=list(getattr(l, "chars", [])), size=l.size, base=l.baseline, bbox=l.bbox, segs=[l]))
        else:
            h["chars"] += list(getattr(l, "chars", [])); h["segs"].append(l)
            h["bbox"] = union(h["bbox"], l.bbox)
    return hosts


def para_text(para, hrules, use_math=True):
    """Return (text, raw_text, has_math, transforms) for a Para of Segments / OcrLines."""
    if isinstance(para.lines[0], OcrLine):
        raw = [l.text for l in para.lines]
        return None, raw, False, []
    vlines, raw_lines, has_math = [], [], False
    for vl in _visual_lines(para.lines):
        chars = sorted(vl["chars"], key=lambda c: c.x0)
        raw_lines.append(textlayer._chars_to_text(chars))
        if not use_math:
            vlines.append(raw_lines[-1]); continue
        parts = []
        for is_m, cs in mathtex.inline_runs(chars):
            if is_m:
                lx, conf, _ = mathtex.build_latex(cs, [r for r in hrules if any(abs(r[1] - c.oy) < 2 * c.size and r[0] - 2 <= c.x0 <= r[2] + 2 for c in cs)])
                if lx and conf >= 0.5:
                    parts.append(" \\(" + lx + "\\) "); has_math = True; continue
            parts.append(textlayer._chars_to_text(cs))
        vlines.append(re.sub(r"\s+", " ", "".join(parts)).strip())
    return vlines, raw_lines, has_math, []


# ---------------------------------------------------------------------------------------------
# Display equations on digital pages
# ---------------------------------------------------------------------------------------------
def display_equations(segs, hrules, body_size, col_ranges):
    """Group math-dense, isolated lines (plus their scripts/limits and fraction rules) into equation regions."""
    cand = []
    for s in segs:
        if not s.horizontal:
            continue
        strong, weak, text = mathtex.line_math_profile(s.chars)
        if strong == 0:
            continue
        if text == 0:
            cand.append(s); continue
        if text <= 1 and strong >= 2:
            # one connective word ('where', 'and') is allowed only on an indented line (displayed, not prose)
            ov = [o.bbox[0] for o in segs if len(o.text) >= 40 and min(o.bbox[2], s.bbox[2]) - max(o.bbox[0], s.bbox[0]) > 0]
            margin = min(ov) if ov else min(o.bbox[0] for o in segs)
            if s.bbox[0] > margin + 2.0 * body_size:
                cand.append(s)
    if not cand:
        return []
    # group candidates that belong to one display (same band, close horizontally), then attach nearby
    # math pieces (limits, scripts, fraction parts) - each piece to exactly one group
    groups = []
    for s in sorted(cand, key=lambda s: (s.bbox[1], s.bbox[0])):
        sz = max(s.size, 6)
        best = None
        for g in groups:
            gb = g["bbox"]
            vgap = max(s.bbox[1] - gb[3], gb[1] - s.bbox[3])
            hgap = max(s.bbox[0] - gb[2], gb[0] - s.bbox[2])
            if vgap < 0.8 * sz and hgap < 6 * sz:
                best = g; break
        if best:
            best["segs"].append(s); best["bbox"] = union(best["bbox"], s.bbox)
        else:
            groups.append(dict(segs=[s], bbox=s.bbox))
    # transitive merge of groups on the same display band
    def _close(a, b):
        sz = max(6, body_size)
        vov = min(a[3], b[3]) - max(a[1], b[1])
        hgap = max(b[0] - a[2], a[0] - b[2])
        vgap = max(b[1] - a[3], a[1] - b[3])
        if vov > 0.5 * min(a[3] - a[1], b[3] - b[1]):
            return hgap < 12 * sz
        return vgap < 0.8 * sz and hgap < 2 * sz
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _close(groups[i]["bbox"], groups[j]["bbox"]):
                    groups[i]["segs"] += groups[j]["segs"]; groups[i]["bbox"] = union(groups[i]["bbox"], groups[j]["bbox"])
                    del groups[j]; merged = True; break
            if merged:
                break
    taken = {id(x) for g in groups for x in g["segs"]}
    pool = [s for s in segs if s.horizontal and id(s) not in taken and len(s.text) < 80]
    changed = True
    while changed:
        changed = False
        for s in list(pool):
            sz = max(s.size, 6)
            st, wk, tx = mathtex.line_math_profile(s.chars)
            fits = []
            for g in groups:
                gb = g["bbox"]
                grown = (gb[0] - 1.5 * sz, gb[1] - 1.3 * sz, gb[2] + 1.5 * sz, gb[3] + 1.3 * sz)
                row = (min(gb[3], s.bbox[3]) - max(gb[1], s.bbox[1])) / max(1e-3, s.bbox[3] - s.bbox[1])
                hgap = max(s.bbox[0] - gb[2], gb[0] - s.bbox[2])
                if (tx <= 1 and overlap_frac(s.bbox, grown) > 0.6) or (st >= 1 and tx <= 2 and row >= 0.6 and hgap < 3 * sz):
                    fits.append((overlap_frac(s.bbox, gb) + row, g))
            if fits:
                g = max(fits, key=lambda t: t[0])[1]
                g["segs"].append(s); g["bbox"] = union(g["bbox"], s.bbox)
                pool.remove(s); changed = True
    out = []
    for g in groups:
        box = g["bbox"]
        chars = [c for s in g["segs"] for c in s.chars]
        rules = [r for r in hrules if box[0] - 3 <= r[0] and r[2] <= box[2] + 3 and box[1] - 2 <= r[1] <= box[3] + 2]
        out.append(dict(segs=g["segs"], bbox=box, chars=chars, rules=rules))
    return out


def _split_rows(chars, rules, size):
    """Split a display region into separate equation lines at clear vertical gaps in the ink
    (glyph boxes + fraction rules), so numerators/denominators stay with their own line."""
    vis = [c for c in chars if c.c.strip()]
    if not vis:
        return []
    boxes = [(c.oy - 0.72 * c.size, c.oy + 0.22 * c.size) for c in vis] + [(r[1] - 0.5, r[1] + 0.5) for r in rules]
    boxes.sort()
    cuts, end = [], None
    for a, b in boxes:
        if end is not None and a - end > 0.35 * size:
            cuts.append((end + a) / 2)
        end = b if end is None else max(end, b)
    if not cuts:
        return [(vis, list(rules))]
    edges = [-1e9] + cuts + [1e9]
    rows = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        rc = [c for c in vis if lo <= c.oy - 0.25 * c.size < hi]
        rr = [r for r in rules if lo <= r[1] < hi]
        if rc:
            rows.append((rc, rr))
    return rows


# ---------------------------------------------------------------------------------------------
# Vision / fallback for visual regions (unchanged contract from v1)
# ---------------------------------------------------------------------------------------------
def vision_blocks(crop_img, bbox, hint, state, tag):
    """Specialised extraction for one visual region. Falls back to a flagged, evidence-preserving block."""
    cx0, cy0, cx1, cy1 = bbox; cw, ch = cx1 - cx0, cy1 - cy0
    err = None
    if state["vision_left"] > 0 and vision.available():
        state["vision_left"] -= 1; state["vision_used"] += 1
        items, err = vision.analyze_crop(crop_img)
        blocks = []
        for it in items:
            t = it.get("type") if it.get("type") in vision.VALID else "unknown"
            rows = None
            if t == "table":
                rr = it.get("rows")
                rows = [[str(c or "").strip() for c in r] for r in rr if isinstance(r, list)] if isinstance(rr, list) else None
                rows = [r for r in (rows or []) if any(r)]
                if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
                    t, rows = "unknown", None
            content = it.get("content") or (it.get("description") if t in ("diagram", "chart", "figure", "unknown") else None)
            if t == "table":
                content = f"Table with {len(rows)} rows"
            try:
                conf = max(0.0, min(0.95, float(it.get("confidence", 0.5))))
            except Exception:
                conf = 0.5
            warns, bb = [], bbox
            raw = it.get("bbox")
            if isinstance(raw, list) and len(raw) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in raw) and raw[2] - raw[0] > 8 and raw[3] - raw[1] > 8:
                bb = (cx0 + raw[0] / 1000 * cw, cy0 + raw[1] / 1000 * ch, cx0 + raw[2] / 1000 * cw, cy0 + raw[3] / 1000 * ch)
            elif len(items) > 1:
                warns.append("Model gave no valid sub-region box; using the whole analysed region.")
            if not content:
                conf = min(conf, 0.45); warns.append("Low confidence — review source. Content withheld rather than guessed.")
            ch_ = it.get("chart") if isinstance(it.get("chart"), dict) else {}
            data = ch_.get("data") if isinstance(ch_.get("data"), list) and ch_.get("data") else None
            b = make_block("", t, 0, bb, conf, "model (self-reported)", VIS_EXT[t], content, latex=it.get("latex") if t == "equation" else None,
                           elements=it.get("elements") or None, relationships=[str(r) for r in it.get("relationships") or []] if t == "diagram" else None,
                           chart_type=ch_.get("chart_type") if t == "chart" else None, title=ch_.get("title") if t == "chart" else None,
                           data=data if t == "chart" else None, rows=rows, axes=ch_.get("axes") if t == "chart" else None, legend=ch_.get("legend") if t == "chart" else None)
            if t == "chart" and not data:
                b["warnings"].append("Chart values not extracted — trend described only; no values invented.")
            if t == "table" and rows:
                tb = tables.from_rows(rows, "vision", conf)
                b["cells"] = tb.to_dict()["cells"]; b["table_method"] = "vision"; b["header_rows"] = 1
            b["warnings"] += warns; blocks.append(b)
        if blocks:
            return blocks
        err = err or "vision model returned no regions"
    why = f"Vision extraction failed ({err}) — region preserved, flagged." if err else "Visual analysis unavailable: no VISION_API_KEY configured — region preserved, flagged."
    oc, ot = (0, "")
    if state.get("ocr_ok") and state["region_ocr_left"] > 0 and tag != "ink":   # ink regions: page already OCR'd
        state["region_ocr_left"] -= 1
        try:
            oc, ot = ocr.quick_conf(crop_img)
        except Exception:
            oc, ot = 0, ""
    b = make_block("", "figure" if hint == "figure" else "unknown", 0, bbox, 0.3, "heuristic (estimated)", "fallback", None,
                   ocr_text=ot if oc >= 0.6 and len(ot) >= 3 else None, preserved_image=True)
    b["warnings"] += [why] + (["Legible printed text found by OCR (see ocr_text); not treated as the region's meaning."] if b.get("ocr_text") else [])
    return [b]


# ---------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------
def process(path, filename, doc_id, out_dir, prog, save_images=True, include_furniture=None):
    timings = prog.timings if hasattr(prog, "timings") else {}
    t_all = time.perf_counter()
    state = dict(vision_left=_env_int("MAX_VISION_REGIONS", 12), vision_used=0, region_ocr_left=_env_int("MAX_REGION_OCR", 6))
    max_pages = _env_int("MAX_PAGES", 1000); warnings = []
    ocr_ok = bool(ocr.available()); state["ocr_ok"] = ocr_ok
    os.makedirs(out_dir, exist_ok=True)

    # DETECT ---------------------------------------------------------------------------------
    with _stage(prog, "DETECT", timings):
        kind, src = open_source(path, filename)
        n_pages = src.page_count if kind == "pdf" else 1
        if n_pages > max_pages:
            warnings.append(f"Document has {n_pages} pages; only the first {max_pages} were processed."); n_pages = max_pages
    prog.set("DETECT", "done", f"{kind.upper()} · {n_pages} page{'s' if n_pages > 1 else ''}")

    # ANALYZE / ROUTE / EXTRACT, in slices of pages -------------------------------------------
    # Pages are independent until ASSEMBLE, so a long document is handled CHUNK pages at a time: each slice is
    # analysed, routed and extracted, then its glyph data and page images are released. Memory stays bounded by the
    # slice, not the document (a 400-page report no longer holds 400 pages of glyphs and bitmaps at once).
    chunk = max(1, _env_int("PARSEFLOW_PAGE_CHUNK", 25))
    page_ctx, per_page = [], {}
    cnt = dict.fromkeys(("digital", "embedded_ocr", "ocr", "unreadable", "error"), 0)
    n_regions = n_blocks = 0
    of = f" of {n_pages}" if n_pages > 1 else ""
    for start in range(0, n_pages, chunk):
        sl = []
        with _stage(prog, "ANALYZE", timings):
            for i in range(start, min(n_pages, start + chunk)):
                prog.set("ANALYZE", "running", f"page {i + 1}{of}")
                sl.append(_analyze_one(i, kind, src, ocr_ok, warnings))
        n_regions += sum(len(c["items"]) for c in sl)
        with _stage(prog, "ROUTE", timings):
            for c in sl:
                cnt[c["route"]] = cnt.get(c["route"], 0) + 1
        prog.set("ROUTE", "running", " · ".join(f"{k} {v}" for k, v in cnt.items() if v))
        with _stage(prog, "EXTRACT", timings):
            for ctx in sl:
                prog.set("EXTRACT", "running", f"page {ctx['n']}{of}")
                per_page[ctx["n"]] = _extract_one(ctx, kind, src, out_dir, save_images, warnings, state)
                n_blocks += len(per_page[ctx["n"]])
                _release(ctx, kind)
        page_ctx.extend(sl)
    prog.set("ANALYZE", "done", f"{n_regions} regions identified on digital pages")
    prog.set("ROUTE", "done", " · ".join(f"{k} {v}" for k, v in cnt.items() if v))
    prog.set("EXTRACT", "done", f"{n_blocks} blocks")

    # ASSEMBLE (furniture classification across the document, ids, captions) ----------------------
    blocks = []
    with _stage(prog, "ASSEMBLE", timings):
        body_sizes = [ctx.get("body_size", 10) for ctx in page_ctx]
        body = statistics.median(body_sizes) if body_sizes else 10
        fpages = []
        for ctx in page_ctx:
            items = []
            for k, b in enumerate(per_page.get(ctx["n"], [])):
                if b["type"] in ("paragraph", "heading", "list", "caption", "code") and b.get("content"):
                    bb = tuple(v / Z for v in b["bbox"])
                    items.append(dict(id=(ctx["n"], k), bbox=bb, text=b.get("raw_text") or b["content"], size=b.get("font_size", body),
                                      rotated=b.get("rotated", False), kind="text"))
            fpages.append(dict(width=ctx["W"], height=ctx["H"], items=items))
        fur = headers_footers.classify(fpages, body)
        for ctx in page_ctx:
            n = ctx["n"]
            bs = per_page.get(n, [])
            for k, b in enumerate(bs):
                f = fur.get((n, k))
                if f:
                    b["furniture"] = f
                    if f["excluded"] or f["score"] >= headers_footers.REMOVE:
                        b["role"] = f["role"]
            ctr = {}
            for ro_i, b in enumerate(sorted(bs, key=lambda b: b["reading_order"]), 1):
                b["reading_order"] = ro_i
                pre = PREFIX.get(b.get("role")) if b.get("role") else None
                pre = pre or PREFIX.get(b["type"]) or ("ocr" if b["extractor"] == "ocr" else "txt")
                if b["type"] in ("heading", "paragraph", "list") and not b.get("role") and b["extractor"] not in ("ocr", "digital_text_parser"):
                    pre = "hand"
                ctr[pre] = ctr.get(pre, 0) + 1; b["block_id"] = f"{pre}_{n}_{ctr[pre]:02d}"; b["page"] = n
                blocks.append(b)
            targets = [b for b in bs if b["type"] in ("table", "chart", "diagram", "figure")]
            for c in [b for b in bs if b["type"] == "caption"]:
                near = sorted(targets, key=lambda t: min(abs(t["bbox"][1] - c["bbox"][3]), abs(c["bbox"][1] - t["bbox"][3])))
                if near and min(abs(near[0]["bbox"][1] - c["bbox"][3]), abs(c["bbox"][1] - near[0]["bbox"][3])) < 160:
                    c["relationships"].append("caption_of:" + near[0]["block_id"]); near[0]["relationships"].append("caption:" + c["block_id"])
        excluded = sum(1 for b in blocks if (b.get("furniture") or {}).get("excluded"))
    prog.set("ASSEMBLE", "done", f"{len(blocks)} blocks ordered · {excluded} running header/footer element(s) set aside")

    # VERIFY ---------------------------------------------------------------------------------
    with _stage(prog, "VERIFY", timings):
        by_page = {}
        for b in blocks:
            by_page.setdefault(b["page"], []).append(b)
        for ctx in page_ctx:
            n = ctx["n"]; a = ctx.get("a", {})
            if len(page_ctx) > 1:
                prog.set("VERIFY", "running", f"page {n}{of}")
            ptxt = ""
            if kind == "pdf" and ctx["route"] != "error":
                try:
                    ptxt = src[n - 1].get_text("text")
                except Exception:
                    ptxt = ""
            info = dict(width=ctx["W"] * Z, height=ctx["H"] * Z, norm_text=re.sub(r"[^a-z0-9]+", "", ptxt.lower()),
                        text_quality=a.get("text_quality", 1.0) if ctx["route"] == "digital" else textlayer.text_quality(a.get("hidden_text", ""))[0],
                        hidden_text_raw=a.get("hidden_text", "") if ctx["route"] == "ocr" else "",
                        hidden_text_norm=re.sub(r"[^a-z0-9]+", "", (a.get("hidden_text", "") if ctx["route"] == "ocr" else "").lower()), ocr_ok=ocr_ok)
            crop = _lazy_crop(ctx, kind, src, out_dir)
            for b in by_page.get(n, []):
                verifier.verify(b, info, crop)
        flagged = sum(1 for b in blocks if b["verification"] != "verified")
    prog.set("VERIFY", "done", f"{len(blocks) - flagged} verified · {flagged} flagged")

    # OUTPUT ---------------------------------------------------------------------------------
    with _stage(prog, "OUTPUT", timings):
        if not vision.available() and any(b["extractor"] == "fallback" for b in blocks):
            warnings.append("Vision API not configured: visual/handwritten regions were preserved and flagged, not interpreted.")
        if not blocks:
            warnings.append("No content could be extracted from this document.")
        pages = []
        for ctx in page_ctx:
            pg = dict(n=ctx["n"], kind=_kind_label(ctx), width=round(ctx["W"] * Z), height=round(ctx["H"] * Z),
                      flags=ctx.get("flags", {}), chars=ctx.get("a", {}).get("chars", 0), image_coverage=ctx.get("a", {}).get("image_coverage", 0),
                      route=ctx["route"], route_reason=ctx["reason"], text_quality=ctx.get("a", {}).get("text_quality"),
                      ocr=ctx.get("ocr_meta"), seconds=round(ctx["per_page_s"], 4))
            pages.append(pg)
        vis = ("chart", "figure", "diagram", "equation", "handwritten_text", "handwritten_code", "unknown")
        lanes = dict(local=sum(1 for b in blocks if b["extractor"] in ("digital_text_parser", "table_engine")), ocr=sum(1 for b in blocks if b["extractor"] == "ocr"),
                     vision=state["vision_used"], flagged=sum(b["extractor"] == "fallback" for b in blocks))
        timings["total"] = round(time.perf_counter() - t_all, 4)
        doc = dict(document_id=doc_id, filename=filename, format=kind, real=True, schema_version=2, pages=pages, blocks=blocks, warnings=warnings,
                   processing_status="complete",
                   router=[dict(block_id=b["block_id"], region_type=b["type"], extractor=b["extractor"]) for b in blocks],
                   stats=dict(pages=n_pages, blocks=len(blocks), tables=sum(b["type"] == "table" for b in blocks), visuals=sum(b["type"] in vis for b in blocks),
                              verified=len(blocks) - flagged, flagged=flagged, furniture_excluded=excluded, lanes=lanes),
                   engines=dict(ocr=ocr.available(), vision=vision.info(), math_recognizer=recognizers.info()),
                   timings=timings, config=dict(remove_furniture=headers_footers.enabled() if include_furniture is None else not include_furniture))
        doc["markdown"] = mdgen.render(doc, include_furniture=bool(include_furniture))
        with open(os.path.join(out_dir, "document.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=True)       # ASCII-escaped: valid even if a PDF carried invalid code points
    prog.set("OUTPUT", "done", "Markdown + JSON ready")
    if kind == "pdf":
        src.close()
    return doc


def _analyze_one(i, kind, src, ocr_ok, warnings):
    n = i + 1
    ctx = dict(n=n, pdf_page=None, img=None, items=[], route=None, reason="", per_page_s=0.0)
    t0 = time.perf_counter()
    try:
        if kind == "pdf":
            p = src[i]; ctx["pdf_page"] = p
            a = analyze_page(p)
            route, reason = route_page(a, ocr_ok)
            W, H = p.rect.width, p.rect.height
        else:
            a = dict(segs=[], chars=0, invisible=0, image_coverage=1.0, text_quality=0.0, quality_detail={}, img_zoom=0, hidden_text="",
                     W=src.width / Z, H=src.height / Z)
            route, reason = ("ocr" if ocr_ok else "unreadable"), "image file"
            W, H = a["W"], a["H"]
        ctx.update(a=a, route=route, reason=reason, W=W, H=H)
        if route in ("digital", "embedded_ocr"):
            _analyze_digital(ctx, a["segs"] if route == "digital" else _hidden_segments(ctx["pdf_page"]), warnings)
        ctx["per_page_s"] += time.perf_counter() - t0
    except Exception as e:
        ctx.update(route="error", reason=f"analysis failed ({type(e).__name__})", W=ctx.get("W", 612), H=ctx.get("H", 792), a=dict(chars=0, image_coverage=0, text_quality=0))
        warnings.append(f"Page {n}: analysis failed ({type(e).__name__}); page preserved.")
    return ctx


def _extract_one(ctx, kind, src, out_dir, save_images, warnings, state):
    t0 = time.perf_counter()
    n = ctx["n"]
    # save_images: True saves every page bitmap; "needed" saves only the bitmaps extraction renders anyway (scans,
    # figures) and leaves the rest to be rendered on demand by the viewer; False saves none.
    need_img = save_images is True or ctx["route"] in ("ocr",) or any(it["kind"] in ("figure",) for it in ctx["items"])
    ctx["had_img"] = need_img
    if need_img and ctx["route"] != "error":
        ctx["img"] = _render(ctx, kind, src)
        if save_images:
            ctx["img"].save(os.path.join(out_dir, f"page_{n}.png"), compress_level=1)
    try:
        if ctx["route"] == "ocr":
            _extract_ocr(ctx, warnings, state)
        elif ctx["route"] == "unreadable":
            warnings.append(f"Page {n}: OCR engine unavailable; scanned content cannot be read.")
            ctx["items"].append(dict(kind="figure", bbox=(0, 0, ctx["W"], ctx["H"]), hint="unknown"))
        out = _blocks_for_page(ctx, state)
    except Exception as ex:
        b = make_block("", "unknown", n, scale((0, 0, ctx["W"], ctx["H"])), 0.0, "heuristic (estimated)", "fallback", None, preserved_image=True)
        b["warnings"].append(f"Extractor error ({type(ex).__name__}) — page preserved, flagged.")
        out = [b]
    ctx["per_page_s"] += time.perf_counter() - t0
    return out


def _release(ctx, kind):
    """Drop what a finished page no longer needs: glyph segments, layout items, the page object and its bitmap.
    VERIFY and the viewer get the page image again on demand."""
    a = ctx.get("a")
    if isinstance(a, dict):
        a.pop("segs", None)
    ctx["items"] = []
    ctx["pdf_page"] = None
    if kind == "pdf":
        ctx["img"] = None
    for k in ("hrules", "text_segs", "vocab", "hybrid"):
        ctx.pop(k, None)


def _lazy_crop(ctx, kind, src, out_dir):
    """crop(bbox) for the verifier: the page bitmap is loaded (saved PNG, else a fresh render) only if a check needs it."""
    cache = {}

    def get():
        if "img" not in cache:
            im = ctx.get("img")
            if im is None and ctx.get("had_img"):    # only pages that had a bitmap during extraction get one here
                fp = os.path.join(out_dir, f"page_{ctx['n']}.png")
                try:
                    if os.path.exists(fp):
                        im = Image.open(fp); im.load()
                    elif kind == "pdf" and ctx["route"] != "error":
                        im = render_page(src, ctx["n"])
                except Exception:
                    im = None
            cache["img"] = im
        return cache["img"]

    def crop(bb):
        im = get()
        return im.crop(tuple(int(v) for v in bb)) if im is not None else None
    return crop


def render_page(src, n, zoom=None):
    """Page n (1-based) of an open PDF as an RGB image at the viewer scale Z (or `zoom`)."""
    z = Z if zoom is None else zoom
    pix = src[n - 1].get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _kind_label(ctx):
    r = ctx["route"]
    fl = ctx.get("flags", {})
    if r == "digital":
        return "mixed" if (fl.get("tables") or fl.get("figures")) else "digital"
    if r == "embedded_ocr":
        return "scanned (embedded OCR layer)"
    if r == "ocr":
        return "scanned" + (" + visual/handwriting" if fl.get("visual_regions") else "") if not ctx.get("a", {}).get("chars") else "digital (broken text layer) -> OCR"
    return r


def _render(ctx, kind, src):
    if kind != "pdf":
        return src
    pix = ctx["pdf_page"].get_pixmap(matrix=pymupdf.Matrix(Z, Z), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _hidden_segments(page):
    """Segments of an invisible (OCR) text layer: read with alpha ignored."""
    d = page.get_text("rawdict", flags=textlayer.RAW_FLAGS)
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                sp["alpha"] = 255
    # re-run the normal reader on the patched dict
    class _P:
        def get_text(self, *_a, **_k):
            return d
    segs, _ = textlayer.read_segments(_P(), ocr_layer=True)
    return segs


def _body_size(segs):
    sizes = []
    for s in segs:
        n = len(s.text)
        sizes += [round(s.size, 1)] * max(1, n)
    if not sizes:
        return 10.0
    return statistics.mode(sizes)


def _merge_drop_caps(segs, body):
    """A single oversized initial letter beside the first line of a paragraph is a drop cap: rejoin it."""
    out = list(segs)
    for d in [s for s in segs if s.horizontal and len(s.text) == 1 and s.text.isalpha() and s.size >= 2.0 * body]:
        right = [s for s in out if s is not d and s.horizontal and 0 <= s.bbox[0] - d.bbox[2] < 1.5 * body
                 and d.bbox[1] - 0.5 * body <= s.bbox[1] <= d.bbox[1] + 1.2 * body]
        if not right:
            continue
        host = min(right, key=lambda s: s.bbox[1])
        f = host.chars[0]
        cap = textlayer.Char(d.text, f.x0 - 0.6 * f.size, f.y0, f.x0, f.y1, f.x0 - 0.6 * f.size, f.oy, f.size, f.font, f.flags)
        host.chars.insert(0, cap)
        host._text = None
        host.bbox = union(host.bbox, d.bbox)
        out.remove(d)
    return out


def _analyze_digital(ctx, segs, warnings):
    p = ctx["pdf_page"]; W, H = ctx["W"], ctx["H"]
    body = _body_size(segs); ctx["body_size"] = body
    segs = _merge_drop_caps(segs, body)
    try:
        drawings = p.get_drawings()
    except Exception:
        drawings = []
    busy = len(drawings) > int(os.getenv("PARSEFLOW_BUSY_VECTOR_PATHS", "2500"))
    if busy:
        # maps, CAD, dense plots: thousands of vector paths. Table-grid search and path clustering would dominate
        # the runtime; the vector art is treated as one figure region instead (bounded cost).
        ctx.setdefault("notes", []).append(f"{len(drawings)} vector paths: busy-graphics mode (no grid-table search)")
        hr, vr = [], []
    else:
        hr, vr = tables.rules_from_drawings(drawings, W, H)
    ctx["hrules"] = hr
    # display equations first (their aligned pieces must not be mistaken for table cells)
    eqs = display_equations(segs, hr, body, None) if ctx["route"] == "digital" else []
    eq_ids = {id(s) for e in eqs for s in e["segs"]}
    noneq = [s for s in segs if id(s) not in eq_ids]
    # tables
    ruled = tables.detect_ruled(p, noneq, len(vr) >= 2, hr, vr) if ctx["route"] == "digital" else []
    aligned, used = tables.detect_aligned(noneq, hr, body, W, exclude=[t.bbox for t in ruled] + [e["bbox"] for e in eqs]) if not busy else ([], set())
    tbs = ruled + aligned
    hdr_used = set()
    for t in tbs:
        t, u = tables.extend_header(t, [s for s in noneq if not any(overlap_frac(s.bbox, o.bbox) > 0.6 for o in tbs)], hr, body)
        tables.merge_symbol_columns(t)
        hdr_used |= u
    tboxes = [t.bbox for t in tbs]
    in_table = lambda s: any(overlap_frac(s.bbox, tb) > 0.6 for tb in tboxes)
    rest = [s for s in noneq if not in_table(s) and id(s) not in hdr_used]
    # figures
    if busy:
        xs = [d["rect"] for d in drawings if d.get("rect") is not None]
        art = (min(r.x0 for r in xs), min(r.y0 for r in xs), max(r.x1 for r in xs), max(r.y1 for r in xs)) if xs else None
        figs = [tuple(art)] if art and (art[2] - art[0]) * (art[3] - art[1]) > 0.05 * W * H else []
    else:
        figs = figures(p, tboxes + [e["bbox"] for e in eqs], rest, drawings)
    figs = [(max(0, f[0]), max(0, f[1]), min(W, f[2]), min(H, f[3])) for f in figs if min(W, f[2]) > max(0, f[0]) and min(H, f[3]) > max(0, f[1])]
    fig_label = {}
    keep = []
    for s in rest:
        f = next((f for f in figs if overlap_frac(s.bbox, f) > 0.6), None)
        if f is not None and len(s.text) < 120:
            fig_label.setdefault(f, []).append(s)
        else:
            keep.append(s)
    rest = keep
    items = []
    for t in tbs:
        items.append(dict(kind="table", bbox=t.bbox, data=t))
    for e in eqs:
        items.append(dict(kind="equation", bbox=e["bbox"], data=e))
    for f in figs:
        items.append(dict(kind="figure", bbox=f, labels=fig_label.get(f, []), hint="figure"))
    ctx["items"] = items
    ctx["text_segs"] = rest
    ctx["flags"] = dict(text_layer=True, scanned=ctx["route"] != "digital", tables=len(tbs), figures=len(figs), equations=len(eqs))


def _extract_ocr(ctx, warnings, state):
    img = ctx["img"]; n = ctx["n"]
    a = ctx.get("a", {})
    # re-render at the scan's native resolution (bounded) for small print
    zoom = Z
    if ctx.get("pdf_page") is not None:
        native = a.get("img_zoom", 0) or Z
        # render at the scan's own resolution: never upsample a low-resolution scan (wasted OCR time), and cap
        # very high resolutions / very large pages to a bounded pixel budget
        zoom = min(max(native, 1.0), float(os.getenv("OCR_MAX_ZOOM", "4.17")))
        max_side = float(os.getenv("OCR_MAX_SIDE_PX", "4200"))
        zoom = min(zoom, max_side / max(ctx["W"], ctx["H"]))
        if abs(zoom - Z) < 0.05:
            oimg, zoom = img.convert("L"), Z
        else:
            pix = ctx["pdf_page"].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csGRAY)
            oimg = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    else:
        oimg = img
    res = ocr.ocr_image(oimg)
    k = zoom   # px per PDF point
    lines = ocr_lines(res["words"], k)
    lenient = False
    if not lines and res["metrics"].get("contrast", 0) >= 60 and res["metrics"].get("dark_ratio", 0) >= 0.005:
        # real ink but nothing passed the noise filter (typically handwriting): keep the plausible words, flagged
        good = [w for w in res["words"] if w["conf"] >= 0.45 and re.fullmatch(r"[^\W\d_]{3,}[.,;:]?|\d[\d.,/:-]*", w["text"])]
        groups = {}
        for w in good:
            groups.setdefault((w["block"], w["par"], w["line"]), []).append(w)
        lines = [OcrLine(ws, k) for ws in groups.values()]
        lenient = bool(lines)
    # hybrid page: visible native text (titles over a photo, stamps on a scan) is exact - keep it and drop
    # OCR lines that duplicate it
    native = [sg for sg in a.get("segs", []) if sg.text.strip()] if a.get("text_quality", 0) >= 0.6 else []
    if native:
        lines = [l for l in lines if not any(overlap_frac(l.bbox, sg.bbox) > 0.3 or overlap_frac(sg.bbox, l.bbox) > 0.3 for sg in native)]
        lines = lines + native
        ctx["hybrid"] = True
    ctx["ocr_meta"] = dict(conf=res["conf"], plausibility=res["plausibility"], steps=res["steps"], attempts=res["attempts"],
                           metrics={k2: v for k2, v in res["metrics"].items() if k2 != "size"}, render_zoom=round(zoom, 2), seconds=res["seconds"])
    body = statistics.median([l.size for l in lines]) if lines else 10
    ctx["body_size"] = body
    ctx["hrules"] = []
    aligned, used = tables.detect_aligned(lines, [], body, ctx["W"])
    tboxes = [t.bbox for t in aligned]
    rest = [l for l in lines if not any(overlap_frac(l.bbox, tb) > 0.6 for tb in tboxes)]
    ctx["text_segs"] = rest
    items = [dict(kind="table", bbox=t.bbox, data=t, ocr=True) for t in aligned]
    boxes_px = [tuple(v * Z for v in l.bbox) for l in lines]
    for r in detect_ink_regions(img, boxes_px):
        items.append(dict(kind="figure", bbox=tuple(v / Z for v in r), hint="unknown", ink=True))
    ctx["items"] = items
    ctx["flags"] = dict(text_layer=bool(a.get("chars")), scanned=True, tables=len(aligned), figures=0,
                        visual_regions=sum(1 for i in items if i.get("ink")))
    if lenient:
        ctx["lenient_ocr"] = True
        warnings.append(f"Page {n}: only isolated low-confidence OCR words were found (likely handwriting); they are kept, "
                        "flagged, and need review — a vision model is required to transcribe handwriting.")
    if res["conf"] < 0.6:
        warnings.append(f"Page {n}: low OCR confidence ({res['conf']:.2f}) after preprocessing ({', '.join(res['steps']) or 'none needed'}).")


def _blocks_for_page(ctx, state):
    n = ctx["n"]; body = ctx.get("body_size", 10); hr = ctx.get("hrules", [])
    segs = ctx.get("text_segs", [])
    route = ctx["route"]
    regions = [layout.Region(it["kind"], it["bbox"], it) for it in ctx["items"]]
    horiz = [s for s in segs if getattr(s, "horizontal", True)]
    rotated = [s for s in segs if not getattr(s, "horizontal", True)]
    ordered, multi = layout.reading_order(horiz + regions, body)
    if multi:
        ctx.setdefault("flags", {})["multi_column"] = True
    paras = layout.assemble_paragraphs(ordered, body)
    tparas = [p for p in paras if isinstance(p, layout.Para)]
    hsizes = layout.heading_size_levels(tparas, body)
    bsizes = layout.body_like_sizes(tparas)
    vocab = ctx.setdefault("vocab", layout.doc_vocab([s.text for s in segs]))
    blocks, ro = [], 0
    is_ocr = route == "ocr"
    for it in paras:
        ro += 1
        if isinstance(it, layout.Para):
            typ, lvl = layout.classify(it, body, hsizes, bsizes)
            is_code = typ == "code" or layout.is_code_lines([l.text for l in it.lines])
            vlines, raw_lines, has_math, _ = para_text(it, hr, use_math=not is_code and not any(isinstance(l, OcrLine) for l in it.lines))
            if vlines is None:
                vlines = raw_lines
            if getattr(it, "code", False) and not any(isinstance(l, OcrLine) for l in it.lines):
                vlines = layout.code_text(it).split("\n")
            if typ in ("list", "code"):
                text = "\n".join(vlines); tr = []
            else:
                text, tr = layout.join_lines(vlines, vocab)
            raw = "\n".join(raw_lines)
            if is_ocr and all(isinstance(l, OcrLine) for l in it.lines) and _looks_like_formula(text):
                eqb = _scanned_equation(ctx, it, text, n, ro, state)
                if eqb is not None:
                    blocks.append(eqb)
                    continue
            if is_ocr and any(isinstance(l, OcrLine) for l in it.lines):
                conf = statistics.mean(getattr(l, "conf", 0.99) for l in it.lines)
                b = make_block("", typ, n, scale(it.bbox), conf, "ocr", "ocr", text, level=lvl if typ == "heading" else None)
            else:
                b = make_block("", typ, n, scale(it.bbox), 0.99, "native text layer", "digital_text_parser", text,
                               level=lvl if typ == "heading" else None)
                if route == "embedded_ocr":
                    b["extractor"] = "ocr"; b["confidence_type"] = "embedded OCR layer"; b["confidence"] = 0.8
                    b["evidence_source"] = "embedded OCR text layer of the PDF (producer unknown)"
            b["raw_text"] = raw
            if tr:
                b["transforms"] = tr
            if has_math:
                b["transforms"] = b.get("transforms", []) + ["inline math converted to LaTeX \\( \\) from glyph geometry"]
                b["inline_math"] = True
            b["font_size"] = round(it.size, 2)
            b["reading_order"] = ro
            blocks.append(b)
            continue
        reg = it.payload
        if it.kind == "table":
            tb = reg["data"]
            rows = tb.rows()
            ext, ctype = ("ocr", "ocr") if reg.get("ocr") else ("table_engine", "heuristic (estimated)")
            b = make_block("", "table", n, scale(tb.bbox), tb.confidence, ctype, ext, f"Table with {tb.n_rows} rows", rows=rows,
                           cells=tb.to_dict()["cells"], table_method=tb.method, header_rows=tb.header_rows,
                           alignment=round(tb.confidence, 2), markdown=tables.serialize(tb))
            b["warnings"] += tb.warnings
            b["reading_order"] = ro
            blocks.append(b)
        elif it.kind == "equation":
            e = reg["data"]
            blocks += _equation_blocks(e, n, body, ro)
        else:
            bb = reg["bbox"]
            labels = reg.get("labels") or []
            img = ctx.get("img")
            if labels:
                lbl_text = " ".join(s.text for s in sorted(labels, key=lambda s: (round(s.bbox[1] / 4), s.bbox[0])))
                b = make_block("", "figure", n, scale(bb), 0.99, "native text layer", "digital_text_parser", lbl_text,
                               figure_labels=True)
                b["warnings"].append("Figure: only the printed labels inside it were extracted (text layer); the graphic itself is not interpreted.")
                b["reading_order"] = ro
                blocks.append(b)
            elif img is not None:
                crop = img.crop(tuple(int(v * Z) for v in bb))
                vb = vision_blocks(crop, scale(bb), reg.get("hint", "figure"), state, "ink" if reg.get("ink") else "figure")
                for b in vb:
                    b["reading_order"] = ro
                blocks += vb
    # rotated text (side-margin stamps, rotated labels) goes last; furniture classification decides its fate
    for s in rotated:
        ro += 1
        b = make_block("", "paragraph", n, scale(s.bbox), 0.99, "native text layer", "digital_text_parser", s.text, rotated=True)
        b["raw_text"] = s.text; b["font_size"] = round(s.size, 2); b["reading_order"] = ro
        blocks.append(b)
    return blocks


_MATHY = set("=+−-×÷^_∑∫∏√∞≤≥≠≈±∂∇∈∉⊂⊆∪∩→←⇒⇔()[]{}|/<>'") | set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")


def _looks_like_formula(t):
    vis = [c for c in t if not c.isspace()]
    if len(vis) < 3 or len(vis) > 160:
        return False
    words = re.findall(r"[A-Za-z]{4,}", t)
    sym = sum(1 for c in vis if c in _MATHY)
    return sym / len(vis) >= 0.2 and len(words) <= 2 and any(c in "=<>≤≥≈∑∫" for c in vis)


def _scanned_equation(ctx, para, text, n, ro, state):
    """Equation-looking OCR line on a scan: run the optional local recognizer on its crop only."""
    img = ctx.get("img")
    rinfo = recognizers.info()
    if rinfo["ready"] and img is not None and state.get("math_left", 20) > 0:
        state["math_left"] = state.get("math_left", 20) - 1
        bb = para.bbox
        pad = 4
        crop = img.crop((max(0, int(bb[0] * Z) - pad), max(0, int(bb[1] * Z) - pad), int(bb[2] * Z) + pad, int(bb[3] * Z) + pad))
        r = recognizers.recognize(crop)
        if r:
            lx, conf = r
            b = make_block("", "equation", n, scale(bb), conf, "model (self-reported)", "math_model", text, latex=lx.strip(), math_display=True,
                           raw_text=text, transforms=[f"LaTeX from local recognizer '{rinfo['name']}' on the region crop; OCR text kept as raw_text"])
            b["reading_order"] = ro
            return b
    b = make_block("", "paragraph", n, scale(para.bbox), statistics.mean(getattr(l, "conf", 0.5) for l in para.lines), "ocr", "ocr", text, raw_text=text)
    b["warnings"].append("Possible equation in a scan: OCR text kept, not converted to LaTeX" +
                         ("" if rinfo["ready"] else " (no local formula recognizer configured: PARSEFLOW_MATH_RECOGNIZER)") + ".")
    b["possible_equation"] = True
    b["reading_order"] = ro
    return b


def _equation_blocks(e, n, body, ro):
    chars, rules = e["chars"], e["rules"]
    size = statistics.median([c.size for c in chars if c.c.strip()] or [body])
    rows = _split_rows(chars, rules, size)
    out = []
    for k, (rc, rr) in enumerate(rows):
        rc2, num = mathtex.split_equation_number(rc, e["bbox"][2])
        lx, conf, notes = mathtex.build_latex(rc2, rr)
        raw = textlayer._chars_to_text(sorted(rc, key=lambda c: (round(c.oy / 3), c.x0)))
        box = (min(c.x0 for c in rc), min(c.y0 for c in rc), max(c.x1 for c in rc), max(c.y1 for c in rc))
        reconstructed = any(x.startswith(("fraction", "script", "radical", "big")) for x in notes)
        if lx and conf >= 0.45:
            b = make_block("", "equation", n, scale(box), conf, "heuristic (estimated)", "digital_text_parser", raw, latex=lx, math_display=True,
                           equation_number=num, reconstructed=reconstructed, raw_text=raw, transforms=notes)
        else:
            b = make_block("", "paragraph", n, scale(box), 0.6, "native text layer", "digital_text_parser", raw, raw_text=raw)
            b["warnings"].append("Equation could not be converted to LaTeX confidently; original glyph text kept.")
        b["reading_order"] = ro + k * 0.01
        out.append(b)
    return out
