# ParseFlow

**ParseFlow** is a document processing system developed by **Team DEADLOCK** for the **DQCL ParseAnything** problem statement (DataQuest 3.0).

It reads PDFs and images that mix digital text, tables, scans, handwriting, equations, diagrams and charts, uses the right extractor for each region, and returns a structured **Digital Twin** (JSON) plus **canonical Markdown**, keeping the evidence behind every result: page, bounding box, extractor, confidence, verification status and warnings.

> **Version 2** (this revision) rebuilt the extraction core for accuracy and speed while keeping the architecture, API and frontend: column-aware layout, running header/footer detection, table structure with merged cells, LaTeX from the PDF text layer, adaptive OCR preprocessing, backend Markdown, a stricter evidence-based verifier, a one-command CLI, a pytest suite and an official olmOCR-Bench harness. See [CHANGES.md](CHANGES.md) and [Benchmark results](#benchmark-results).

---

## Problem Statement

Most extraction systems work on clean digital text, but real documents mix digital text, tables, scanned pages, handwriting, handwritten code, equations, diagrams, charts and images. One technique for everything loses or invents information, and AI models can produce content that is not in the document. ParseFlow processes each kind of content with a suitable method and records where every result came from.

---

## Our Approach

```text
Document → Detect → Analyze → Route → Extract → Assemble → Verify → Structured Output (JSON + Markdown)
```

| Stage | What happens (v2) |
|---|---|
| **Detect** | File type, page count, encryption, size limits. Per page (no rendering): visible text amount, hidden OCR text layer, image coverage, **text-layer quality** (broken font maps and garbage OCR layers are detected). |
| **Analyze** | Character-level reading of the text layer (font, size, baseline, bbox for every glyph); display equations; ruled and borderless tables; figures; drop caps; split accents. Scanned pages: adaptive OCR. |
| **Route** | Per page and per region, cheapest suitable method first (table below). |
| **Extract** | Paragraphs, headings, lists, code, captions, tables (cell model), equations (LaTeX), figures (vision or flagged). |
| **Assemble** | Column-aware reading order, **document-level header/footer/page-number classification**, block ids, caption links. |
| **Verify** | Evidence-based status per block (independence rules below). |
| **Output** | Digital Twin JSON (backward compatible, additive fields) + canonical Markdown generated in the backend. |

### Routing

| Content | Method |
|---|---|
| Digital text (trustworthy text layer) | Native text layer (PyMuPDF), column-aware layout |
| Text layer present but broken (unmapped glyphs, implausible words) | Adaptive OCR; the text layer is kept as evidence |
| Scanned page with an embedded OCR layer | Embedded layer when plausible, otherwise Tesseract |
| Scanned page | Adaptive preprocessing + Tesseract |
| Photo/graphic page with a few real text lines | **Hybrid**: native lines kept, OCR for the rest |
| Ruled tables | Vector ruling grid → cells with rowspan/colspan |
| Borderless / booktabs tables | Column alignment + horizontal rules for header blocks |
| Equations (digital) | LaTeX rebuilt from glyph geometry (no model) |
| Equations (scanned) | Optional local recognizer; otherwise OCR text kept and flagged |
| Handwriting, diagrams, charts, figures | Vision model when configured; otherwise preserved and flagged |

Vision models are optional and never required. Everything else runs **offline without API keys**.

---

## What is extracted and how

### Digital text and reading order
Glyphs are read with their font, size and baseline. Lines are split at wide gaps; spaces are synthesised adaptively when a PDF positions words without space glyphs; ligatures are expanded; accents drawn as separate glyphs are composed (`a` + `´` → `á`); drop caps are re-attached.
Reading order uses a recursive XY-cut that prefers real text gutters, treats full-width elements (titles, running headers) as separators, and handles sidebars next to text columns. Hyphenated line breaks are rejoined only with evidence (the joined word occurs in the document, or the prefix is not a compound-forming prefix such as `self-`, `well-`, `non-`).

### Running headers, footers and page numbers
A document-level classifier scores each edge-of-page line with independent signals and keeps the reasons: position (top/bottom band, extreme edge, rotated side-margin stamps), isolation from the body, **repetition across pages** (digits normalised), page-number patterns (`12`, `- 12 -`, `xii`, `Page 3 of 20`), publication boilerplate (URLs, DOI, ©, "Downloaded from", journal/volume/issue, revision/part numbers, dates), small font, rows of short items spread across the line, items sharing a line with a page number, and content that ends far from the paper edge. Protective signals keep one-off titles, footnotes, captions and section headings. A structural signal (isolation, repetition, page number, margin, edge row) is required before anything is removed; a fixed margin alone never removes text.
Removed items stay in the JSON (`role`, `furniture.score`, `furniture.reasons`, `furniture.excluded`). Removal is configurable (`PARSEFLOW_REMOVE_FURNITURE=0`, CLI `--include-headers`, API `?include_furniture=1`, frontend toggle).

### Tables
* **Ruled tables**: PyMuPDF `find_tables()` runs only inside regions where vector rules form a grid; cell geometry is mapped onto row/column edges so merged cells get real `rowspan`/`colspan`.
* **Borderless/booktabs tables**: consecutive rows that split into aligned fragments; columns from x-coverage, plus whitespace "rivers" (e.g. a `#` column next to a text column); rejection of multi-column prose (long fragments, high row coverage, misaligned cell edges) and leader dots; category rows inside tables are allowed.
* **Header blocks** above a table (e.g. *Year Ended* over date columns, rotated 45° column labels, group labels such as *ReACT*) are attached by column alignment; a rule under a header cell defines its span.
* Currency-only columns are folded into their values (`$178`), nothing is dropped.
* Output: Markdown for rectangular tables with one header row, **HTML with `<th>`, `rowspan`, `colspan`** otherwise. Pipes are escaped, in-cell line breaks normalised, empty cells kept empty (`filled: false`). Values are never invented.

### Equations
On digital PDFs the LaTeX is rebuilt from the text layer: Unicode→LaTeX symbol table (Greek, operators, relations, arrows, ℝ/ℕ, math alphanumerics), TeX font families (`\mathbb`, `\mathcal`, `\mathfrak`, `\mathbf`), sub/superscripts from baseline offsets and sizes, fractions from vector rules, big operators with limits, radicals with overbars, `\operatorname` for upright function names, multi-line displays split into separate equations, equation numbers separated. Display math is emitted as `\[ … \]`, inline math as `\( … \)` (token-level detection: italic words, footnote markers and Greek-language words are not wrapped). Glyphs without a Unicode mapping lower the confidence and are reported instead of guessed. Reconstructed structure is marked `review`, never `verified`. Original glyph text is kept in `raw_text`.

### OCR for scans (`processing/ocr.py`)
1. **Measure**: contrast (ink vs background), background unevenness, speckle (full resolution), skew (projection profile, applied only when the estimate is clearly better than 0°), character height.
2. **Apply only what the measurements call for**: illumination normalisation, contrast stretch, median despeckle, deskew, upscaling of small print, Sauvola binarisation.
3. **Bounded fallback**: if the result is weak, one alternative representation (the conservative "minimal" image if risky steps were used); never more than 2 attempts by default; OSD rotation only on request.
4. Pages are rendered at the scan's native resolution (bounded), never upsampled needlessly. Results are cached by image hash. Noise lines from textures/photos are filtered; handwriting-only pages keep isolated words, flagged.

### Verification (stricter than v1)
* `verified` requires **independent evidence**, or content read directly from the document's encoded text layer (no recognition step) with passing quality checks — that basis is recorded as *document-encoded*, not as independent.
* Outputs derived from the same extraction never count as independent (table cells are not "verified" by the text layer they were read from; two Tesseract passes are not independent).
* OCR alone is `review`; it becomes `verified` when an independent recogniser (the PDF's embedded OCR layer) agrees. Model output is `review` unless independently confirmed (e.g. table cells found in the text layer, OCR token overlap).
* Ruled tables are `verified` (grid geometry is independent of the text); borderless tables are `review` (structure inferred).
* Every block carries `evidence` (short string, v1-compatible) and `evidence_detail` (`basis`, `independent`, `signals`, `confidence_method`), plus `raw_text` and `transforms` whenever normalisation/reconstruction was applied. Statuses: `verified`, `review` (uncertain), `failed`.

---

### Ask Your Document (`processing/ask.py`)
Answers are **copied from extracted blocks, never generated** (no model, no API key). The question is matched with BM25 over blocks plus the heading each block sits under, phrase and definition bonuses (`... is called ...`), and stemming that handles `debugging`/`debug`. Four strategies: *page* ("What does page 4 explain?"), *table cell* (row label × column header), *section* (the question names a heading → the heading with its content) and *passage* (the best sentences, a code step such as `// Right Heavy Case` with the lines under it, a term with the description beside it, or a passage the layout split across blocks joined back together). Math stays LaTeX with `\( \)` delimiters and the viewer renders it with a vendored KaTeX (`frontend/vendor/katex`, works offline); code keeps its line numbers and indentation; tables are rendered as tables. Every answer lists its evidence blocks with **View Source**.

### Long documents
A 400-page PDF is processed in slices of `PARSEFLOW_PAGE_CHUNK` pages: each slice is analysed, routed and extracted, then its glyph data and page bitmaps are released; the processing screen shows `page N of M` for each stage. Digital pages are not rasterised during processing — the viewer requests page images and thumbnails on demand (lazy-loaded), and the long lists in the viewer (Blocks, Tables, Visuals, JSON) render in pages with *Show more*. Measured on a 400-page, 13 MB mixed test PDF (398 digital pages, 6,644 blocks, 2 vCPU container): peak memory **362 MB** (was 3.7 GB with every page bitmap held), API upload → viewer open **71 s**, document JSON 9 MB. Scanned pages are OCR'd one by one and take far longer than digital pages.

## Digital Twin (JSON)

All v1 fields are unchanged. Example block:

```json
{
  "block_id": "eq_1_01", "type": "equation", "content": "λg = ∑ s∈S ζ ψ(gs) n", "page": 1,
  "bbox": [412, 288, 794, 364], "reading_order": 2, "confidence": 0.8,
  "confidence_type": "heuristic (estimated)", "extractor": "digital_text_parser", "verification": "review",
  "relationships": [], "warnings": [],
  "latex": "\\lambda_{g} = \\sum_{s \\in S} \\zeta_{n}^{\\psi ( gs )}", "math_display": true, "reconstructed": true,
  "raw_text": "λg = ∑ s∈S ζ ψ(gs) n", "transforms": ["big operator limits reconstructed", "script placement from baseline offsets"],
  "evidence": "text-layer quality 1.00; LaTeX structure reconstructed from glyph geometry (inferred, not encoded)",
  "evidence_detail": {"basis": "document-encoded text layer (no recognition step)", "independent": false, "signals": ["…"], "confidence_method": "text-layer quality checks"}
}
```

Additive document fields: `markdown`, `schema_version: 2`, `timings` (per stage, seconds), `pages[].route`, `pages[].route_reason`, `pages[].text_quality`, `pages[].ocr` (preprocessing steps, attempts, metrics), `pages[].seconds`, `stats.furniture_excluded`, `engines.math_recognizer`, `config`. Table blocks add `cells` (row, col, rowspan, colspan, text, bbox, header, filled), `header_rows`, `table_method`, `markdown`.

---

## Installation

Requirements: Python ≥ 3.10, the Tesseract binary (for scans), Linux or macOS (Windows works via `run.bat`).

```bash
# system OCR engine
sudo apt install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                # macOS

git clone https://github.com/ADII6769/PARSERFLOWS.git
cd PARSERFLOWS
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt               # app + CLI
pip install -r requirements-dev.txt           # tests and benchmark tooling (optional)
```

Or install as a package (provides the `parseflow` command): `pip install -e ".[server]"`.

### Web app
```bash
bash run.sh          # Linux/macOS  (run.bat on Windows)  → http://localhost:8000
```

**Screens.** The landing page has the logo top-left, *Try Live Demo* and *Log in / Sign up* top-right, an auto-playing feature tour (7 slides, pauses on hover, arrows/dots to step; it replaces the old "Present 90s demo" button), the upload card, and — further down — features, the seven stages and a drop zone. After an upload, the processing screen and the document viewer are exactly the v2 screens.

**Guests and accounts.** Without an account, a file is processed and opened in the viewer, and deleted from the server after `PARSEFLOW_JOB_TTL_HOURS`. With an account (email + password, stored on this server only — no third-party login), every upload is also saved to *My documents*: a workspace with a left sidebar (upload, My documents, Needs review, recent files, Settings, Privacy & data, storage used, log out), filters, search by name, a preview panel (first page, summary, open in viewer, Markdown/JSON download, rename, delete) and account settings (name, password, delete account). *New document* in the viewer returns guests to the landing page and logged-in users to their library. Passwords are salted scrypt hashes; sessions are random tokens in an HttpOnly, SameSite=Lax cookie (only their SHA-256 is stored); saved documents are readable only by their owner. Set `PARSEFLOW_SECURE_COOKIE=1` when the app is served over HTTPS.

### One-command CLI (offline, no API keys)
```bash
./parseflow input_pdfs/ --output-dir output_markdown/
# or, after `pip install -e .`:
parseflow input_pdfs/ --output-dir output_markdown/
```
| Option | Meaning |
|---|---|
| `-r/--recursive` | search folders recursively (sub-folders are mirrored, so names never collide) |
| `-w/--workers N` | parallel worker processes (default min(4, CPUs)) |
| `-t/--timeout S` | per-document timeout (the worker is killed, the batch continues) |
| `--json` | also write the Digital Twin `<name>.json` |
| `--report r.json` | machine-readable summary of every file |
| `--log-json` | one JSON line per finished document on stdout |
| `--include-headers` | keep running headers/footers/page numbers |
| `--plain` | no HTML-comment warnings in the Markdown |
| `--skip-existing`, `--images` | skip done files; also convert PNG/JPG |

Each document runs in its own process (a malformed PDF cannot take the batch down); successful outputs are written atomically and kept when other files fail; the exit code is `1` if any file failed, `0` otherwise, `2` for usage errors.

### Python
```python
import sys; sys.path.insert(0, "backend")
from processing.runner import convert
doc = convert("paper.pdf")          # Digital Twin dict
print(doc["markdown"])
```

---

## API (FastAPI)

| Endpoint | Description |
|---|---|
| `GET /api/health` | backend status, OCR engine, vision config, math recognizer, PyMuPDF version, limits |
| `POST /api/process` | upload (PDF/PNG/JPG; type checked before storing; size limit `MAX_UPLOAD_MB`); returns `document_id, filename, size, type, pages` |
| `GET /api/status/{id}` | `status`, `stages`, `error`, **`timings`** |
| `GET /api/document/{id}` | Digital Twin JSON (now includes `markdown`) |
| `GET /api/document/{id}/markdown` | **new**: canonical Markdown; `?include_furniture=1` restores headers/footers, `?annotate=0` drops HTML-comment warnings |
| `GET /api/page/{id}/{n}.png` | page image for the viewer; rendered on first request and cached (digital pages are not rasterised during processing) |
| `GET /api/thumb/{id}/{n}.jpg` | **new**: small JPEG thumbnail for the page strip, rendered on demand and cached |
| `POST /api/auth/signup` · `POST /api/auth/login` · `POST /api/auth/logout` · `GET /api/auth/me` | **new**: accounts; JSON bodies `{name, email, password}` / `{email, password}`; session cookie `pf_session` |
| `GET /api/my/documents` · `PATCH /api/my/documents/{id}` · `DELETE /api/my/documents/{id}` | **new**: the logged-in user's library (status, live progress, per-document stats, usage), rename (`{filename}`), delete |
| `PATCH /api/account` · `POST /api/account/password` · `POST /api/account/delete` | **new**: change name, change password (signs out other sessions), delete account and all its documents |
| `GET /api/ask/{id}?q=...` | **new**: Ask Your Document — extractive answer (`parts` typed as text with inline LaTeX, math, code, list, table, heading), `strategy` (page · table cell · section · passage), `evidence` with block ids, `confidence`, matched/missing terms; `found=false` with the closest passages when nothing matches |

Uploads that fail validation are deleted immediately; guest documents expire after `PARSEFLOW_JOB_TTL_HOURS` (default 24); documents uploaded while logged in are kept until their owner deletes them. Document routes (`/api/document`, `/markdown`, `/status`, `/page`, `/thumb`, `/ask`) answer 404 for a saved document unless the request carries its owner's session; guest documents stay reachable by their unguessable id, as before. The frontend renders the backend Markdown (with an *include running headers/footers* toggle) and keeps its own `md()` only for demo/browser mode.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VISION_PROVIDER`, `VISION_API_KEY`, `VISION_MODEL`, `MAX_VISION_REGIONS` | gemini, –, provider default, 12 | optional vision model (never required) |
| `MAX_PAGES` | 1000 | page limit per document (was 60) |
| `MAX_UPLOAD_MB` | 300 | API upload limit (was 50) |
| `PARSEFLOW_PAGE_CHUNK` | 25 | pages analysed + extracted per slice; memory is bounded by the slice, not the document |
| `OCR_LANG` | eng | Tesseract language(s), e.g. `eng+deu` |
| `OCR_TIMEOUT` | 60 | seconds per Tesseract call |
| `OCR_MAX_ATTEMPTS` | 2 | bounded OCR fallback attempts (3 enables OSD rotation) |
| `OCR_MAX_ZOOM`, `OCR_MAX_SIDE_PX` | 4.17, 4200 | render-resolution bounds for scans |
| `MAX_REGION_OCR` | 6 | OCR of picture regions on digital pages (per document) |
| `PARSEFLOW_REMOVE_FURNITURE` | 1 | exclude confidently detected headers/footers from Markdown |
| `PARSEFLOW_MATH_RECOGNIZER` | none | `none` · `pix2tex` · `rapid_latex_ocr` · `command` (scanned equations) |
| `PARSEFLOW_MATH_MODEL_DIR`, `PARSEFLOW_MATH_CMD` | – | local weights folder / external recognizer command |
| `PARSEFLOW_DATA_DIR`, `PARSEFLOW_JOB_TTL_HOURS`, `PARSEFLOW_WORKERS` | data/, 24, 2 | API storage (documents + `parseflow.db` with accounts), guest expiry, concurrency |
| `PARSEFLOW_SESSION_DAYS`, `PARSEFLOW_SECURE_COOKIE` | 30, 0 | login session lifetime; set the cookie's `Secure` flag (use with HTTPS) |

### Formula recognition for scanned equations (optional)
Digital PDFs need no model. For equations in **scans**, a recognizer can be plugged in; it runs only on detected equation regions and **never downloads weights at runtime** (missing weights are reported via `/api/health` and block warnings):

* `pix2tex` (LaTeX-OCR, MIT): `pip install "pix2tex[cli]"` (needs PyTorch; CPU works, roughly seconds per equation). Put `weights.pth` and `image_resizer.pth` from the project's GitHub release into `PARSEFLOW_MATH_MODEL_DIR`.
* `rapid_latex_ocr` (Apache-2.0, ONNX Runtime only): `pip install rapid_latex_ocr`; put `encoder.onnx`, `decoder.onnx`, `image_resizer.onnx`, `tokenizer.json` into `PARSEFLOW_MATH_MODEL_DIR`.
* `command`: any local executable that takes an image path and prints LaTeX (`PARSEFLOW_MATH_CMD`).

These adapters could not be exercised here (no PyPI/model access in the build environment); only the `command` plugin and the "weights missing" path are covered by tests. Evaluate a model on old-scans math before relying on it.

---

## Tests

```bash
pytest -q                                   # 67 unit/integration tests on generated fixtures (~30 s; OCR tests need tesseract)
OLMOCR_PATH=/path/to/olmocr pytest -m bench # regression on the official sample with the official test classes
```
Covered: native text, scanned OCR (skew, noise, blur, low contrast, uneven light, small text; preprocessing never worse than plain OCR), two-column and sidebar reading order, header/footer classification (repeated headers, page numbers, first-page title, footnotes, section headings, documents without headers, configurability), equation boundaries/LaTeX/symbols/fallback, ruled tables with merged cells, borderless tables, prose rejection, Markdown escaping, hyphenation, Unicode/accents, empty/corrupt/encrypted/multi-page/image inputs, OCR timeout fallback, verifier independence/provenance, JSON and Markdown API responses, upload validation and cleanup, CLI batch/timeout, determinism. Failures name the PDF, page, test id and the scorer's explanation.

---

## Benchmark (official olmOCR-Bench scorer)

`bench/run_olmocr_bench.py` converts every PDF, writes `<candidate>/<pdf>_pg{page}_repeat1.md`, then runs the **unmodified official CLI** and records scores, latency (median/p90/p95/max, throughput), errors, timeouts, hardware and configuration as JSON + Markdown. Failed or timed-out documents get an empty Markdown file, so their tests fail; nothing is skipped.

```bash
git clone https://github.com/allenai/olmocr.git          # scorer (we used commit f7cfe4c)
pip install fuzzysearch rapidfuzz tqdm beautifulsoup4 pypdf playwright && playwright install chromium

# full benchmark: 1,403 PDFs from Hugging Face
huggingface-cli download --repo-type dataset --resume-download allenai/olmOCR-bench --local-dir ./olmOCR-bench
python bench/run_olmocr_bench.py --bench-dir olmOCR-bench/bench_data --olmocr olmocr --results bench/results/full.json

# fixed regression sample: N PDFs per category, deterministic by hash (manifest saved)
python bench/run_olmocr_bench.py --bench-dir olmOCR-bench/bench_data --sample-per-category 25 --seed 0 --olmocr olmocr

# original v1 implementation under identical conditions
python bench/run_olmocr_bench.py --bench-dir … --impl legacy --legacy-backend /path/to/v1/backend --legacy-md plain

# regression gate (CI): fail if a protected category drops by more than 1 point
python bench/compare.py --baseline bench/baselines/sample_v2.json --current bench/results/parseflow.json --tolerance 1.0 --protected dataset.jsonl,type:math,type:table,type:absent
```
The scorer command that is run is exactly: `python -m olmocr.bench.benchmark --dir <bench-dir> --candidate parseflow`.

### Benchmark results

**Only the official scorer's bundled sample (21 PDFs, 100 tests) could be run in the build environment**: Hugging Face (and therefore the 1,403-PDF dataset) was not reachable. The sample is also the set this version was developed against, so these are **in-sample numbers, not a held-out or full-benchmark score**. The full run has to be done with the commands above.

Hardware: Intel Xeon @ 2.80 GHz, 2 logical cores, 7.8 GB RAM, Linux; PyMuPDF 1.28.2, Tesseract 5.3.4; 2 workers; vision off; no math model; Python `rapidfuzz` implementation in the scorer. Latency = time inside each worker around the conversion call (imports and process spawn excluded, cold caches, one fresh process per document), identical method for v1 and v2.

| Official sample, `dataset.jsonl` (99 tests) | v1 original | v2 |
|---|---|---|
| **Main score** | **34.3 %** (34/99) | **83.8 %** (83/99) |
| absent (headers/footers) | 33.3 – 37.5 | 100.0 |
| math | 0.0 | 100.0 |
| table | 50.0 | 100.0 |
| order | 46.2 | 84.6 |
| present | 30.0 – 33.3 | 53.3 |
| baseline / blank-page checks | 100 / 0–100 | 100 / 100 |
| median latency (s/page) | 0.50 | **0.17** |
| p90 / p95 / max (s/page) | 1.6–1.8 / 3.3 / 11.8–12.2 | 2.2 / 2.5 / 6.8 |
| errors / timeouts | 0 / 0 | 0 / 0 (peak worker RSS 421 MB) |

(v1 ranges cover the two ways of turning v1 output into Markdown: the frontend `md()` with provenance comments, and plain text. The scorer's printed "overall" averages the three JSONL groups of the sample — 44.8/78.1 vs 94.6 — and is not meaningful on this sample.) Robustness check (not scored): the CLI converted all 37 PDFs in olmocr's `tests/gnarly_pdfs` (498 pages: rotated, image-only, broken text layers, maps, forms) with no failures, median 0.21 s/page, in 82 s on the same 2 cores. The remaining sample failures are handwriting (needs a vision model), errors already present in a scan's OCR, logo text inside an image, and one ambiguous card-layout order.

**Is the score above 80 %?** On the official *sample* the main test file reaches 83.8 %, but that is in-sample and not the benchmark. A reproducible **full olmOCR-Bench score above 80 % has not been demonstrated**. For context, the olmocr README reports Marker 76.1, MinerU 75.2, DeepSeek-OCR 75.7 and Mistral OCR 72.0; the results above 80 (PaddleOCR-VL 80.0, olmOCR v0.4 82.4, Infinity-Parser 82.5, Chandra 83.1) come from neural vision-language models. The full set contains old-scan, handwriting and scanned-math material where an offline Tesseract pipeline is fundamentally limited. Run the full benchmark to get the real number.

---

## Project structure

```text
PARSERFLOWS/
├── backend/
│   ├── app.py                FastAPI app (v1 API + markdown, ask, thumbnails, accounts, library)
│   ├── accounts.py           local accounts, sessions, saved documents (SQLite, stdlib only)
│   ├── cli.py                batch CLI
│   └── processing/
│       ├── pipeline.py       stages, routing, assembly, timings
│       ├── textlayer.py      glyph-level text layer, spacing, accents, quality
│       ├── layout.py         reading order, paragraphs, headings, dehyphenation
│       ├── headers_footers.py running header/footer/page-number classifier
│       ├── tables.py         table detection + cell model + Markdown/HTML
│       ├── mathtex.py        math detection + LaTeX reconstruction
│       ├── recognizers.py    optional local formula recognizers
│       ├── ocr.py            adaptive preprocessing + bounded OCR
│       ├── verifier.py       evidence-based verification
│       ├── markdown.py       canonical Markdown serializer
│       ├── runner.py         isolated batch execution (CLI + benchmark)
│       ├── vision.py         optional Gemini/Claude vision (unchanged)
│       ├── schema.py         block schema helpers
│       └── pdf_parser.py     v1 helpers (kept for compatibility, unused by v2 pipeline)
├── frontend/index.html
├── bench/                    official-scorer harness, v1 runner, compare, baselines
├── tests/                    pytest suite
├── parseflow                 CLI launcher
├── requirements.txt, requirements-dev.txt, pyproject.toml, run.sh, run.bat, .env.example
└── .github/workflows/ci.yml
```

---

## Deployment notes

* ParseFlow needs the **Tesseract binary** for scans and runs background jobs in a thread pool with page images on local disk. Serverless platforms (e.g. Vercel functions) typically lack the binary, cap execution time and do not keep local files between invocations; **this was not tested on any serverless platform**, so use a normal server or container there.
* Licences: PyMuPDF/MuPDF is **AGPL-3.0** (or commercial licence from Artifex); Tesseract is Apache-2.0. Check this before distributing ParseFlow as a hosted service.

## Current limitations

* Handwriting, hand-drawn diagrams and charts need a configured vision model; offline they are preserved and flagged.
* Scanned math is OCR text + flag unless a local recognizer is configured; inline math detection on digital PDFs is heuristic.
* Unmapped glyphs (e.g. big delimiters in some TeX fonts) are dropped from LaTeX and flagged.
* Borderless tables with irregular layouts or tables spanning pages may be split or merged imperfectly (marked `review`).
* Card/grid layouts can be read column-wise where row-wise was intended.
* OCR quality on degraded historic scans is bounded by Tesseract.
* Ask is extractive keyword/heading/table matching: it does not paraphrase, summarise across sections or answer questions whose words are not in the document.
* The CLI's default per-document timeout (`-t 300`) can be too short for long scanned documents; raise it for those.

## Team

Team DEADLOCK · Project: ParseFlow · Problem Statement: DQCL ParseAnything · Event: DataQuest 3.0 · Repository: https://github.com/ADII6769/PARSERFLOWS

## In Short

Don't treat every part of a document the same. Understand the content first, use the right extractor, and keep the evidence with the result.
