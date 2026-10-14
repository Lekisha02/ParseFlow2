# ParseFlow v2 — changes

Baseline: the uploaded `PARSERFLOWS-main` (commit `c729770` in this history). All v1 API routes, response fields,
frontend features, upload workflow and downloads are kept; JSON changes are additive.

## Modified files
| File | Change |
|---|---|
| `backend/processing/pipeline.py` | Rewritten around the same 7 stages: per-page routing by measured text-layer quality (digital / embedded OCR / OCR / hybrid / sparse), per-stage timings, render-on-demand, native-resolution OCR rendering, display-equation, table, figure, drop-cap handling, document-level furniture pass, scanned-equation recognizer hook, busy-vector-page mode, invalid-code-point safety, backend Markdown in the output. |
| `backend/processing/ocr.py` | Adaptive preprocessing from measured metrics (contrast, background, speckle, skew, text size), bounded fallback (conservative image when risky steps were used), optional OSD, image-hash cache, configurable language/timeout. v1 helpers kept. |
| `backend/processing/verifier.py` | Evidence rules: independence, document-encoded basis, OCR/model never verified alone, ruled vs borderless tables, `evidence_detail`. |
| `backend/processing/schema.py` | Documented contract, `Z`, `union`, `scale` helpers. |
| `backend/app.py` | Markdown endpoint, upload type check before storing, safe filenames, cleanup on failure, job expiry, timings in status, richer health. Existing routes/fields unchanged. |
| `frontend/index.html` | Markdown tab renders backend Markdown (+ include-headers toggle; browser/demo mode keeps `md()`), Inspector shows evidence basis, furniture reasons, transforms, original text, HTML tables with spans; timings in Overview; dashed overlays for excluded headers/footers; Header/Footer filter. |
| `README.md`, `.env.example`, `pyproject.toml`, `requirements-dev.txt` (new), `.gitignore` (new) | Documentation, configuration, packaging (`parseflow` console script), dev/benchmark dependencies. |

## New files
| File | Purpose |
|---|---|
| `backend/processing/textlayer.py` | Glyph-level text layer reader: segments with font/size/baseline, adaptive spacing, ligatures, accents, stray-accent repair, zero-width/surrogate cleanup, text-quality score. |
| `backend/processing/layout.py` | Recursive XY-cut reading order with gutter preference, sidebars, paragraphs, headings (size rarity, bold, ALL-CAPS), lists, captions, code, evidence-based dehyphenation, quote normalisation. |
| `backend/processing/headers_footers.py` | Document-level running header/footer/page-number classifier with scores and reasons; configurable removal. |
| `backend/processing/tables.py` | Cell model; ruled-grid tables (clipped `find_tables`, spans); aligned/booktabs tables; header extension (rotated labels, underline spans); currency columns; Markdown/HTML serialisation. |
| `backend/processing/mathtex.py` | Math detection (token level) and LaTeX reconstruction from glyph geometry. |
| `backend/processing/recognizers.py` | Optional local formula recognizers (pix2tex, rapid_latex_ocr, command plugin); never downloads weights. |
| `backend/processing/markdown.py` | Canonical deterministic Markdown serializer. |
| `backend/processing/runner.py` | Isolated per-document worker processes with timeouts (CLI + benchmark). |
| `backend/cli.py`, `parseflow`, `backend/__init__.py` | One-command batch CLI. |
| `bench/run_olmocr_bench.py`, `bench/legacy_runner.py`, `bench/compare.py`, `bench/baselines/*` | Official-scorer harness, v1 runner (same conditions), regression gate, recorded baselines. |
| `tests/*` | 67 pytest tests + official-scorer regression test on the bundled sample. |
| `.github/workflows/ci.yml` | CI: tests + sample regression gate (written, not executed here). |

`backend/processing/pdf_parser.py` and `vision.py` are unchanged (pdf_parser is no longer used by the pipeline).


## v2.1 — Ask, readable math, long documents
| File | Change |
|---|---|
| `backend/processing/ask.py` (new) | Extractive Ask engine (BM25 + section context, phrase/definition bonuses, table-cell lookup, page overviews, code-step excerpts, term/description rows, split-passage joining); typed answer parts with LaTeX kept for rendering. |
| `backend/app.py` | `GET /api/ask/{id}`; page images and thumbnails rendered on demand (`/api/page`, new `/api/thumb`); limits raised to 1000 pages / 300 MB. |
| `backend/processing/pipeline.py` | Pages processed in slices with per-page progress; glyph data and bitmaps released after each slice; verifier crops loaded lazily (same results); `save_images="needed"`. |
| `backend/processing/headers_footers.py` | Repetition measured at the same height and capped at 4 pages for long documents; page numbers that swap sides on even/odd pages; heading-sized one-off page titles kept in long documents. |
| `backend/processing/layout.py`, `tables.py`, `mathtex.py` | Numbered pseudo-code / enumerated listings are not tables; code paragraphs keep indentation and skip inline math; Unicode super/subscripts in LaTeX; crash on multi-code-point glyphs fixed. |
| `frontend/index.html`, `frontend/vendor/katex/` | Ask tab uses the API and renders math (KaTeX, vendored), code, lists and tables, with evidence snippets and suggested questions from the document; math rendered in Inspector/Blocks/Tables; lazy thumbnails, paged long lists, server-side upload limit. The post-upload screens look the same. |
| `tests/test_ask_and_large_docs.py` (new) | Ask answers, API, on-demand images, slice equivalence, page progress, long-document furniture. |

## v2.2 — new landing page, accounts and saved documents
| File | Change |
|---|---|
| `backend/accounts.py` (new) | SQLite users / sessions / documents; scrypt passwords; hashed session tokens; login rate limit; library stats. |
| `backend/app.py` | Auth, account and library endpoints; uploads by a logged-in user are saved; owner-only access to saved documents; guest expiry skips saved documents. |
| `frontend/index.html` | New landing page (logo, Try Live Demo, Log in / Sign up, auto-playing 7-slide feature tour replacing "Present 90s demo", upload card with guest / save options, moving feature ticker, features, stages, drop zone, footer); sign-in dialog; logged-in workspace (sidebar, library with filters, search, preview panel, rename/delete/download, settings, privacy). Processing screen and viewer unchanged; *New document* returns to the landing page or the library. Logo gradient now renders in every view. |
| `tests/test_accounts.py` (new) | Sign-up/login/logout, password + session storage, rate limit, privacy of saved documents, guest behaviour, rename/delete/expiry, settings, account deletion, interrupted jobs. |
