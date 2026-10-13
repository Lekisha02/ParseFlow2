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
