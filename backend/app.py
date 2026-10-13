import os, re, sys, json, time, uuid, shutil, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
try:
    for line in (ROOT/".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if v.strip(): os.environ.setdefault(k.strip(), v.strip())
except FileNotFoundError: pass
from processing import pipeline, ocr, vision, recognizers
from processing import markdown as mdgen

VERSION = "2.0.0"
DATA = Path(os.getenv("PARSEFLOW_DATA_DIR", str(ROOT/"data"))); DATA.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50"))*1024*1024
JOB_TTL = float(os.getenv("PARSEFLOW_JOB_TTL_HOURS", "24"))*3600
ALLOWED = {"pdf", "png", "jpg", "jpeg"}
JOBS = {}; POOL = ThreadPoolExecutor(int(os.getenv("PARSEFLOW_WORKERS", "2")))
app = FastAPI(title="ParseFlow", version=VERSION)

def err(code, title, msg, status=400): return JSONResponse(dict(error=dict(code=code, title=title, message=msg)), status_code=status)
def _safe_id(doc_id): return bool(re.fullmatch(r"[a-f0-9]{12}", doc_id or ""))

def _cleanup_old():
    """Delete processed documents older than PARSEFLOW_JOB_TTL_HOURS (uploads are temporary)."""
    if JOB_TTL <= 0: return
    now = time.time()
    for d in DATA.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > JOB_TTL:
                shutil.rmtree(d, True); JOBS.pop(d.name, None)
        except OSError: pass

@app.get("/api/health")
def health():
    try:
        import pymupdf; pm = pymupdf.VersionBind
    except Exception: pm = None
    return dict(ok=True, version=VERSION, ocr=ocr.available(), vision=vision.info(), math_recognizer=recognizers.info(), pymupdf=pm,
                max_upload_mb=MAX_BYTES//(1024*1024), max_pages=int(os.getenv("MAX_PAGES", "60")))

@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    _cleanup_old()
    name = os.path.basename(file.filename or "upload")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext not in ALLOWED:
        return err("PF-204", "Unsupported Document", "ParseFlow could not process this file type. Supported: PDF, PNG, JPG/JPEG.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[-120:] or "upload"
    doc_id = uuid.uuid4().hex[:12]; d = DATA/doc_id; d.mkdir(); src = d/("source_"+doc_id+"_"+safe)
    size = 0
    try:
        with open(src, "wb") as f:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    f.close(); shutil.rmtree(d, True)
                    return err("PF-206", "File Too Large", f"Maximum size is {MAX_BYTES//(1024*1024)} MB.", 413)
                f.write(chunk)
    except Exception:
        shutil.rmtree(d, True); return err("PF-500", "Upload Failed", "The upload could not be stored.", 500)
    try: pages = pipeline.page_count(str(src), name)
    except pipeline.PFError as e: shutil.rmtree(d, True); return JSONResponse(dict(error=e.dict()), status_code=400)
    except Exception: shutil.rmtree(d, True); return err("PF-201", "Corrupt Document", "The file could not be read.")
    prog = pipeline.Progress(); JOBS[doc_id] = dict(prog=prog, doc=None, started=time.time())
    def run():
        try: JOBS[doc_id]["doc"] = pipeline.process(str(src), name, doc_id, str(d), prog); prog.status = "complete"
        except pipeline.PFError as e: prog.status, prog.error = "failed", e.dict()
        except Exception as e: prog.status, prog.error = "failed", dict(code="PF-500", title="Processing Error", message=f"Unexpected failure ({type(e).__name__}); the server is still running.")
        finally: JOBS[doc_id]["doc"] = None   # the result lives in document.json; keep memory bounded
    POOL.submit(run)
    return dict(document_id=doc_id, filename=name, size=size, type=ext.upper(), pages=pages)

@app.get("/api/status/{doc_id}")
def status(doc_id: str):
    j = JOBS.get(doc_id)
    if not j: return err("PF-404", "Unknown Document", "No such processing job.", 404)
    p = j["prog"]
    return dict(status=p.status, stages=p.stages, error=p.error, timings=getattr(p, "timings", {}))

@app.get("/api/document/{doc_id}")
def document(doc_id: str):
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    return FileResponse(p, media_type="application/json")

@app.get("/api/document/{doc_id}/markdown")
def document_markdown(doc_id: str, include_furniture: bool = Query(False), annotate: bool = Query(True)):
    """Canonical backend Markdown. include_furniture=1 restores running headers/footers; annotate=0 drops HTML-comment warnings."""
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    doc = json.loads(p.read_text())
    return PlainTextResponse(mdgen.render(doc, include_furniture=include_furniture, annotate=annotate), media_type="text/markdown; charset=utf-8")

@app.get("/api/page/{doc_id}/{n}.png")
def page_png(doc_id: str, n: int):
    p = DATA/doc_id/f"page_{n}.png"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Found", "Page image not found.", 404)
    return FileResponse(p, media_type="image/png")

app.mount("/", StaticFiles(directory=str(ROOT/"frontend"), html=True), name="ui")
