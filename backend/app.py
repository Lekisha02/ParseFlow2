import os, re, sys, json, time, uuid, shutil, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Request, Body
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
from processing import ask as askmod
import accounts

VERSION = "2.1.0"
DATA = Path(os.getenv("PARSEFLOW_DATA_DIR", str(ROOT/"data"))); DATA.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "300"))*1024*1024
JOB_TTL = float(os.getenv("PARSEFLOW_JOB_TTL_HOURS", "24"))*3600
ALLOWED = {"pdf", "png", "jpg", "jpeg"}
JOBS = {}; POOL = ThreadPoolExecutor(int(os.getenv("PARSEFLOW_WORKERS", "2")))
app = FastAPI(title="ParseFlow", version=VERSION)
STORE = accounts.Store(DATA/"parseflow.db")
SECURE_COOKIE = os.getenv("PARSEFLOW_SECURE_COOKIE", "0").lower() in ("1", "true", "yes")

def err(code, title, msg, status=400): return JSONResponse(dict(error=dict(code=code, title=title, message=msg)), status_code=status)
def _safe_id(doc_id): return bool(re.fullmatch(r"[a-f0-9]{12}", doc_id or ""))

def _cleanup_old():
    """Delete guest documents older than PARSEFLOW_JOB_TTL_HOURS. Documents saved to an account are kept until
    their owner deletes them."""
    if JOB_TTL <= 0: return
    now = time.time(); owned = STORE.owned_ids()
    for d in DATA.iterdir():
        try:
            if d.is_dir() and d.name not in owned and now - d.stat().st_mtime > JOB_TTL:
                shutil.rmtree(d, True); JOBS.pop(d.name, None)
        except OSError: pass

@app.get("/api/health")
def health():
    try:
        import pymupdf; pm = pymupdf.VersionBind
    except Exception: pm = None
    return dict(ok=True, version=VERSION, ocr=ocr.available(), vision=vision.info(), math_recognizer=recognizers.info(), pymupdf=pm,
                max_upload_mb=MAX_BYTES//(1024*1024), max_pages=int(os.getenv("MAX_PAGES", "1000")))

# ------------------------------------------------------------------------------------------- accounts
def _user(request):
    return STORE.user_for(request.cookies.get(accounts.COOKIE))

def _guard(request, doc_id):
    """Guest documents are reachable by their unguessable id; a saved document only by its owner."""
    if not _safe_id(doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    owner = STORE.owner(doc_id)
    if owner is not None:
        u = _user(request)
        if not u or u["id"] != owner: return err("PF-404", "Not Found", "Document not available.", 404)
    return None

def _session_response(user, token):
    r = JSONResponse(dict(user=STORE.public(user)))
    r.set_cookie(accounts.COOKIE, token, max_age=int(accounts.SESSION_DAYS * 86400), httponly=True, samesite="lax", secure=SECURE_COOKIE, path="/")
    return r

def _auth_err(e): return err(e.code, "Account", e.message, e.status)

def _need_user(request):
    u = _user(request)
    return (u, None) if u else (None, err("PF-401", "Not Logged In", "Log in to use your library.", 401))

def _remove_doc_files(doc_id):
    if _safe_id(doc_id): shutil.rmtree(DATA/doc_id, True); JOBS.pop(doc_id, None)

@app.post("/api/auth/signup")
def signup(payload: dict = Body(...)):
    try: u = STORE.signup(payload.get("name"), payload.get("email"), payload.get("password"))
    except accounts.AuthError as e: return _auth_err(e)
    return _session_response(u, STORE.new_session(u["id"]))

@app.post("/api/auth/login")
def login(payload: dict = Body(...)):
    try: u = STORE.login(payload.get("email"), payload.get("password"))
    except accounts.AuthError as e: return _auth_err(e)
    return _session_response(u, STORE.new_session(u["id"]))

@app.post("/api/auth/logout")
def logout(request: Request):
    STORE.end_session(request.cookies.get(accounts.COOKIE))
    r = JSONResponse(dict(ok=True)); r.delete_cookie(accounts.COOKIE, path="/"); return r

@app.get("/api/auth/me")
def me(request: Request):
    return dict(user=STORE.public(_user(request)))

@app.patch("/api/account")
def account_update(request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    try: return dict(user=STORE.public(STORE.update_name(u["id"], payload.get("name"))))
    except accounts.AuthError as x: return _auth_err(x)

@app.post("/api/account/password")
def account_password(request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    try: STORE.change_password(u["id"], payload.get("current"), payload.get("new"), keep_token=request.cookies.get(accounts.COOKIE))
    except accounts.AuthError as x: return _auth_err(x)
    return dict(ok=True)

@app.post("/api/account/delete")
def account_delete(request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    try: ids = STORE.delete_user(u["id"], payload.get("password"))
    except accounts.AuthError as x: return _auth_err(x)
    for i in ids: _remove_doc_files(i)
    r = JSONResponse(dict(ok=True, deleted_documents=len(ids))); r.delete_cookie(accounts.COOKIE, path="/"); return r

@app.get("/api/my/documents")
def my_documents(request: Request):
    u, e = _need_user(request)
    if e: return e
    docs = STORE.documents(u["id"])
    for d in docs:
        j = JOBS.get(d["id"])
        if d["status"] == "processing" and j:
            run = [s for s in j["prog"].stages if s["state"] == "running"]
            d["progress"] = (run[-1]["name"].title() + (" · " + run[-1]["detail"] if run[-1]["detail"] else "")) if run else None
    return dict(documents=docs, usage=dict(documents=len(docs), pages=sum(d["pages"] or 0 for d in docs), bytes=sum(d["size"] or 0 for d in docs)))

@app.patch("/api/my/documents/{doc_id}")
def my_document_rename(doc_id: str, request: Request, payload: dict = Body(...)):
    u, e = _need_user(request)
    if e: return e
    if not STORE.document(u["id"], doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    try: STORE.rename(u["id"], doc_id, payload.get("filename"))
    except accounts.AuthError as x: return _auth_err(x)
    return dict(document=STORE.document(u["id"], doc_id))

@app.delete("/api/my/documents/{doc_id}")
def my_document_delete(doc_id: str, request: Request):
    u, e = _need_user(request)
    if e: return e
    if not STORE.document(u["id"], doc_id): return err("PF-404", "Not Found", "Document not available.", 404)
    if JOBS.get(doc_id) and JOBS[doc_id]["prog"].status == "running":
        return err("PF-409", "Still Processing", "Wait until processing finishes, then delete the document.", 409)
    STORE.delete_document(u["id"], doc_id); _remove_doc_files(doc_id)
    return dict(ok=True)

# ------------------------------------------------------------------------------------------- documents
@app.post("/api/process")
async def process(request: Request, file: UploadFile = File(...)):
    _cleanup_old()
    user = _user(request)
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
    if user: STORE.add_document(doc_id, user["id"], name, ext.upper(), size, pages)
    def run():
        try:
            doc = pipeline.process(str(src), name, doc_id, str(d), prog, save_images="needed"); prog.status = "complete"
            if user: STORE.finish_document(doc_id, "complete", accounts.doc_stats(doc))
        except pipeline.PFError as e:
            prog.status, prog.error = "failed", e.dict()
            if user: STORE.finish_document(doc_id, "failed", error=e.message)
        except Exception as e:
            prog.status, prog.error = "failed", dict(code="PF-500", title="Processing Error", message=f"Unexpected failure ({type(e).__name__}); the server is still running.")
            if user: STORE.finish_document(doc_id, "failed", error=prog.error["message"])
        finally: JOBS[doc_id]["doc"] = None   # the result lives in document.json; keep memory bounded
    POOL.submit(run)
    return dict(document_id=doc_id, filename=name, size=size, type=ext.upper(), pages=pages, saved=bool(user))

@app.get("/api/status/{doc_id}")
def status(doc_id: str, request: Request):
    if (g := _guard(request, doc_id)): return g
    j = JOBS.get(doc_id)
    if not j: return err("PF-404", "Unknown Document", "No such processing job.", 404)
    p = j["prog"]
    return dict(status=p.status, stages=p.stages, error=p.error, timings=getattr(p, "timings", {}))

@app.get("/api/document/{doc_id}")
def document(doc_id: str, request: Request):
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    return FileResponse(p, media_type="application/json")

@app.get("/api/document/{doc_id}/markdown")
def document_markdown(doc_id: str, request: Request, include_furniture: bool = Query(False), annotate: bool = Query(True)):
    """Canonical backend Markdown. include_furniture=1 restores running headers/footers; annotate=0 drops HTML-comment warnings."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    doc = json.loads(p.read_text())
    return PlainTextResponse(mdgen.render(doc, include_furniture=include_furniture, annotate=annotate), media_type="text/markdown; charset=utf-8")

@app.get("/api/ask/{doc_id}")
def ask_document(doc_id: str, request: Request, q: str = Query("", max_length=500)):
    """Extractive answer to a question about a processed document: parts copied from blocks, with evidence."""
    if (g := _guard(request, doc_id)): return g
    p = DATA/doc_id/"document.json"
    if not _safe_id(doc_id) or not p.exists(): return err("PF-404", "Not Ready", "Document not available.", 404)
    key = (doc_id, p.stat().st_mtime)
    doc = None if key in askmod._CACHE else json.loads(p.read_text())
    return askmod.answer(doc, q, key=key)

RENDER_LOCK = threading.Lock()
THUMB_W = 180

def _source(doc_id):
    d = DATA/doc_id
    src = next(iter(sorted(d.glob("source_*"))), None) if d.is_dir() else None
    return d, src

def _page_image(doc_id, n, thumb=False):
    """Page bitmaps are rendered on first request and cached next to the document (a 400-page upload is not
    rasterised up front). Full pages are PNG at the viewer scale; thumbnails are small JPEGs."""
    d, src = _source(doc_id)
    out = d/(f"thumb_{n}.jpg" if thumb else f"page_{n}.png")
    if out.exists(): return out
    full = d/f"page_{n}.png"
    if src is None or not (d/"document.json").exists(): return None
    with RENDER_LOCK:
        if out.exists(): return out
        from PIL import Image
        name = src.name.split("_", 2)[-1]
        if thumb and full.exists():
            im = Image.open(full); im.load()
        else:
            kind, s = pipeline.open_source(str(src), name)
            try:
                if kind == "pdf":
                    if not 1 <= n <= s.page_count: return None
                    w = s[n - 1].rect.width or 612
                    im = pipeline.render_page(s, n, zoom=(THUMB_W * 2 / w) if thumb else None)
                else:
                    if n != 1: return None
                    im = s
            finally:
                if kind == "pdf": s.close()
        tmp = out.with_suffix(".tmp")
        if thumb:
            im = im.convert("RGB"); im.thumbnail((THUMB_W * 2, THUMB_W * 4)); im.save(tmp, "JPEG", quality=72)
        else:
            im.save(tmp, "PNG", compress_level=1)
        os.replace(tmp, out)
    return out

@app.get("/api/page/{doc_id}/{n}.png")
def page_png(doc_id: str, n: int, request: Request):
    if (g := _guard(request, doc_id)): return g
    p = _page_image(doc_id, n)
    if p is None: return err("PF-404", "Not Found", "Page image not found.", 404)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

@app.get("/api/thumb/{doc_id}/{n}.jpg")
def page_thumb(doc_id: str, n: int, request: Request):
    if (g := _guard(request, doc_id)): return g
    p = _page_image(doc_id, n, thumb=True)
    if p is None: return err("PF-404", "Not Found", "Page image not found.", 404)
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

app.mount("/", StaticFiles(directory=str(ROOT/"frontend"), html=True), name="ui")
