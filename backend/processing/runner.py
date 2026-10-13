"""Batch execution shared by the CLI and the benchmark harness.

Each document runs in its own worker process so that a malformed PDF, a native-library crash or a timeout
affects only that file. Workers are started with the 'spawn' method (no inherited state, deterministic),
at most `workers` run at once, and documents are streamed one by one (never all loaded in memory).
"""
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback


def convert(path, include_furniture=False, save_images=False, out_dir=None):
    """Convert one document in-process. Returns the Digital Twin dict (with doc['markdown'])."""
    from . import pipeline
    name = os.path.basename(path)
    prog = pipeline.Progress()
    tmp = None
    if out_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="parseflow_")
        out_dir = tmp.name
    try:
        doc = pipeline.process(path, name, "cli", out_dir, prog, save_images=save_images, include_furniture=include_furniture)
        return doc
    finally:
        if tmp is not None:
            tmp.cleanup()


def _worker(path, opts, q):
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")       # one Tesseract thread per worker process
    try:
        from . import markdown as md, pipeline  # noqa: F401  (imports are not part of the measured conversion time)
    except BaseException as e:
        q.put(dict(ok=False, seconds=0.0, error=f"import failed: {type(e).__name__}: {e}"))
        return
    t0 = time.perf_counter()
    try:
        doc = convert(path, include_furniture=opts.get("include_furniture", False))
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # Linux: KiB -> MiB
        except Exception:
            rss = None
        res = dict(ok=True, seconds=time.perf_counter() - t0, pages=len(doc["pages"]), max_rss_mb=rss,
                   page_seconds=[p.get("seconds") for p in doc["pages"]], timings=doc.get("timings"),
                   routes=[p.get("route") for p in doc["pages"]], warnings=doc.get("warnings", []))
        if opts.get("per_page_markdown"):
            res["page_markdown"] = {p["n"]: md.render(dict(doc, blocks=[b for b in doc["blocks"] if b["page"] == p["n"]], warnings=[], filename=None),
                                                      include_furniture=opts.get("include_furniture", False), annotate=opts.get("annotate", True))
                                    for p in doc["pages"]}
        res["markdown"] = md.render(doc, include_furniture=opts.get("include_furniture", False), annotate=opts.get("annotate", True))
        if opts.get("want_json"):
            res["doc"] = doc
        q.put(res)
    except BaseException as e:   # noqa: report every failure, including SystemExit from native code
        code = getattr(e, "code", None)
        q.put(dict(ok=False, seconds=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}", code=code if isinstance(code, str) else None,
                   trace=traceback.format_exc(limit=4)))


def run_batch(paths, opts=None, workers=None, timeout=300, on_result=None):
    """Run documents in isolated processes. Yields (path, result) as each finishes; results are dicts with ok/error."""
    opts = opts or {}
    workers = max(1, workers or os.cpu_count() or 1)
    ctx = mp.get_context("spawn")
    pending = list(paths)
    running = {}       # path -> (proc, queue, start)
    results = []
    while pending or running:
        while pending and len(running) < workers:
            p = pending.pop(0)
            q = ctx.Queue()
            pr = ctx.Process(target=_worker, args=(p, opts, q), daemon=True)
            pr.start()
            running[p] = (pr, q, time.perf_counter())
        done = []
        for p, (pr, q, t0) in running.items():
            res = None
            try:
                res = q.get_nowait()
            except queue.Empty:
                pass
            if res is not None:
                pr.join(5)
                done.append((p, res))
            elif not pr.is_alive():
                done.append((p, dict(ok=False, seconds=time.perf_counter() - t0, error=f"worker exited with code {pr.exitcode} (crash)")))
            elif time.perf_counter() - t0 > timeout:
                pr.terminate(); pr.join(5)
                if pr.is_alive():
                    pr.kill()
                done.append((p, dict(ok=False, seconds=time.perf_counter() - t0, error=f"timeout after {timeout}s", timeout=True)))
        for p, res in done:
            running.pop(p, None)
            results.append((p, res))
            if on_result:
                on_result(p, res)
        if not done:
            time.sleep(0.02)
    return results
