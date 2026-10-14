"""Local accounts, sessions and saved documents for the ParseFlow web app.

Everything stays on the ParseFlow server: one SQLite file (PARSEFLOW_DATA_DIR/parseflow.db), standard library only.

* Passwords: scrypt (n=2^14, r=8, p=1) with a per-user random salt; never stored or logged in clear.
* Sessions: a random 256-bit token in an HttpOnly, SameSite=Lax cookie; only its SHA-256 is stored, so a copy of
  the database does not contain usable sessions. Sessions expire after PARSEFLOW_SESSION_DAYS (default 30).
* Documents: a row per upload made while logged in. Guest uploads have no row and expire after
  PARSEFLOW_JOB_TTL_HOURS as before; saved documents stay until their owner deletes them (or the account).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time

EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,24}$")
MIN_PASSWORD = 8
SESSION_DAYS = float(os.getenv("PARSEFLOW_SESSION_DAYS", "30"))
COOKIE = "pf_session"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL, pw TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, created REAL NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, filename TEXT NOT NULL,
  type TEXT, size INTEGER, pages INTEGER, created REAL NOT NULL, status TEXT NOT NULL, stats TEXT, error TEXT);
CREATE INDEX IF NOT EXISTS documents_user ON documents(user_id, created);
"""


class AuthError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def hash_password(pw, salt=None):
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$%s$%s" % (salt.hex(), h.hex())


def check_password(pw, stored):
    try:
        kind, salt, h = stored.split("$")
        return kind == "scrypt" and hmac.compare_digest(hash_password(pw, bytes.fromhex(salt)), stored)
    except Exception:
        return False


def _tok(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.Lock()
        self.fails = {}                    # email -> [timestamps of failed logins]
        with self._db() as c:
            c.executescript(SCHEMA)
            # documents that were processing when the server stopped will never finish
            c.execute("UPDATE documents SET status='failed', error='Processing was interrupted (server restart). Upload the file again.' WHERE status='processing'")

    def _db(self):
        c = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _q(self, sql, args=(), one=False, write=False):
        with self.lock:
            c = self._db()
            try:
                cur = c.execute(sql, args)
                rows = cur.fetchall()
                if write:
                    c.commit()
                return (rows[0] if rows else None) if one else rows
            finally:
                c.close()

    # ---------------------------------------------------------------------------------------- users
    @staticmethod
    def public(u):
        return dict(id=u["id"], email=u["email"], name=u["name"], created=u["created"]) if u else None

    def signup(self, name, email, password):
        name, email = (name or "").strip()[:80], (email or "").strip().lower()
        if not name:
            raise AuthError("PF-401", "Please enter your name.")
        if not EMAIL.match(email):
            raise AuthError("PF-401", "Please enter a valid email address.")
        if len(password or "") < MIN_PASSWORD:
            raise AuthError("PF-401", f"Use a password of at least {MIN_PASSWORD} characters.")
        try:
            self._q("INSERT INTO users(email,name,pw,created) VALUES(?,?,?,?)", (email, name, hash_password(password), time.time()), write=True)
        except sqlite3.IntegrityError:
            raise AuthError("PF-409", "An account with this email already exists. Log in instead.", 409)
        return self._q("SELECT * FROM users WHERE email=?", (email,), one=True)

    def login(self, email, password):
        email = (email or "").strip().lower()
        now = time.time()
        recent = [t for t in self.fails.get(email, []) if now - t < 600]
        if len(recent) >= 8:
            raise AuthError("PF-429", "Too many failed attempts. Wait a few minutes and try again.", 429)
        u = self._q("SELECT * FROM users WHERE email=?", (email,), one=True)
        if not u or not check_password(password or "", u["pw"]):
            self.fails[email] = recent + [now]
            raise AuthError("PF-401", "Email or password is incorrect.", 401)
        self.fails.pop(email, None)
        return u

    def update_name(self, uid, name):
        name = (name or "").strip()[:80]
        if not name:
            raise AuthError("PF-401", "Please enter your name.")
        self._q("UPDATE users SET name=? WHERE id=?", (name, uid), write=True)
        return self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)

    def change_password(self, uid, current, new, keep_token=None):
        u = self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)
        if not u or not check_password(current or "", u["pw"]):
            raise AuthError("PF-401", "Your current password is incorrect.", 401)
        if len(new or "") < MIN_PASSWORD:
            raise AuthError("PF-401", f"Use a password of at least {MIN_PASSWORD} characters.")
        self._q("UPDATE users SET pw=? WHERE id=?", (hash_password(new), uid), write=True)
        # sign out every other session
        self._q("DELETE FROM sessions WHERE user_id=? AND token<>?", (uid, _tok(keep_token) if keep_token else ""), write=True)

    def delete_user(self, uid, password):
        u = self._q("SELECT * FROM users WHERE id=?", (uid,), one=True)
        if not u or not check_password(password or "", u["pw"]):
            raise AuthError("PF-401", "Your password is incorrect.", 401)
        ids = [r["id"] for r in self._q("SELECT id FROM documents WHERE user_id=?", (uid,))]
        self._q("DELETE FROM users WHERE id=?", (uid,), write=True)
        return ids

    # ---------------------------------------------------------------------------------------- sessions
    def new_session(self, uid):
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._q("DELETE FROM sessions WHERE expires<?", (now,), write=True)
        self._q("INSERT INTO sessions(token,user_id,created,expires) VALUES(?,?,?,?)", (_tok(token), uid, now, now + SESSION_DAYS * 86400), write=True)
        return token

    def user_for(self, token):
        if not token:
            return None
        return self._q("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires>?", (_tok(token), time.time()), one=True)

    def end_session(self, token):
        if token:
            self._q("DELETE FROM sessions WHERE token=?", (_tok(token),), write=True)

    # ---------------------------------------------------------------------------------------- documents
    def add_document(self, doc_id, uid, filename, typ, size, pages):
        self._q("INSERT INTO documents(id,user_id,filename,type,size,pages,created,status) VALUES(?,?,?,?,?,?,?,?)",
                (doc_id, uid, filename, typ, size, pages, time.time(), "processing"), write=True)

    def finish_document(self, doc_id, status, stats=None, error=None):
        self._q("UPDATE documents SET status=?, stats=?, error=? WHERE id=?",
                (status, json.dumps(stats) if stats is not None else None, error, doc_id), write=True)

    def owner(self, doc_id):
        r = self._q("SELECT user_id FROM documents WHERE id=?", (doc_id,), one=True)
        return r["user_id"] if r else None

    def owned_ids(self):
        return {r["id"] for r in self._q("SELECT id FROM documents")}

    @staticmethod
    def doc_public(r):
        return dict(id=r["id"], filename=r["filename"], type=r["type"], size=r["size"], pages=r["pages"], created=r["created"],
                    status=r["status"], stats=json.loads(r["stats"]) if r["stats"] else None, error=r["error"])

    def documents(self, uid):
        return [self.doc_public(r) for r in self._q("SELECT * FROM documents WHERE user_id=? ORDER BY created DESC", (uid,))]

    def document(self, uid, doc_id):
        r = self._q("SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, uid), one=True)
        return self.doc_public(r) if r else None

    def rename(self, uid, doc_id, filename):
        filename = re.sub(r"[\x00-\x1f]", "", (filename or "").strip())[:160]
        if not filename:
            raise AuthError("PF-400", "The name cannot be empty.")
        self._q("UPDATE documents SET filename=? WHERE id=? AND user_id=?", (filename, doc_id, uid), write=True)

    def delete_document(self, uid, doc_id):
        self._q("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, uid), write=True)


def doc_stats(doc):
    """Library summary of a processed document (what the 'My documents' list and filters show)."""
    bl = doc.get("blocks", [])
    pages = doc.get("pages", [])
    flagged = sum(1 for b in bl if b.get("verification") != "verified")
    return dict(pages=len(pages), blocks=len(bl), tables=sum(b.get("type") == "table" for b in bl),
                equations=sum(b.get("type") == "equation" for b in bl) + sum(1 for b in bl if b.get("inline_math")),
                figures=sum(b.get("type") in ("figure", "chart", "diagram") for b in bl),
                handwriting=sum(str(b.get("type", "")).startswith("handwritten") for b in bl),
                scanned=sum(1 for p in pages if p.get("route") in ("ocr", "embedded_ocr")),
                verified=len(bl) - flagged, flagged=flagged,
                furniture=sum(1 for b in bl if (b.get("furniture") or {}).get("excluded")),
                seconds=(doc.get("timings") or {}).get("total"), warnings=len(doc.get("warnings") or []))
