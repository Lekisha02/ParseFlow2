"""Mathematical content: detection on the native text layer and LaTeX reconstruction from glyph geometry.

No model is needed for digital PDFs: TeX-produced (and most other) PDFs keep every math glyph with its font,
size and baseline. Structure is rebuilt deterministically:

  * fractions      - a horizontal rule (vector line) with glyph groups directly above and below it
  * big operators  - sum/prod/integral/union... with smaller glyphs stacked above/below -> \\sum_{..}^{..}
  * radicals       - a radical sign followed by an overbar rule -> \\sqrt{..}
  * scripts        - smaller glyphs whose baseline is raised/lowered relative to the main baseline
  * delimiters     - tall brackets -> \\left( .. \\right)
  * symbols        - Unicode -> LaTeX command table; TeX font families -> \\mathbb, \\mathcal, \\mathfrak, \\mathbf

Every reconstruction returns (latex, confidence, notes). When glyphs cannot be mapped (unmapped/private-use
codes) the confidence drops and the caller keeps the original text and flags the block instead of inventing.

Scanned math needs an image recognizer; see recognizers.py (optional, pluggable, never downloads weights).
"""
import re
import statistics
import unicodedata

MATH_FONT = re.compile(r"(cmmi|cmsy|cmex|cmbsy|cmmib|msbm|msam|eufm|eusm|eusb|rsfs|stmary|wasy|esint|lmmath|latinmodern-math|"
                       r"math|symbol|stix|cambria\s*math|mtextra|mt\s*extra|euclid|txsy|pxsy|txex|pxex|txmi|pxmi|zxmi|ntxmi|"
                       r"xits|asana|texgyre.*math|mathjax|libertinusmath|fira.?math|dejavumath)", re.I)
TEXTY_SYMBOL_FONT = re.compile(r"(wingding|dingbat|zapf)", re.I)

GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ϵ": "epsilon", "ζ": "zeta", "η": "eta",
         "θ": "theta", "ϑ": "vartheta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
         "ϖ": "varpi", "ρ": "rho", "ϱ": "varrho", "σ": "sigma", "ς": "varsigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
         "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda",
         "Ξ": "Xi", "Π": "Pi", "Σ": "Sigma", "Υ": "Upsilon", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega", "ε": "varepsilon"}
# U+03B5 'ε' is \varepsilon in TeX output, U+03F5 'ϵ' is \epsilon
GREEK["ε"] = "varepsilon"; GREEK["φ"] = "varphi"
SYMBOLS = {
    "∑": r"\sum", "∏": r"\prod", "∐": r"\coprod", "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    "⋃": r"\bigcup", "⋂": r"\bigcap", "⨁": r"\bigoplus", "⨂": r"\bigotimes", "⨀": r"\bigodot", "⋁": r"\bigvee", "⋀": r"\bigwedge",
    "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "≅": r"\cong",
    "∝": r"\propto", "≪": r"\ll", "≫": r"\gg", "≺": r"\prec", "≻": r"\succ", "⪯": r"\preceq", "⪰": r"\succeq", "≼": r"\preceq",
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni", "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq", "⊇": r"\supseteq",
    "⊊": r"\subsetneq", "∪": r"\cup", "∩": r"\cap", "∖": r"\setminus", "∅": r"\emptyset", "∀": r"\forall", "∃": r"\exists",
    "∄": r"\nexists", "¬": r"\neg", "∧": r"\wedge", "∨": r"\vee", "⊕": r"\oplus", "⊗": r"\otimes", "⊙": r"\odot", "⊖": r"\ominus",
    "∂": r"\partial", "∇": r"\nabla", "∞": r"\infty", "ℓ": r"\ell", "ℏ": r"\hbar", "℘": r"\wp", "ℜ": r"\Re", "ℑ": r"\Im", "ℵ": r"\aleph",
    "×": r"\times", "÷": r"\div", "·": r"\cdot", "⋅": r"\cdot", "∗": r"\ast", "∘": r"\circ", "∙": r"\bullet", "•": r"\bullet",
    "±": r"\pm", "∓": r"\mp", "−": "-", "–": "-", "→": r"\rightarrow", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow", "↦": r"\mapsto", "⟶": r"\longrightarrow",
    "⟹": r"\Longrightarrow", "⟺": r"\Longleftrightarrow", "↑": r"\uparrow", "↓": r"\downarrow", "↪": r"\hookrightarrow",
    "⊥": r"\perp", "∥": r"\parallel", "∣": r"\mid", "|": "|", "‖": r"\|", "⟨": r"\langle", "⟩": r"\rangle", "〈": r"\langle", "〉": r"\rangle",
    "⌊": r"\lfloor", "⌋": r"\rfloor", "⌈": r"\lceil", "⌉": r"\rceil", "√": r"\sqrt", "′": "'", "″": "''", "∠": r"\angle",
    "△": r"\triangle", "□": r"\square", "◦": r"\circ", "⋯": r"\cdots", "…": r"\ldots", "⋮": r"\vdots", "⋱": r"\ddots",
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "ℙ": r"\mathbb{P}",
    "⊤": r"\top", "⊢": r"\vdash", "⊨": r"\models", "†": r"\dagger", "‡": r"\ddagger", "≔": r":=", "∶": ":", "⊲": r"\triangleleft",
    "⋆": r"\star", "★": r"\star", "♯": r"\sharp", "♭": r"\flat", "°": r"^{\circ}", "{": r"\{", "}": r"\}", "#": r"\#", "%": r"\%",
    "&": r"\&", "_": r"\_", "$": r"\$", "\\": r"\backslash", "~": r"\sim", "^": r"\hat{}", "‰": r"\permil",
}
BIG_OPS = {r"\sum", r"\prod", r"\coprod", r"\bigcup", r"\bigcap", r"\bigoplus", r"\bigotimes", r"\bigodot", r"\bigvee", r"\bigwedge"}
INTEGRALS = {r"\int", r"\iint", r"\iiint", r"\oint"}
FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "arcsin", "arccos", "arctan", "log", "ln", "lg",
         "exp", "lim", "liminf", "limsup", "max", "min", "sup", "inf", "det", "dim", "ker", "deg", "gcd", "lcm", "arg", "Pr", "hom", "mod"}
OPEN, CLOSE = set("([{⟨⌊⌈|‖"), set(")]}⟩⌋⌉|‖")
MATH_UNICODE = set(GREEK) | {k for k in SYMBOLS if k not in "{}#%&_$\\~^|•·…°–"} | set("=<>+") | set("∑∏∫")


def is_math_char(ch):
    c = ch.c
    if not c.strip():
        return False
    if TEXTY_SYMBOL_FONT.search(ch.font):
        return False
    if MATH_FONT.search(ch.font):
        return True
    if c in MATH_UNICODE or c in GREEK:
        return True
    cp = ord(c)
    return 0x2200 <= cp <= 0x22FF or 0x1D400 <= cp <= 0x1D7FF or 0x27C0 <= cp <= 0x27EF or 0x2980 <= cp <= 0x2AFF


def math_density(chars):
    vis = [c for c in chars if c.c.strip()]
    if not vis:
        return 0.0
    return sum(1 for c in vis if is_math_char(c)) / len(vis)


# ---------------------------------------------------------------------------------------------
# Glyph -> token
# ---------------------------------------------------------------------------------------------
def _font_cmd(ch):
    f = ch.font.lower()
    if "msbm" in f or "bbold" in f or "dsrom" in f:
        return "mathbb"
    if "eufm" in f or "eufb" in f or "fraktur" in f:
        return "mathfrak"
    if ("cmsy" in f or "eusm" in f or "rsfs" in f or "calligraph" in f) and "A" <= ch.c <= "Z":
        return "mathcal"
    if ("cmbx" in f or "bold" in f or "cmmib" in f) and ch.c.isalnum():
        return "mathbf"
    return None


def _math_alnum(c):
    """Map Mathematical Alphanumeric Symbols (U+1D400..) back to (letter, style)."""
    cp = ord(c)
    if 0x1D400 <= cp <= 0x1D7FF:
        name = unicodedata.name(c, "")
        m = re.match(r"MATHEMATICAL (.*?) (CAPITAL|SMALL|DIGIT) (\w+)", name)
        if m:
            style, kind, letter = m.groups()
            base = letter if kind == "CAPITAL" else letter.lower() if kind == "SMALL" else str("ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split().index(letter)) if letter in "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split() else letter
            if len(base) > 1:   # greek names
                base = "\\" + (base.lower() if kind == "SMALL" else base.capitalize())
            cmd = "mathbb" if "DOUBLE-STRUCK" in style else "mathfrak" if "FRAKTUR" in style else "mathcal" if "SCRIPT" in style else "mathbf" if "BOLD" in style else None
            return base, cmd
    return None


SUPERS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni"))
SUBS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₙₘₖₗₚₛₜ", "0123456789+-=()aeoxnmklpst"))


def token(ch):
    """Return (latex, ok) for one glyph. ok=False for glyphs we cannot map."""
    c = ch.c
    if c in SUPERS:
        return "^{" + SUPERS[c] + "}", True
    if c in SUBS:
        return "_{" + SUBS[c] + "}", True
    if c in GREEK:
        return "\\" + GREEK[c], True
    if c in SYMBOLS:
        return SYMBOLS[c], True
    ma = _math_alnum(c)
    if ma:
        base, cmd = ma
        return (f"\\{cmd}{{{base}}}" if cmd else base), True
    if c == "\ufffd" or 0xE000 <= ord(c) <= 0xF8FF:
        return "", False
    cmd = _font_cmd(ch)
    if cmd and c.isalnum():
        return f"\\{cmd}{{{c}}}", True
    if c.isalnum() or c in "+-=<>()[],.;:!?'/*|@":
        return c, True
    cat = unicodedata.category(c)
    if cat.startswith(("L", "N")):
        return c, True
    return c, cat.startswith(("S", "P"))


# ---------------------------------------------------------------------------------------------
# Structure reconstruction
# ---------------------------------------------------------------------------------------------
class _Atom:
    __slots__ = ("x0", "y0", "x1", "y1", "base", "size", "latex", "kind", "ok")

    def __init__(self, x0, y0, x1, y1, base, size, latex, kind="ord", ok=True):
        self.x0, self.y0, self.x1, self.y1, self.base, self.size, self.latex, self.kind, self.ok = x0, y0, x1, y1, base, size, latex, kind, ok

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


def _glyph_box(ch):
    """Tighter vertical box than the font ascender/descender box: [baseline - 0.72 size, baseline + 0.22 size]."""
    return (ch.x0, ch.oy - 0.72 * ch.size, ch.x1, ch.oy + 0.22 * ch.size)


def _atoms_from_chars(chars):
    out = []
    for ch in chars:
        if not ch.c.strip():
            continue
        lx, ok = token(ch)
        if not ok and not lx:
            out.append(None)          # unmapped glyph (e.g. a big delimiter with no Unicode): counted, not placed
            continue
        x0, y0, x1, y1 = _glyph_box(ch)
        kind = "op" if lx in BIG_OPS else "int" if lx in INTEGRALS else "rad" if lx == r"\sqrt" else "ord"
        if kind in ("op", "int", "rad"):
            y0, y1 = ch.y0, ch.y1    # big glyphs: use the font box
        a = _Atom(x0, y0, x1, y1, ch.oy, ch.size, lx, kind, ok)
        out.append(a)
    return out


def build_latex(chars, hrules=(), display=False):
    """Reconstruct LaTeX for a set of glyphs (one inline run or one display equation).
    hrules: horizontal rule segments (x0, y, x1, y) near the glyphs (fraction bars, radical overbars).
    Returns (latex, confidence, notes)."""
    raw_atoms = _atoms_from_chars(chars)
    atoms = [a for a in raw_atoms if a is not None]
    if not atoms:
        return "", 0.0, ["no glyphs"]
    notes = []
    bad = sum(1 for a in raw_atoms if a is None or not a.ok)
    rules = sorted([r for r in hrules], key=lambda r: r[2] - r[0])   # shortest first: inner fractions before outer
    used_rules = []
    atoms = _collapse_radicals(atoms, rules, used_rules, notes)
    atoms = _collapse_fractions(atoms, [r for r in rules if r not in used_rules], notes)
    atoms = _collapse_bigops(atoms, notes)
    latex = _linear(atoms, notes)
    latex = _post(latex)
    conf = 0.9 - 0.5 * (bad / len(atoms)) - (0.1 if any(n.startswith("script") for n in notes) else 0) - (0.1 if any(n.startswith("fraction") for n in notes) else 0)
    if bad:
        notes.append(f"{bad} glyph(s) without a Unicode mapping")
    return latex, round(max(0.05, conf), 2), notes


def _post(s):
    s = re.sub(r"\s+", " ", s).strip()
    # Unicode script characters arrive one by one: x ^{2} ^{3} -> x^{23}; and attach to the preceding atom
    for k in ("^", "_"):
        s = re.sub(r"(?:\s*\%s\{([^{}]*)\}){2,}" % k, lambda m: k + "{" + "".join(re.findall(r"\%s\{([^{}]*)\}" % k, m.group(0))) + "}", s)
        s = re.sub(r"\s+\%s\{" % k, k + "{", s)
    s = re.sub(r"\\operatorname\{(" + "|".join(sorted(FUNCS, key=len, reverse=True)) + r")\}", r"\\\1", s)
    s = s.replace(r"\Pr", r"\Pr").replace(r"\mod", r"\bmod")
    return s


def _collapse_radicals(atoms, rules, used, notes):
    out = list(atoms)
    for a in [x for x in atoms if x.kind == "rad"]:
        bar = None
        for r in rules:
            if abs(r[0] - a.x1) < 0.6 * a.size + 1 and a.y0 - 0.4 * a.size <= r[1] <= a.y0 + 0.6 * a.size:
                bar = r; break
        if bar is None:
            # no overbar: a bare radical sign applies to the next atom
            continue
        inside = [x for x in out if x is not a and bar[0] - 1 <= x.cx <= bar[2] + 1 and x.y0 >= bar[1] - 1.5 and x.y1 <= a.y1 + 1]
        index = [x for x in out if x is not a and x.x1 <= a.cx and x.y1 < a.cy and x.size < 0.8 * a.size and x.x0 >= a.x0 - a.size]
        body = _linear(inside, notes)
        idx = _linear(index, notes) if index else ""
        lx = f"\\sqrt[{idx}]{{{body}}}" if idx else f"\\sqrt{{{body}}}"
        used.append(bar)
        keep = [x for x in out if x is not a and x not in inside and x not in index]
        m = _Atom(a.x0, min([a.y0] + [x.y0 for x in inside]), bar[2], a.y1, max([a.base] + [x.base for x in inside]), max(x.size for x in inside) if inside else a.size, lx, "ord")
        out = keep + [m]
        notes.append("radical reconstructed from overbar rule")
    return out


def _collapse_fractions(atoms, rules, notes):
    out = list(atoms)
    for r in rules:
        x0, y, x1 = r[0], r[1], r[2]
        span = [a for a in out if x0 - 1.5 <= a.cx <= x1 + 1.5]
        if not span:
            continue
        sz = statistics.median(a.size for a in span)
        num = _adjacent(sorted([a for a in span if a.cy < y], key=lambda a: -a.y1), y, sz, up=True)
        den = _adjacent(sorted([a for a in span if a.cy > y], key=lambda a: a.y0), y, sz, up=False)
        if not num or not den:
            continue
        nl, dl = _linear(num, notes), _linear(den, notes)
        if not nl or not dl:
            continue
        rest = [a for a in out if a not in num and a not in den]
        base = y + 0.25 * sz
        m = _Atom(min(x0, *(a.x0 for a in num + den)), min(a.y0 for a in num), max(x1, *(a.x1 for a in num + den)), max(a.y1 for a in den),
                  base, sz, f"\\frac{{{nl}}}{{{dl}}}", "frac")
        out = rest + [m]
        notes.append("fraction reconstructed from rule")
    return out


def _adjacent(cands, y, sz, up):
    """Take the glyphs vertically connected to the rule (stop at the first vertical gap > 0.6 size)."""
    picked, edge = [], y
    for a in cands:
        d = (edge - a.y1) if up else (a.y0 - edge)
        if d > 0.75 * sz and picked:
            break
        if d > 1.2 * sz and not picked:
            break
        picked.append(a)
        edge = min(edge, a.y0) if up else max(edge, a.y1)
    return picked


def _collapse_bigops(atoms, notes):
    out = list(atoms)
    for a in sorted([x for x in atoms if x.kind == "op"], key=lambda x: x.x0):
        if a not in out:
            continue
        w = a.x1 - a.x0
        lim = [x for x in out if x is not a and a.x0 - 0.6 * w <= x.cx <= a.x1 + 0.6 * w and x.size < 0.9 * a.size]
        below = _adjacent(sorted([x for x in lim if x.cy > a.y1 - 0.15 * a.size], key=lambda x: x.y0), a.y1, a.size * 0.6, up=False)
        above = _adjacent(sorted([x for x in lim if x.cy < a.y0 + 0.15 * a.size], key=lambda x: -x.y1), a.y0, a.size * 0.6, up=True)
        below = _extend_row(below, out, a)
        above = _extend_row(above, out, a)
        if not below and not above:
            continue
        lx = a.latex + (f"_{{{_linear(below, notes)}}}" if below else "") + (f"^{{{_linear(above, notes)}}}" if above else "")
        rest = [x for x in out if x is not a and x not in below and x not in above]
        all_ = [a] + below + above
        main_base = a.base
        m = _Atom(min(x.x0 for x in all_), min(x.y0 for x in all_), max(x.x1 for x in all_), max(x.y1 for x in all_), main_base, a.size, lx, "opl")
        out = rest + [m]
        notes.append("big operator limits reconstructed")
    return out


def _extend_row(group, pool, op):
    """Limits are often wider than the operator glyph: grow the group sideways along its own band."""
    if not group:
        return group
    group = list(group)
    changed = True
    while changed:
        changed = False
        gy0 = min(x.y0 for x in group); gy1 = max(x.y1 for x in group)
        gx0 = min(x.x0 for x in group); gx1 = max(x.x1 for x in group)
        for x in pool:
            if x is op or x in group or x.size >= 0.9 * op.size or x.kind in ("op", "opl"):
                continue
            if gy0 - 0.2 * x.size <= x.cy <= gy1 + 0.2 * x.size and (gx0 - 0.6 * x.size <= x.x1 and x.x0 <= gx1 + 0.6 * x.size):
                group.append(x); changed = True
    return group


def _main_level(atoms):
    norm = [a for a in atoms if a.kind not in ("op", "int", "rad", "opl")]
    pool = norm or atoms
    big = max(a.size for a in pool)
    cand = [a for a in pool if a.size >= 0.9 * big]
    bases = sorted(a.base for a in cand)
    # most populated baseline cluster
    best, best_n = bases[0], 0
    for b in bases:
        n = sum(1 for x in bases if abs(x - b) < 0.12 * big)
        if n > best_n:
            best, best_n = b, n
    return best, big


def _linear(atoms, notes):
    """Left-to-right serialisation with sub/superscripts relative to the main baseline."""
    if not atoms:
        return ""
    base, size = _main_level(atoms)
    atoms = sorted(atoms, key=lambda a: (a.x0, a.y0))
    mains, scripts = [], []
    for a in atoms:
        if a.kind in ("frac", "opl", "op", "int") or a.size >= 0.9 * size and abs(a.base - base) < 0.18 * size:
            mains.append(a)
        elif abs(a.base - base) < 0.12 * size and a.size >= 0.8 * size:
            mains.append(a)
        else:
            scripts.append(a)
    # assign each script atom to the main atom immediately to its left
    attach = {}
    for s in scripts:
        host = None
        for m in mains:
            if m.x0 <= s.x0 + 0.2 * size:
                host = m
        key = id(host) if host else None
        pos = "sup" if s.base < base - 0.15 * size else "sub"
        attach.setdefault(key, {"sup": [], "sub": []})[pos].append(s)
    parts = []
    if None in attach:      # scripts before any main atom (pre-scripts): emit as {}^{..}
        pre = attach[None]
        parts.append("{}" + (f"_{{{_linear(pre['sub'], notes)}}}" if pre["sub"] else "") + (f"^{{{_linear(pre['sup'], notes)}}}" if pre["sup"] else ""))
    prev = None
    for m in mains:
        lx = m.latex
        if prev is not None:
            gap = m.x0 - prev.x1
            if _wordy(prev.latex) and _wordy(lx) and gap < 0.12 * size:
                parts[-1] = parts[-1] + lx      # same word (e.g. upright function name)
                prev = m
                sc = attach.get(id(m))
                if sc:
                    parts[-1] += _scripts(sc, notes)
                continue
        parts.append(lx)
        sc = attach.get(id(m))
        if sc:
            parts[-1] += _scripts(sc, notes)
            notes.append("script placement from baseline offsets")
        prev = m
    out = " ".join(p for p in parts if p)
    out = _functions(out)
    return out


def _scripts(sc, notes):
    s = ""
    if sc["sub"]:
        s += "_{" + _linear(sc["sub"], notes) + "}"
    if sc["sup"]:
        sup = _linear(sc["sup"], notes)
        s += "^{" + sup + "}" if sup not in ("'", "''") else sup
    return s


def _wordy(lx):
    return bool(re.fullmatch(r"[A-Za-z]+", lx))


def _functions(s):
    def rep(m):
        w = m.group(0)
        if w in FUNCS:
            return "\\" + w if w not in ("mod",) else r"\bmod"
        if len(w) >= 3 and w[0].isupper() and w[1:].islower():
            return r"\operatorname{" + w + "}"
        return w
    return re.sub(r"(?<!\\operatorname\{)(?<![\\A-Za-z{])[A-Za-z]{2,}(?![A-Za-z])", rep, s)


# ---------------------------------------------------------------------------------------------
# Detection helpers used by the pipeline
# ---------------------------------------------------------------------------------------------
_TEXT_PUNCT = set(".,;:!?'\"’‘“”")


def _tokens(chars):
    """Split a line's chars into whitespace-separated tokens: list of lists of chars (spaces kept separately)."""
    toks, cur = [], []
    for c in chars:
        if not c.c.strip():
            if cur:
                toks.append(cur); cur = []
            toks.append([c]); continue
        if cur and c.x0 - cur[-1].x1 > 0.25 * max(c.size, cur[-1].size, 1):
            toks.append(cur); cur = []
        cur.append(c)
    if cur:
        toks.append(cur)
    return toks


def _is_strong(c, main):
    if c.c in GREEK or (c.c in SYMBOLS and c.c not in "{}#%&_$\\~^|•·…°–—-" ) or c.c in "=<>+−":
        return True
    if MATH_FONT.search(c.font) and not TEXTY_SYMBOL_FONT.search(c.font) and c.c.strip() and c.c not in _TEXT_PUNCT:
        return True
    if len(c.c) != 1:                 # multi-code-point glyphs (ligatures, combining sequences) are text
        return False
    cp = ord(c.c)
    return 0x2200 <= cp <= 0x22FF or 0x1D400 <= cp <= 0x1D7FF or 0x27C0 <= cp <= 0x27EF or 0x2980 <= cp <= 0x2AFF


def token_class(tok, main):
    """'strong' (contains a math glyph), 'weak' (math-compatible: italic letters, digits, brackets, scripts) or 'text'."""
    vis = [c for c in tok if c.c.strip()]
    if not vis:
        return "space"
    letters0 = [c for c in vis if c.c.isalpha()]
    if len(letters0) >= 3 and all(c.c in GREEK or "\u0370" <= c.c <= "\u03ff" or "\u1f00" <= c.c <= "\u1fff" for c in letters0) \
            and not any(c.c in SYMBOLS or c.c in "=<>+" for c in vis) and not MATH_FONT.search(vis[0].font):
        return "text"      # a Greek-language word, not a formula
    if any(_is_strong(c, main) for c in vis):
        return "strong"
    letters = [c for c in vis if c.c.isalpha()]
    small = [c for c in vis if c.size < 0.85 * main]
    if not letters:
        return "weak" if all(c.c.isdigit() or c.c in "()[]/,.'|′" or c.size < 0.85 * main for c in vis) else "text"
    ital = all((c.flags & 2) or re.search(r"(ital|oblique|cmmi|-it)", c.font, re.I) for c in letters)
    big_letters = [c for c in letters if c.size >= 0.85 * main]
    if ital and len(big_letters) <= 2:
        return "weak"
    if small and big_letters and len(big_letters) <= 2 and any(c.size < 0.85 * main for c in letters):
        return "weak"      # e.g. 'Rs' with a subscript
    if small and any(c.c in "()[]/=^_" for c in vis) and len(big_letters) <= 6:
        return "weak"      # e.g. 'R/Ann_R(x_i)': scripts plus brackets/slash only occur in formulas
    return "text"


def inline_runs(chars):
    """Split a text line's chars into [(is_math, [chars])]. A math run is a maximal sequence of strong/weak tokens
    containing at least one strong token (a real math glyph); prose words always end a run, so ordinary italic
    words or footnote markers are never wrapped. Trailing sentence punctuation stays outside the math."""
    if not chars:
        return []
    sizes = [c.size for c in chars if c.c.strip()]
    main = statistics.median(sizes) if sizes else 10
    toks = _tokens(chars)
    cls = [token_class(t, main) for t in toks]
    out = []
    i = 0
    n = len(toks)
    while i < n:
        if cls[i] in ("strong", "weak"):
            j = i
            while j + 1 < n and (cls[j + 1] in ("strong", "weak") or (cls[j + 1] == "space" and j + 2 < n and cls[j + 2] in ("strong", "weak"))):
                j += 1
            run = [k for k in range(i, j + 1)]
            if any(cls[k] == "strong" for k in run):
                cs = [c for k in run for c in toks[k]]
                # keep sentence punctuation that ends the run as text
                tail = []
                while cs and (not cs[-1].c.strip() or (cs[-1].c in ".,;:" and cs[-1].size >= 0.85 * main)):
                    tail.insert(0, cs.pop())
                # a lone '=' or '+' between prose words is still math, but a run made only of punctuation is not
                if [c for c in cs if c.c.strip()]:
                    out.append((True, cs))
                if tail:
                    out.append((False, tail))
            else:
                out.append((False, [c for k in run for c in toks[k]]))
            i = j + 1
        else:
            out.append((False, toks[i]))
            i += 1
    merged = []
    for m, cs in out:
        if merged and merged[-1][0] == m and not m:
            merged[-1] = (m, merged[-1][1] + cs)
        else:
            merged.append((m, cs))
    return merged


def line_math_profile(chars):
    """(n_strong, n_weak, n_text) token counts for a line - used to spot display equations."""
    sizes = [c.size for c in chars if c.c.strip()]
    main = statistics.median(sizes) if sizes else 10
    cls = [token_class(t, main) for t in _tokens(chars)]
    return cls.count("strong"), cls.count("weak"), cls.count("text")


EQ_NUMBER = re.compile(r"^\(\s*(?:[A-Z]?\d+(?:\.\d+)*[a-z]?|[ivx]+|\*+|[A-Z])\s*\)$")


def split_equation_number(chars, region_x1):
    """Separate a trailing '(3.2)' equation label from a display equation."""
    vis = [c for c in chars if c.c.strip()]
    if len(vis) < 4:
        return chars, None
    txt = "".join(c.c for c in vis)
    m = re.search(r"\(\s*[A-Z]?\d+(?:\.\d+)*[a-z]?\s*\)$|\(\s*[ivx]+\s*\)$", txt)
    if not m:
        return chars, None
    n = len(m.group(0).replace(" ", ""))
    tail = vis[-n:]
    rest = [c for c in chars if c not in tail]
    restv = [c for c in rest if c.c.strip()]
    if restv and tail[0].x0 - max(c.x1 for c in restv) > 1.5 * tail[0].size:
        return rest, "".join(c.c for c in tail)
    return chars, None
