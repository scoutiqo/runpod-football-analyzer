import io, re

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
orig = txt

# ---- A) Keep only the FIRST LIVE2 block, remove any later duplicates ----
# LIVE2 sections are delimited by the headers in your file:
#   ^# =================== LIVE2 ... ===================
#   ... (content) ...
#   ^# ==========================================================================
sec_pat = re.compile(
    r"(?ms)^\s*#\s*===================\s*LIVE2.*?^\s*#\s*={10,}\s*$"
)
all_secs = list(sec_pat.finditer(txt))
if len(all_secs) > 1:
    # keep first, delete the rest
    first = all_secs[0]
    keep_start, keep_end = first.span()
    # build new text by removing later sections
    pieces = []
    last = 0
    for i, m in enumerate(all_secs):
        if i == 0:
            # keep
            pieces.append(txt[last:m.start()])
            pieces.append(txt[m.start():m.end()])
        else:
            # skip (delete)
            pieces.append(txt[last:m.start()])
        last = m.end()
    pieces.append(txt[last:])
    txt = "".join(pieces)

# ---- B) Make sure numpy is imported once (needed by possession panel) ----
if re.search(r'(?m)^\s*import\s+numpy\s+as\s+np\s*$', txt) is None:
    # insert near top after the first import block if possible
    m = re.search(r'(?m)^(?:from\s+\w[^\n]*\n|import\s+\w[^\n]*\n)+', txt)
    ins = m.end() if m else 0
    txt = txt[:ins] + "import numpy as np\n" + txt[ins:]

# ---- C) Ensure only ONE StaticFiles /web mount line ----------------------
# remove duplicates of app.mount("/web", StaticFiles(...), name="web")
txt = re.sub(
    r'(?m)^\s*app\.mount\(\s*["\']/web["\']\s*,\s*StaticFiles\([^\)]*\)\s*,\s*name\s*=\s*["\']web["\']\s*\)\s*$',
    "", txt
)

# If there isn't at least one guarded mount, add one right after app=FastAPI(...)
if 'app.mount("/web", StaticFiles(' not in txt:
    if "from fastapi.staticfiles import StaticFiles" not in txt and \
       "from starlette.staticfiles import StaticFiles" not in txt:
        # inject import alongside other fastapi imports
        txt = re.sub(r'(from\s+fastapi\s+import\s+[^\n]+\n)',
                     r'\1from fastapi.staticfiles import StaticFiles\n',
                     txt, count=1)
        if "from fastapi.staticfiles import StaticFiles" not in txt:
            txt = "from fastapi.staticfiles import StaticFiles\n" + txt
    m = re.search(r'(?m)^\s*app\s*=\s*FastAPI\s*\(.*?\)\s*$', txt)
    if m:
        i = m.end()
        mount = (
            '\n# Serve minimal UI once\n'
            'try:\n'
            '    app.mount("/web", StaticFiles(directory="web", html=True), name="web")\n'
            'except Exception:\n'
            '    pass\n'
        )
        txt = txt[:i] + mount + txt[i:]

# ---- D) Write back if anything changed ----------------------------------
if txt != orig:
    io.open(P, "w", encoding="utf-8", newline="\n").write(txt)
    print("server.py: LIVE2 deduped, numpy/staticfiles normalized.")
else:
    print("server.py: no changes needed.")
