import io, re, sys, os

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
orig = txt

# ---- (A) HARD CLEAN: drop any line that looks like PowerShell or transcript junk ----
clean = []
for ln in txt.splitlines():
    s = ln.strip()
    # any dollar sign means PS or env var—Python never uses it
    if "$" in ln:
        continue
    # PS prompts / error decorations / here-string sentinels
    if s.startswith("PS ") or s.startswith("At line:") or "FullyQualifiedErrorId" in s:
        continue
    if s in ("@'", "'@"):
        continue
    # stray REPL leader
    if s.startswith(">> "):
        continue
    clean.append(ln)

txt = "\n".join(clean) + "\n"

# strip backticks that sometimes got inserted (`r `n ...)
txt = txt.replace("`r", "").replace("`n", "").replace("`t", "")

# ---- (B) Fix obvious indentation damage caused earlier (safe no-ops if fine) ----
# move any top-level "while True:" to 12-space indent (inside gen())
txt = re.sub(r'(?m)^\s*while True:\s*$', "            while True:", txt)
# drop any top-level duplicate "last_ball = None"
txt = re.sub(r'(?m)^last_ball\s*=\s*None\s*$', "", txt)

# ---- (C) Guard ball KF -> never unpack None ----
guard = r"""
res = metr.ballkf.step(bxy, fps=fps)
if res is None:
    if bxy is not None:
        bx,by = bxy
    elif last_ball is not None:
        bx,by = last_ball
    else:
        bx,by = (W//2, H//2)
else:
    bx,by = res
last_ball = (bx,by)
""".strip()

txt = re.sub(
    r'\bbx,by\s*=\s*metr\.ballkf\.step\(\s*bxy\s*,\s*fps\s*=\s*fps\s*\)',
    guard,
    txt
)

# ---- (D) Minimal UI mount at /web (harmless if duplicated) ----
if "from fastapi.staticfiles import StaticFiles" not in txt:
    txt = re.sub(
        r'(from\s+fastapi\s+import\s+[^\n]+\n)',
        r'\1from fastapi.staticfiles import StaticFiles\n',
        txt,
        count=1
    )

if 'app.mount("/web"' not in txt:
    m = re.search(r'(^\s*app\s*=\s*FastAPI\s*\(.*?\)\s*)', txt, flags=re.S)
    if m:
        i = m.end()
        txt = txt[:i] + '\n\n# Serve minimal UI\napp.mount("/web", StaticFiles(directory="web", html=True), name="web")\n' + txt[i:]
    else:
        txt += '\n# Serve minimal UI\napp.mount("/web", StaticFiles(directory="web", html=True), name="web")\n'

# ---- (E) Write back ----
io.open(P, "w", encoding="utf-8", newline="\n").write(txt)
print("server.py cleaned (aggressive) & patched.")
