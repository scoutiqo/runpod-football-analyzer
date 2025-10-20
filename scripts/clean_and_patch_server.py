import io, re, sys

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
orig = txt

# -------- Remove obvious PowerShell and console garbage lines ----------
clean_lines = []
for ln in txt.splitlines():
    if re.match(r'^\s*\$[A-Za-z_]\w*\s*=', ln):             # $prefix = ..., $raw = ...
        continue
    if re.match(r'^\s*PS\s+[A-Z]:\\', ln):                   # PS C:\... prompt lines
        continue
    if re.match(r'^\s*At line:\d+', ln):                     # PS error tail
        continue
    if re.match(r'^\s*CategoryInfo', ln) or 'FullyQualifiedErrorId' in ln:
        continue
    if re.match(r'^\s*>>\s', ln):                            # pasted ">> " from REPL
        continue
    if '-replace ' in ln and '$' in ln:                      # PS -replace scripts
        continue
    clean_lines.append(ln)

txt = "\n".join(clean_lines) + "\n"

# -------- Remove stray backticks (from PS `r`n etc) --------------------
txt = txt.replace('`r', '').replace('`n', '').replace('`t', '')

# -------- Fix any duplicated / mis-indented bits created earlier -------
# remove any unindented "while True:" that slipped to column 0
txt = re.sub(r'(?m)^\s*while True:\s*$', "            while True:", txt)

# remove any duplicate unindented "last_ball = None" at column 0
txt = re.sub(r'(?m)^last_ball\s*=\s*None\s*$', "", txt)

# -------- Guard ball KF: bx,by = metr.ballkf.step(bxy, fps=fps) -------
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

# -------- Ensure StaticFiles UI mount (optional /web) ------------------
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
        insert_at = m.end()
        txt = txt[:insert_at] + '\n\n# Serve minimal UI\napp.mount("/web", StaticFiles(directory="web", html=True), name="web")\n' + txt[insert_at:]
    else:
        txt += '\n# Serve minimal UI\napp.mount("/web", StaticFiles(directory="web", html=True), name="web")\n'

# -------- Write back if changed ----------------------------------------
if txt != orig:
    io.open(P, "w", encoding="utf-8", newline="\n").write(txt)
    print("server.py cleaned & patched.")
else:
    print("server.py unchanged (already clean).")
