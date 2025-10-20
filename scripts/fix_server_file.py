import io, re

P = r".\server\server.py"
src = io.open(P, "r", encoding="utf-8").read()
orig = src

lines = src.splitlines()

# Remove stray PowerShell/transcript noise
keep = []
for ln in lines:
    if re.match(r'^\s*\$[A-Za-z_]\w*\s*=', ln):   # e.g. $prefix = $matches[1]
        continue
    if re.match(r'^\s*Value\[\d+\]:', ln):        # transcript noise
        continue
    if re.match(r'^\s*PS\s+[A-Z]:\\', ln):        # PS prompt line
        continue
    if re.match(r'^\s*>>\s', ln):                 # '>> ' prompt
        continue
    keep.append(ln)

src = "\n".join(keep) + "\n"

# Guard the ball KF line so it never crashes
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

src = re.sub(
    r'bx,by\s*=\s*metr\.ballkf\.step\(\s*bxy\s*,\s*fps\s*=\s*fps\s*\)',
    guard,
    src
)

# Ensure StaticFiles mount so /web works (optional, harmless if already present)
if "from fastapi.staticfiles import StaticFiles" not in src:
    src = re.sub(
        r'(from\s+fastapi\s+import\s+[^\n]+\n)',
        r'\1from fastapi.staticfiles import StaticFiles\n',
        src,
        count=1
    )

if 'app.mount("/web"' not in src:
    m = re.search(r'(^\s*app\s*=\s*FastAPI\s*\(.*?\)\s*)', src, flags=re.S)
    if m:
        insert_at = m.end()
        src = src[:insert_at] + '\n\n# Serve minimal UI\napp.mount("/web", StaticFiles(directory="web", html=True), name="web")\n' + src[insert_at:]
    else:
        src += '\n# Serve minimal UI\napp.mount("/web", StaticFiles(directory="web", html=True), name="web")\n'

if src != orig:
    io.open(P, "w", encoding="utf-8", newline="\n").write(src)
    print("server.py cleaned & patched.")
else:
    print("server.py already clean (no changes).")
