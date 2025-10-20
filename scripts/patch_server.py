import io, re

P = r".\server\server.py"
src = io.open(P, "r", encoding="utf-8").read()
orig = src

# ---- (A) Guard the ball-KF call if present ----------------------------------
# Replace:  bx,by = metr.ballkf.step(bxy, fps=fps)
# with a safe fallback that never crashes on None
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
    r'\bbx,by\s*=\s*metr\.ballkf\.step\(\s*bxy\s*,\s*fps\s*=\s*fps\s*\)',
    guard,
    src,
    flags=re.M
)

# ---- (B) Ensure StaticFiles UI at /web (harmless if already present) ---------
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

# Write out only if changed
if src != orig:
    io.open(P, "w", encoding="utf-8", newline="\n").write(src)
    print("server.py patched OK.")
else:
    print("server.py already OK (no changes).")
