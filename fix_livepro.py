import io, re, os

PATH = r".\server\server.py"

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

orig = src
changed = False

# --- A) remove a stray "try:" that directly precedes "while True:" in the live_pro generator
# match same indent level
src2 = re.sub(r'\n([ \t]*)try:\s*\n(\1)while True:', r'\n\2while True:', src, count=1, flags=re.S)
if src2 != src:
    src = src2
    changed = True

# --- B) ensure last_ball is initialized after "i = 0" inside the live_pro gen
src2 = re.sub(r'(i\s*=\s*0)(\s*)', r'\1\2\n            last_ball = None\n', src, count=1)
if src2 != src:
    src = src2
    changed = True

# --- C) guard the Kalman ball step (avoid None unpack)
guard = (
"res = metr.ballkf.step(bxy, fps=fps)\n"
"if res is None:\n"
"    if bxy is not None:\n"
"        bx,by = bxy\n"
"    elif last_ball is not None:\n"
"        bx,by = last_ball\n"
"    else:\n"
"        bx,by = (W//2, H//2)\n"
"else:\n"
"    bx,by = res\n"
"last_ball = (bx,by)"
)
if 'bx,by = metr.ballkf.step(bxy, fps=fps)' in src:
    src = src.replace('bx,by = metr.ballkf.step(bxy, fps=fps)', guard)
    changed = True

# --- D) mount simple UI at /web if not present
if "app.mount('/web'" not in src:
    mount = (
    "\n# ---- simple static UI mount ----\n"
    "try:\n"
    "    from fastapi.staticfiles import StaticFiles\n"
    "    app.mount('/web', StaticFiles(directory='web', html=True), name='web')\n"
    "except Exception:\n"
    "    pass\n")
    src = src.rstrip() + mount
    changed = True

if changed:
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("Patched server.py")
else:
    print("No changes applied")
