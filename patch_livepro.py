import io, sys, re, os

PATH = r".\server\server.py"

with io.open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

orig = src

# --- A) add last_ball init right after "i = 0" inside /live_pro generator ---
src = re.sub(r'(i\s*=\s*0)(\s*)',
             r'\1\2\n            last_ball = None\n',
             src, count=1)

# --- B) replace the single crashing line with guarded logic ---
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

src = src.replace('bx,by = metr.ballkf.step(bxy, fps=fps)', guard)

# --- C) mount a simple static UI at /web (append, no regex games) ---
mount_snippet = (
"\n# ---- simple static UI mount ----\n"
"try:\n"
"    from fastapi.staticfiles import StaticFiles\n"
"    app.mount('/web', StaticFiles(directory='web', html=True), name='web')\n"
"except Exception:\n"
"    pass\n"
)

if "fastapi.staticfiles" not in src and "app.mount('/web'" not in src:
    src = src.rstrip() + mount_snippet

if src != orig:
    with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("Patched server.py")
else:
    print("No changes were applied (already patched?)")
