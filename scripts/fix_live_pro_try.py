import io

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
lines = txt.splitlines()

def indent(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

# find the live_pro decorator
k = next((i for i,l in enumerate(lines) if l.strip().startswith('@app.get("/live_pro")')), None)
if k is None:
    print('No @app.get("/live_pro") found. No changes.')
else:
    # walk upward to find the nearest "try:" that is at same/top-level and whose block
    # would end right before the decorator (i.e., decorator is dedented vs that try)
    i = k - 1
    fixed = False
    while i >= 0:
        ln = lines[i].rstrip("\n")
        if ln.strip().endswith("try:"):
            ind_try = indent(ln)
            # the decorator indent level
            ind_dec = indent(lines[k])
            # if decorator is NOT more indented than the try, the try-block is ending here → needs except
            if ind_dec <= ind_try:
                # insert an except just before the decorator, at the same indent as try
                lines.insert(k, " " * ind_try + "except Exception:")
                lines.insert(k + 1, " " * (ind_try + 4) + "pass")
                fixed = True
                break
        # stop if we hit a blank line or another def/class/decorator at the same level
        if lines[i].strip() and lines[i].lstrip().startswith(("@", "def ", "class ")):
            # we crossed a new block; give up
            break
        i -= 1

    if fixed:
        io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        print("Inserted missing 'except Exception: pass' before @app.get('/live_pro').")
    else:
        print("Did not find a dangling try: immediately before /live_pro. No changes.")
