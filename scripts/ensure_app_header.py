import io, re

P = r".\server\server.py"
src = io.open(P, "r", encoding="utf-8").read()
lines = src.splitlines()

def find_first_decorator_idx(lines):
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("@app."):
            return i
    return None

def has_app_assign_before(lines, idx):
    pat = re.compile(r'^\s*app\s*=\s*FastAPI\s*\(', re.M)
    # search only up to the decorator block (exclusive)
    head = "\n".join(lines[:idx if idx is not None else len(lines)])
    return pat.search(head) is not None

def has_fastapi_import_before(lines, idx):
    pat = re.compile(r'^\s*from\s+fastapi\s+import\s+FastAPI\s*$', re.M)
    head = "\n".join(lines[:idx if idx is not None else len(lines)])
    return pat.search(head) is not None

k = find_first_decorator_idx(lines)
if k is None:
    # no decorators; just ensure header at very top if missing
    need_assign = not has_app_assign_before(lines, len(lines))
    need_import = not has_fastapi_import_before(lines, len(lines))
    header = []
    if need_import:
        header.append("from fastapi import FastAPI")
    if need_assign:
        header.append("app = FastAPI()")
    if header:
        new = "\n".join(header) + "\n" + "\n".join(lines) + "\n"
        io.open(P, "w", encoding="utf-8", newline="\n").write(new)
        print("Inserted FastAPI header at file top.")
    else:
        print("FastAPI header already present.")
else:
    # there is a decorator; ensure app+import appear BEFORE k
    need_assign = not has_app_assign_before(lines, k)
    need_import = not has_fastapi_import_before(lines, k)
    header = []
    if need_import:
        header.append("from fastapi import FastAPI")
    if need_assign:
        header.append("app = FastAPI()")
    if header:
        # insert the header right BEFORE first decorator line
        new_lines = lines[:k] + header + [""] + lines[k:]
        io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(new_lines) + "\n")
        print(f"Inserted FastAPI header above first decorator (line {k+1}).")
    else:
        print("FastAPI header already precedes first decorator.")
