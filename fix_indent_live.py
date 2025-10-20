import io

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
lines = txt.splitlines()

# Find the /live block window to avoid touching other parts
start = None
end = None
for i, ln in enumerate(lines):
    if '@app.get("/live")' in ln.replace("'", '"'):
        start = i
        break

if start is None:
    print("Could not find @app.get(\"/live\"). Aborting.")
    raise SystemExit(1)

# find the end of this block by the next 'except Exception as _e:' or end of file
for j in range(start, len(lines)):
    if lines[j].strip() == "except Exception as _e:":
        end = j
        break
if end is None:
    end = len(lines)-1

changed = 0

# Within [start, end], remove *unindented* duplicate 'last_ball = None'
for idx in range(start, end+1):
    if lines[idx].strip() == "last_ball = None" and lines[idx].startswith("last_ball"):
        # only if it is column 0 (starts with 'l'), not the indented one
        lines[idx] = ""  # delete this line
        changed += 1
        break

# Within [start, end], fix any 'while True:' at column 0 → indent to 12 spaces
for idx in range(start, end+1):
    if lines[idx].startswith("while True:"):
        lines[idx] = "            while True:"
        changed += 1
        break

# Also ensure we have a properly indented 'last_ball = None' after 'i = 0'
for idx in range(start, end+1):
    if lines[idx].strip().startswith("i = 0"):
        # If the next non-empty line isn't the indented last_ball, insert it
        k = idx + 1
        while k <= end and lines[k].strip() == "":
            k += 1
        expected = "            last_ball = None"
        if k > end or lines[k].strip() != "last_ball = None":
            lines.insert(idx+1, expected)
            changed += 1
        break

if changed:
    io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"Patched /live block (changes: {changed})")
else:
    print("No changes applied (already OK?)")
