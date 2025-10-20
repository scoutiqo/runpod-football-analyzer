import io, re

P = r".\server\server.py"
txt = io.open(P, "r", encoding="utf-8").read()
lines = txt.splitlines()

def indent_of(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

i = 0
insertions = 0
while i < len(lines):
    ln = lines[i]
    if ln.rstrip().endswith("try:"):
        ind = indent_of(ln)
        # find next non-empty, non-comment line
        j = i + 1
        while j < len(lines) and (lines[j].strip() == "" or lines[j].lstrip().startswith("#")):
            j += 1
        # if next significant line is NOT indented more than the try, the body is empty
        if j >= len(lines) or indent_of(lines[j]) <= ind or lines[j].lstrip().startswith(("except", "finally")):
            lines.insert(i + 1, " " * (ind + 4) + "pass")
            insertions += 1
            i += 1  # skip the inserted line
    i += 1

if insertions:
    io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"Inserted pass into {insertions} empty try-block(s).")
else:
    print("No empty try-blocks found.")
