import io

P = r".\server\server.py"
src = io.open(P, "r", encoding="utf-8").read()
lines = src.splitlines()

def indent(s: str) -> int:
    # count leading spaces only; tabs aren’t expected here
    return len(s) - len(s.lstrip(" "))

# Stack of dicts: {idx_try, ind, has_handler}
stack = []
out = []
i = 0

def starts_with_any(s, prefixes):
    s2 = s.lstrip()
    return any(s2.startswith(p) for p in prefixes)

while i < len(lines):
    ln = lines[i]
    ind = indent(ln)

    # close any try blocks that end BEFORE this line (dedent)
    while stack and ind <= stack[-1]["ind"]:
        t = stack.pop()
        if not t["has_handler"]:
            # Insert handler right before current line, at try's indent
            out.append(" " * t["ind"] + "except Exception:")
            out.append(" " * (t["ind"] + 4) + "pass")
        # continue; multiple stacked tries can close here

    # Track handler lines for the current top try
    if stack:
        # if this line begins with 'except' or 'finally', mark it
        if starts_with_any(ln, ("except ", "except:", "finally:")):
            stack[-1]["has_handler"] = True

    # If this line starts a new try, push it
    if ln.rstrip().endswith("try:"):
        stack.append({"idx_try": len(out), "ind": ind, "has_handler": False})

    out.append(ln)
    i += 1

# End of file: close any remaining open tries
while stack:
    t = stack.pop()
    if not t["has_handler"]:
        out.append(" " * t["ind"] + "except Exception:")
        out.append(" " * (t["ind"] + 4) + "pass")

io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("Inserted handlers for any dangling try: blocks.")
