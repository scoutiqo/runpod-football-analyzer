import io

P = r".\server\server.py"
src = io.open(P, "r", encoding="utf-8").read()
lines = src.splitlines()

def indent(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

out = []
stack = []  # each: {ind, has_handler, had_body}

i = 0
while i < len(lines):
    ln = lines[i]
    ind = indent(ln)
    stripped = ln.lstrip()

    # close try blocks that end before this line (dedent)
    while stack and ind <= stack[-1]["ind"]:
        t = stack.pop()
        # if body was empty (no statement), insert a pass at one indent deeper
        if not t["had_body"]:
            out.append(" " * (t["ind"] + 4) + "pass")
        # if no except/finally seen, add generic except/pass
        if not t["has_handler"]:
            out.append(" " * t["ind"] + "except Exception:")
            out.append(" " * (t["ind"] + 4) + "pass")

    # mark handler lines
    if stack and (stripped.startswith("except") or stripped.startswith("finally:")):
        stack[-1]["has_handler"] = True

    # track body presence for the top try
    if stack and stripped and not stripped.startswith(("#", "except", "finally")):
        # any real line at deeper indent counts as body
        if indent(ln) > stack[-1]["ind"]:
            stack[-1]["had_body"] = True

    # new try?
    if ln.rstrip().endswith("try:"):
        stack.append({"ind": ind, "has_handler": False, "had_body": False})

    out.append(ln)
    i += 1

# EOF: close remaining try blocks
while stack:
    t = stack.pop()
    if not t["had_body"]:
        out.append(" " * (t["ind"] + 4) + "pass")
    if not t["has_handler"]:
        out.append(" " * t["ind"] + "except Exception:")
        out.append(" " * (t["ind"] + 4) + "pass")

io.open(P, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("Fixed dangling/empty try blocks.")
