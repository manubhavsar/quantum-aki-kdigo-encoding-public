"""Produce a single self-contained manuscript by inlining every input directive.

The working manuscript pulls sections, generated tables and numeric macros from
separate files so that results can be regenerated. For submission a single file
is easier to hand over, so this resolves the includes in place and leaves a
provenance comment where each was expanded.
"""
import re, os, sys

SRC = "paper/manuscript.tex"
OUT = "paper/manuscript_final.tex"


def expand(path, depth=0):
    if depth > 5:
        raise RuntimeError("input nesting too deep")
    text = open(path).read()

    def repl(m):
        name = m.group(1).strip()
        cand = name if name.endswith(".tex") else name + ".tex"
        full = os.path.join("paper", cand)
        if not os.path.exists(full):
            full = os.path.join(os.path.dirname(path), cand)
        if not os.path.exists(full):
            print(f"  [warn] unresolved input: {name}")
            return m.group(0)
        body = expand(full, depth + 1).rstrip()
        return (f"% ---- begin {cand} (inlined) ----\n{body}\n"
                f"% ---- end {cand} ----")

    return re.sub(r"\\input\{([^}]*)\}", repl, text)


def main():
    out = expand(SRC)
    # graphics live one level up from paper/; make paths relative to the file itself
    out = out.replace("{../results/figures/", "{figures/")
    open(OUT, "w").write(out)
    n_in = len(re.findall(r"\\input\{", out))
    print(f"wrote {OUT} ({len(out.splitlines())} lines, {n_in} unresolved inputs)")


if __name__ == "__main__":
    main()
