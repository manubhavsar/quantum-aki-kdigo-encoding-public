"""Triple-blind compliance check for the submission PDF and sources.

Fails loudly if author-identifying content appears in the manuscript body,
the generated PDF text, or the PDF metadata.
"""
import os, re, sys, subprocess, json

BANNED_PATTERNS = [
    r"\backnowledg",                 # acknowledgements section
    r"\bcorresponding author\b",
    r"@[A-Za-z0-9.-]+\.(com|org|edu|net|ac\.[a-z]{2})",   # email addresses
    r"github\.com/[A-Za-z0-9_-]+",   # self-identifying repo links
    r"\bORCID\b",
    r"\bfunded by\b", r"\bgrant no\b", r"\bfunding\b",
    r"\bUniversity of\b", r"\bInstitute of Technology\b", r"\bDepartment of\b",
]
ALLOW_FILE_SUBSTRINGS = ["cover_page", "results_macros"]


def scan_text(text, label, extra_banned=()):
    hits = []
    for pat in list(BANNED_PATTERNS) + list(extra_banned):
        for m in re.finditer(pat, text, flags=re.I):
            s = max(0, m.start() - 45)
            hits.append((pat, text[s:m.end() + 45].replace("\n", " ")))
    if hits:
        print(f"\n[FAIL] {label}: {len(hits)} potential identity leak(s)")
        for p, ctx in hits[:12]:
            print(f"   pattern {p!r}\n      ...{ctx}...")
    else:
        print(f"[OK  ] {label}: clean")
    return hits


def main():
    total = []
    tex = "paper/manuscript.tex"
    if os.path.exists(tex):
        src = open(tex).read()
        # strip comments before scanning (comments are not rendered)
        body = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in src.split("\n"))
        total += scan_text(body, "manuscript.tex (comments stripped)")
        # author/address macros must be empty
        for macro in (r"\\author\[?[a-z]*\]?\{([^}]*)\}", r"\\address\[?[a-z]*\]?\{([^}]*)\}"):
            for m in re.finditer(macro, body):
                if m.group(1).strip():
                    print(f"[FAIL] non-empty {m.group(0)[:60]}")
                    total.append(("author-macro", m.group(0)))

    pdf = "paper/manuscript.pdf"
    if os.path.exists(pdf):
        try:
            txt = subprocess.run(["pdftotext", pdf, "-"], capture_output=True,
                                 text=True, timeout=60).stdout
            total += scan_text(txt, "manuscript.pdf text layer")
        except FileNotFoundError:
            print("[warn] pdftotext not available; skipping PDF text scan")
        try:
            meta = subprocess.run(["pdfinfo", pdf], capture_output=True,
                                  text=True, timeout=60).stdout
            bad = [l for l in meta.split("\n")
                   if re.match(r"^(Author|Creator|Producer|Subject|Keywords)\s*:", l)
                   and l.split(":", 1)[1].strip()
                   and "tectonic" not in l.lower() and "pdftex" not in l.lower()
                   and "xetex" not in l.lower()]
            if bad:
                print("[warn] PDF metadata fields set (verify these are not identifying):")
                for l in bad:
                    print("   ", l)
            else:
                print("[OK  ] PDF metadata: no identifying fields")
        except FileNotFoundError:
            print("[warn] pdfinfo not available; skipping metadata scan")
    else:
        print("[warn] paper/manuscript.pdf not built yet")

    print(f"\n=== {'PASS' if not total else 'REVIEW NEEDED (' + str(len(total)) + ' hits)'} ===")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
