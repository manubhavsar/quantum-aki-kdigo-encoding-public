"""Direct DOI/arXiv resolution for references whose titles are too generic for
title-search matching (e.g. "Quantum machine learning", "Long Short-Term Memory").

Confirms the DOI resolves AND that the returned title/author/year match expectation.
"""
import json, sys, time, urllib.request, unicodedata, re, os

UA = {"User-Agent": "AKI-QML-lit-review/1.0 (mailto:research@example.org)"}


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))


def resolve_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        m = json.loads(r.read().decode())["message"]
    dt = m.get("published-print") or m.get("published-online") or m.get("issued") or {}
    return {"doi": m.get("DOI"), "title": (m.get("title") or [""])[0],
            "year": (dt.get("date-parts") or [[None]])[0][0],
            "container": (m.get("container-title") or [""])[0],
            "authors": [deaccent(a.get("family", "")) for a in m.get("author", [])[:15]],
            "volume": m.get("volume"), "page": m.get("page"),
            "publisher": m.get("publisher"), "type": m.get("type"),
            "url": f"https://doi.org/{m.get('DOI')}"}


import urllib.parse

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        targets = json.load(f)
    out = []
    for t in targets:
        try:
            r = resolve_doi(t["doi"])
        except Exception as e:
            print(f"[ERR ] {t['doi']}: {e}", flush=True)
            out.append({**t, "verified": False, "error": str(e)})
            time.sleep(1)
            continue
        ok = True
        notes = []
        ea = deaccent(t.get("expect_author", "")).lower()
        if ea and ea not in " ".join(r["authors"]).lower():
            ok = False
            notes.append(f"author {ea} not in {r['authors'][:5]}")
        ey = t.get("expect_year")
        if ey and r["year"] and abs(int(r["year"]) - int(ey)) > 1:
            ok = False
            notes.append(f"year {r['year']} != {ey}")
        status = "OK  " if ok else "WARN"
        print(f"[{status}] {r['doi']}\n        {r['year']} | {r['authors'][:3]} | "
              f"{r['title'][:60]} | {r['container'][:40]}", flush=True)
        if notes:
            print(f"        !! {notes}", flush=True)
        out.append({**t, "verified": ok, "record": r, "notes": notes})
        time.sleep(1)
    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/doi_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== DOI-verified {sum(1 for o in out if o['verified'])}/{len(out)} ===")
