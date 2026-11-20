"""Verify every candidate reference resolves against Crossref / Semantic Scholar / arXiv.

Ground rule: no citation enters references.bib unless confirmed here AND the match
is corroborated by an expected first-author surname and/or publication year.

A loose title match is NOT sufficient -- short generic titles produce confident-looking
false matches (e.g. "PennyLane" matching an unrelated 2005 technical report), so every
candidate carries `expect_author` / `expect_year` and anything that fails corroboration
is reported as NEEDS_REVIEW rather than silently accepted.
"""
import json, time, sys, urllib.parse, urllib.request, re, os, copy, unicodedata


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))

UA = {"User-Agent": "AKI-QML-lit-review/1.0 (mailto:research@example.org)"}
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
TITLE_MIN = 0.85          # strict token overlap
YEAR_TOL = 1              # allow preprint/published year drift


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _norm(s):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if w]


STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "with", "to", "using"}


def _sim(a, b):
    """Symmetric token-overlap on content words -- penalises both over- and under-match."""
    A = set(_norm(a)) - STOP
    B = set(_norm(b)) - STOP
    if not A or not B:
        return 0.0
    return 2 * len(A & B) / (len(A) + len(B))          # Dice coefficient


def _corroborate(rec, expect_author, expect_year):
    """Return (ok, notes). Requires author surname and/or year to line up."""
    notes = []
    ok = True
    if expect_author:
        auths = deaccent(" ".join(rec.get("authors") or [])).lower()
        if deaccent(expect_author).lower() not in auths:
            ok = False
            notes.append(f"author '{expect_author}' not in {rec.get('authors')}")
    if expect_year and rec.get("year"):
        if abs(int(rec["year"]) - int(expect_year)) > YEAR_TOL:
            ok = False
            notes.append(f"year {rec['year']} != expected {expect_year}")
    return ok, notes


def crossref(title):
    q = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=5"
    out = []
    try:
        items = _get(url)["message"]["items"]
    except Exception as e:
        return [{"source": "crossref", "error": str(e)}]
    for it in items:
        cand = (it.get("title") or [""])[0]
        dt = it.get("published-print") or it.get("published-online") or it.get("issued") or {}
        year = (dt.get("date-parts") or [[None]])[0][0]
        out.append({
            "source": "crossref", "title": cand, "doi": it.get("DOI"), "year": year,
            "container": (it.get("container-title") or [""])[0],
            "type": it.get("type"),
            "authors": [a.get("family", "") for a in it.get("author", [])[:12]],
            "url": f"https://doi.org/{it.get('DOI')}", "sim": round(_sim(cand, title), 3),
        })
    return out


def semantic_scholar(title):
    q = urllib.parse.quote(title)
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?query=" + q +
           "&limit=5&fields=title,year,venue,externalIds,authors,url")
    out = []
    try:
        data = _get(url)
    except Exception as e:
        return [{"source": "s2", "error": str(e)}]
    for it in data.get("data", []) or []:
        ext = it.get("externalIds") or {}
        out.append({"source": "s2", "title": it.get("title", ""), "year": it.get("year"),
                    "container": it.get("venue"), "doi": ext.get("DOI"),
                    "arxiv_id": ext.get("ArXiv"), "pmid": ext.get("PubMed"),
                    "authors": [a["name"].split()[-1] for a in (it.get("authors") or [])[:12]],
                    "url": it.get("url"), "sim": round(_sim(it.get("title", ""), title), 3)})
    return out


def arxiv(title):
    q = urllib.parse.quote(f'all:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"
    out = []
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            xml = r.read().decode()
    except Exception as e:
        return [{"source": "arxiv", "error": str(e)}]
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        t = " ".join(t.group(1).split()) if t else ""
        idm = re.search(r"<id>(.*?)</id>", e)
        pub = re.search(r"<published>(\d{4})", e)
        doi = re.search(r'<arxiv:doi[^>]*>(.*?)</arxiv:doi>', e)
        out.append({"source": "arxiv", "title": t,
                    "url": idm.group(1) if idm else None,
                    "arxiv_id": idm.group(1).rsplit("/", 1)[-1].split("v")[0] if idm else None,
                    "doi": doi.group(1) if doi else None,
                    "year": int(pub.group(1)) if pub else None,
                    "authors": [n.split()[-1] for n in re.findall(r"<name>(.*?)</name>", e)[:12]],
                    "sim": round(_sim(t, title), 3)})
    return out


def verify(c):
    title = c["title"]
    expect_author = c.get("expect_author")
    expect_year = c.get("expect_year")
    pool = []
    for fn in (crossref, semantic_scholar, arxiv):
        pool.extend(fn(title))
        time.sleep(1.2)
    cands = sorted([p for p in pool if p.get("sim") is not None],
                   key=lambda p: -p["sim"])
    accepted, review = None, []
    for p in cands:
        if p["sim"] < TITLE_MIN:
            continue
        ok, notes = _corroborate(p, expect_author, expect_year)
        if ok and accepted is None:
            accepted = p
        elif not ok:
            review.append({**p, "reject_reason": notes})
    return {"query_title": title, "thread": c.get("thread", ""),
            "expect_author": expect_author, "expect_year": expect_year,
            "verified": accepted is not None, "accepted": accepted,
            "near_misses": review[:3], "top_candidates": cands[:4]}


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        cands = json.load(f)
    results = []
    for c in cands:
        r = verify(c)
        if r["verified"]:
            a = r["accepted"]
            print(f"[OK  ] {c['title'][:70]}\n"
                  f"        {a.get('year')} | {a.get('container','')[:40]} | "
                  f"{a.get('doi') or a.get('arxiv_id')} | sim={a['sim']}", flush=True)
        else:
            print(f"[FAIL] {c['title'][:70]}", flush=True)
            for p in r["top_candidates"][:2]:
                print(f"        ?? {p.get('year')} sim={p.get('sim')} "
                      f"{str(p.get('title'))[:60]} | {p.get('doi') or p.get('arxiv_id')}",
                      flush=True)
        results.append(r)
    os.makedirs(OUT, exist_ok=True)
    outname = sys.argv[2] if len(sys.argv) > 2 else "citation_verification.json"
    with open(os.path.join(OUT, outname), "w") as f:
        json.dump(results, f, indent=2)
    n = sum(1 for r in results if r["verified"])
    yrs = [r["accepted"]["year"] for r in results if r["verified"] and r["accepted"].get("year")]
    recent = sum(1 for y in yrs if y >= 2020)
    print(f"\n=== Verified {n}/{len(results)} | 2020+: {recent}/{len(yrs)} "
          f"({100*recent/max(len(yrs),1):.0f}%) ===")
