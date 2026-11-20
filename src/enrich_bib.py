"""Re-resolve each verified reference by DOI/arXiv id to obtain full author
names (given + family), then regenerate references.bib.

Author *initials* matter for a real submission, and the earlier verification
passes stored only family names. This does not add any new reference: it only
enriches entries that already passed verification.
"""
import json, re, os, time, urllib.request, urllib.parse, unicodedata

UA = {"User-Agent": "AKI-QML-lit-review/1.0 (mailto:research@example.org)"}


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))


def by_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        m = json.loads(r.read().decode())["message"]
    auth = []
    for a in m.get("author", []):
        fam, giv = deaccent(a.get("family", "")), deaccent(a.get("given", ""))
        if fam:
            auth.append(f"{fam}, {giv}" if giv else fam)
    dt = m.get("published-print") or m.get("published-online") or m.get("issued") or {}
    # Crossref stores some subtitles separately; rejoin so titles are complete
    title = (m.get("title") or [""])[0]
    sub = (m.get("subtitle") or [""])
    if sub and sub[0] and sub[0].lower() not in title.lower():
        title = f"{title}: {sub[0]}"
    return {"authors": auth, "title": title,
            "year": (dt.get("date-parts") or [[None]])[0][0],
            "container": (m.get("container-title") or [""])[0],
            "volume": m.get("volume"), "page": m.get("page"),
            "doi": m.get("DOI"), "type": m.get("type")}


def by_arxiv(aid):
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        xml = r.read().decode()
    names = re.findall(r"<name>(.*?)</name>", xml)
    auth = []
    for n in names:
        parts = deaccent(n).split()
        auth.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else parts[0])
    t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
    y = re.search(r"<published>(\d{4})", xml)
    return {"authors": auth, "title": " ".join(t.group(1).split()) if t else "",
            "year": int(y.group(1)) if y else None, "arxiv_id": aid,
            "container": None, "doi": None}


def main():
    recs = []
    for p in ("results/logs/cite_batch1.json", "results/logs/cite_batch2.json",
              "results/logs/cite_batch3.json", "results/logs/cite_batch4.json",
              "results/logs/cite_batch5.json"):
        if os.path.exists(p):
            for r in json.load(open(p)):
                if r.get("verified"):
                    recs.append((r["accepted"], r.get("thread", "")))
    for p in ("results/logs/doi_verification.json", "results/logs/doi_batch2.json",
              "results/logs/doi_batch3.json"):
        if os.path.exists(p):
            for r in json.load(open(p)):
                if r.get("verified"):
                    recs.append((r["record"], r.get("thread", "")))
    seen, out = set(), []
    for rec, th in recs:
        key = (rec.get("doi") or rec.get("arxiv_id") or rec.get("title", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if rec.get("doi"):
                full = by_doi(rec["doi"])
            elif rec.get("arxiv_id"):
                full = by_arxiv(rec["arxiv_id"])
            else:
                full = rec
            full["thread"] = th
            # never let enrichment silently change which paper this is
            if full.get("title") and rec.get("title"):
                a = set(re.sub(r"[^a-z ]", " ", full["title"].lower()).split())
                b = set(re.sub(r"[^a-z ]", " ", rec["title"].lower()).split())
                if len(a & b) / max(min(len(a), len(b)), 1) < 0.6:
                    print(f"  [SKIP enrich] title drift for {key}")
                    full = {**rec, "thread": th}
            out.append(full)
            print(f"  ok {key[:45]:45s} {len(full.get('authors') or [])} authors")
        except Exception as e:
            print(f"  [keep original] {key[:40]}: {str(e)[:40]}")
            out.append({**rec, "thread": th})
        time.sleep(0.8)
    json.dump(out, open("results/logs/enriched_refs.json", "w"), indent=2)
    print(f"\nenriched {len(out)} references")


if __name__ == "__main__":
    main()
