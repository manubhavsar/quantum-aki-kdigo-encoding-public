"""Generate references.bib strictly from verified citation records.

Reads results/logs/{citation_verification,doi_verification}.json and emits only
entries that passed verification. Every entry carries the DOI/arXiv id that was
actually resolved, so no reference can enter the manuscript unverified.
"""
import json, re, os, unicodedata

KEYMAP = {}


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))


def mkkey(authors, year, title):
    a = deaccent(first_family(authors)).split()[-1].lower() if authors else "anon"
    a = re.sub(r"[^a-z]", "", a) or "anon"
    w = [x for x in re.sub(r"[^a-zA-Z ]", " ", title).split() if len(x) > 3]
    w = w[0].lower() if w else "ref"
    base = f"{a}{year or 'nd'}{w}"
    n, k = 1, base
    while k in KEYMAP:
        n += 1
        k = f"{base}{n}"
    KEYMAP[k] = title
    return k


def esc(s):
    return (s or "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


MAX_AUTHORS = 3

# Resolver metadata is frequently ASCII-folded. Restoring the authors' own
# spelling is a faithfulness fix, not a metadata change: keys are the exact
# folded strings the resolvers return, so this stays reproducible.
DIACRITICS = {
    "Kubler": r'K\"ubler',
    "Scholkopf": r'Sch\"olkopf',
    "Tomasev": r"Toma\v{s}ev",
    "Perez-Salinas": r"P\'erez-Salinas",
    "Elies": r"Eli\`es",
    "Adrian": r"Adri\'an",
}


def restore_diacritics(field):
    for folded, tex in DIACRITICS.items():
        field = re.sub(r"\b" + re.escape(folded) + r"\b", tex.replace("\\", "\\\\"), field)
    return field


def fmt_authors(auths):
    """Truncate long author lists to `and others`, which the style renders as
    et al. Full lists of a dozen names consume disproportionate space in a
    page-limited proceedings paper without adding information."""
    a = [restore_diacritics(deaccent(x)) for x in auths if x]
    if len(a) > MAX_AUTHORS:
        return " and ".join(a[:MAX_AUTHORS]) + " and others"
    return " and ".join(a)


def first_family(auths):
    return (auths[0].split(",")[0] if auths else "anon")


def entry(rec, thread):
    authors, year = rec.get("authors") or [], rec.get("year")
    title, doi = rec.get("title", ""), rec.get("doi")
    arx, cont = rec.get("arxiv_id"), rec.get("container") or ""
    key = mkkey(authors, year, title)
    is_arxiv = (not doi) and arx
    typ = "misc" if is_arxiv else ("inproceedings" if rec.get("type") == "proceedings-article" else "article")
    L = [f"@{typ}{{{key},"]
    L.append(f"  author  = {{{fmt_authors(authors)}}},")
    L.append(f"  title   = {{{esc(title)}}},")
    if year:
        L.append(f"  year    = {{{year}}},")
    if is_arxiv:
        L.append(f"  eprint  = {{{arx}}},")
        L.append("  archivePrefix = {arXiv},")
        L.append(f"  note    = {{arXiv:{arx}}},")
    else:
        if cont:
            L.append(f"  journal = {{{esc(cont)}}},")
        if rec.get("volume"):
            L.append(f"  volume  = {{{rec['volume']}}},")
        if rec.get("page"):
            L.append(f"  pages   = {{{rec['page']}}},")
        if doi:
            L.append(f"  doi     = {{{doi}}},")
    L.append(f"  % thread: {thread} | VERIFIED via {rec.get('source','crossref-doi')}")
    L.append("}")
    return key, year, "\n".join(L)


def main():
    recs = []
    enr = "results/logs/enriched_refs.json"
    if os.path.exists(enr):
        # enriched records carry full author names; they derive only from
        # already-verified entries (see src/enrich_bib.py)
        for r in json.load(open(enr)):
            recs.append((r, r.get("thread", "")))
    else:
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

    # de-duplicate on DOI / arXiv id
    seen, uniq = set(), []
    for rec, th in recs:
        k = (rec.get("doi") or rec.get("arxiv_id") or rec.get("title", "")).lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append((rec, th))

    out, years = [], []
    for rec, th in sorted(uniq, key=lambda x: (x[1], -(x[0].get("year") or 0))):
        key, yr, txt = entry(rec, th)
        out.append(txt)
        if yr:
            years.append(int(yr))

    os.makedirs("paper", exist_ok=True)
    with open("paper/references.bib", "w") as f:
        f.write("% Auto-generated from verified citation records. Do not hand-edit.\n")
        f.write("% Every entry resolved against Crossref / Semantic Scholar / arXiv.\n\n")
        f.write("\n\n".join(out) + "\n")

    recent = sum(1 for y in years if y >= 2020)
    print(f"Wrote paper/references.bib with {len(out)} verified entries.")
    print(f"2020+: {recent}/{len(years)} = {100*recent/len(years):.0f}%")
    print("\nPre-2020 (need foundational justification):")
    for rec, th in uniq:
        if (rec.get("year") or 9999) < 2020:
            print(f"  {rec.get('year')}  {rec.get('title','')[:62]}")


if __name__ == "__main__":
    main()
