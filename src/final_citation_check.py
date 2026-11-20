"""Final pre-submission pass: confirm every reference actually cited in the
manuscript still resolves, and report the recency split.

This re-hits the network rather than trusting the earlier verification logs, so
a link that has rotted since Phase 1 is caught before submission.
"""
import re, json, sys, time, urllib.request, urllib.error, urllib.parse

UA = {"User-Agent": "AKI-QML-final-check/1.0 (mailto:research@example.org)"}
SRC = ("paper/manuscript.tex", "paper/sections/results.tex",
       "paper/sections/discussion.tex", "paper/sections/limitations.tex")


def cited_keys():
    keys = set()
    for f in SRC:
        for m in re.finditer(r"\\cite\{([^}]*)\}", open(f).read()):
            keys |= {k.strip() for k in m.group(1).split(",")}
    return keys


def parse_bib(path="paper/references.bib"):
    out = {}
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", open(path).read(), re.S):
        d = {}
        for field in ("doi", "eprint", "year", "title", "journal", "author"):
            m = re.search(field + r"\s*=\s*\{(.*?)\},", body, re.S)
            if m:
                d[field] = " ".join(m.group(1).split())
        out[key] = d
    return out


def crossref_alive(doi):
    """Publishers commonly return 403 to non-browser clients; the Crossref API is
    the authoritative check that a DOI is registered and resolvable."""
    try:
        req = urllib.request.Request(
            "https://api.crossref.org/works/" + urllib.parse.quote(doi), headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode())["message"].get("DOI") is not None
    except Exception:
        return False


def resolves(url):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status < 400, r.status
    except urllib.error.HTTPError as e:
        if e.code in (403, 405):                       # publisher blocks HEAD
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=25) as r:
                    return r.status < 400, r.status
            except Exception as e2:
                return False, str(e2)[:40]
        return False, e.code
    except Exception as e:
        return False, str(e)[:40]


def main():
    keys, bib = cited_keys(), parse_bib()
    ok, bad, years = 0, [], []
    for k in sorted(keys):
        e = bib.get(k)
        if not e:
            bad.append((k, "NOT IN BIB"))
            continue
        if e.get("year"):
            years.append(int(e["year"]))
        url = (f"https://doi.org/{e['doi']}" if e.get("doi")
               else f"https://arxiv.org/abs/{e['eprint']}" if e.get("eprint") else None)
        if url is None:
            bad.append((k, "no DOI or arXiv id"))
            continue
        good, status = resolves(url)
        if not good and e.get("doi") and crossref_alive(e["doi"]):
            good, status = True, "403 at publisher, DOI live in Crossref"
        if good:
            ok += 1
            note = f"  ({status})" if isinstance(status, str) else ""
            print(f"[OK  ] {k:28s} {url}{note}", flush=True)
        else:
            bad.append((k, f"{url} -> {status}"))
            print(f"[FAIL] {k:28s} {url} -> {status}", flush=True)
        time.sleep(0.4)

    recent = sum(1 for y in years if y >= 2020)
    print(f"\n=== cited: {len(keys)} | resolved: {ok} | failures: {len(bad)} ===")
    print(f"=== 2020+: {recent}/{len(years)} = {100*recent/max(len(years),1):.0f}% "
          f"(requirement >=70%) ===")
    if bad:
        print("\nFAILURES:")
        for k, why in bad:
            print(f"  {k}: {why}")
    return len(bad)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
