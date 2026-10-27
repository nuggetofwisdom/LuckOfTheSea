"""
collect_data.py - step 1, grab the raw data

Pulls known phishing URLs and known legit URLs and dumps each source into
data/raw/<source>_<date>.csv

columns: url, label, source, collected_at   (label 1 = phishing, 0 = legit)

the phishing feeds only list sites that are live right now, so run this once a
day for a couple weeks to build up more data. every run writes new dated files,
nothing gets overwritten.

run from the repo root:
    python -m scripts.collect_data
    python -m scripts.collect_data --only phishing
    python -m scripts.collect_data --only legit --legit-domains 3000
    python -m scripts.collect_data --only legit --legit-source commoncrawl   # the old slow way

legit collection saves progress as it goes. if you ctrl+c it, just run the
same command again and it picks up where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import random
import re
import sys
import time
import zipfile
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from phishdetect.features import registered_domain

RAW_DIR = Path("data/raw")
USER_AGENT = "LuckOfTheSea/0.1"
TODAY = date.today().isoformat()
NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def save(rows, source, label) -> Path | None:
    # keep only http/https urls, dedupe, sort
    cleaned = set()
    for r in rows:
        if not r:
            continue
        r = r.strip()
        if r.lower().startswith(("http://", "https://")):
            cleaned.add(r)
    rows = sorted(cleaned)

    if not rows:
        print(f"  [!] {source}: nothing to save")
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{source}_{TODAY}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label", "source", "collected_at"])
        for url in rows:
            writer.writerow([url, label, source, NOW])

    print(f"  [+] {source}: saved {len(rows):,} URLs -> {out}")
    return out


def get(url, timeout=60, retries=3):
    """GET with a few retries. Returns None if it never gets a 200."""
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            print(f"      {url} -> HTTP {r.status_code} (attempt {attempt}/{retries})")
            # no point retrying these
            if r.status_code in (401, 403, 404):
                return None
        except requests.RequestException as e:
            print(f"      {url} -> {type(e).__name__} (attempt {attempt}/{retries})")
        time.sleep(2 * attempt)
    return None


# --- phishing feeds (label 1) ---

def collect_openphish():
    # plain text, one url per line. try the main site first, then the github mirror
    print("OpenPhish...")
    mirrors = [
        "https://openphish.com/feed.txt",
        "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt",
    ]
    for url in mirrors:
        r = get(url)
        if r:
            save(r.text.splitlines(), "openphish", 1)
            return
    print("  [!] OpenPhish unavailable, skipping")


def collect_phishunt():
    # free json feed, no api key
    print("phishunt...")
    r = get("https://phishunt.io/feed.json")
    if not r:
        print("  [!] phishunt unavailable, skipping")
        return

    try:
        data = r.json()
        # sometimes it's a bare list, sometimes wrapped in {"data": [...]}
        if isinstance(data, list):
            items = data
        else:
            items = data.get("data", [])
        urls = [i["url"] for i in items if isinstance(i, dict) and i.get("url")]
        save(urls, "phishunt", 1)
    except (ValueError, KeyError, TypeError) as e:
        print(f"  [!] phishunt: couldn't parse feed ({e}), skipping")


def collect_phishtank():
    # PhishTank stopped letting people sign up years ago, so this is the keyless
    # download which gets rate limited a lot. fine if it fails, the other two are enough
    print("PhishTank (optional)...")
    r = get("http://data.phishtank.com/data/online-valid.csv", timeout=120, retries=1)
    if not r:
        print("  [!] PhishTank refused the request (common without an API key), skipping")
        return
    reader = csv.DictReader(io.StringIO(r.text))
    save([row["url"] for row in reader if row.get("url")], "phishtank", 1)


# --- legit urls (label 0) ---

def tranco_domains(top_n):
    print(f"Tranco top {top_n:,} domains...")
    r = get("https://tranco-list.eu/top-1m.csv.zip", timeout=180)
    if not r:
        sys.exit("  [x] Couldn't download the Tranco list. Check your internet connection.")

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = z.namelist()[0]
        lines = z.read(fname).decode().splitlines()

    # each line is "rank,domain"
    return [line.split(",", 1)[1].strip() for line in lines[:top_n] if "," in line]


def latest_commoncrawl_index():
    r = get("https://index.commoncrawl.org/collinfo.json")
    if not r:
        sys.exit("  [x] Couldn't reach the Common Crawl index.")
    return r.json()[0]["cdx-api"]  # newest crawl comes first


def commoncrawl_urls(api, domain, k):
    # Pull real page urls (with paths) for a legit domain. If every legit
    # example was just https://site.com/ the model would figure out
    # "has a path = phishing" and call it a day, which isn't what we want.
    params = {
        "url": f"{domain}/*",
        "output": "json",
        "limit": str(k * 10),
        "filter": ["status:200", "mime:text/html"],
        "fl": "url",
    }
    for attempt in range(3):
        try:
            r = session.get(api, params=params, timeout=60)
            if r.status_code == 404:
                return []  # domain isn't in this crawl
            if r.status_code == 200:
                urls = []
                for line in r.text.splitlines():
                    if line.strip():
                        urls.append(json.loads(line)["url"])
                # drop super long ones and dupes (keeps order)
                urls = list(dict.fromkeys(u for u in urls if len(u) < 500))
                random.shuffle(urls)
                return urls[:k]
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3 * (attempt + 1))  # CC gets overloaded, back off a bit
    return []


# common crawl was way too slow for 2000 domains, so sitemaps are the default now.
# these regexes pull <loc> entries out of sitemap xml and skip stuff that isn't a page
LOC_RE = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</loc>", re.IGNORECASE | re.DOTALL)
NOT_A_PAGE = re.compile(
    r"\.(jpe?g|png|gif|webp|svg|ico|pdf|mp4|webm|mp3|zip|gz|xml|txt|css|js|json|woff2?)(\?|$)", re.IGNORECASE)


def fetch_text(url, max_bytes=3_000_000, timeout=10):
    # download a text file but bail after max_bytes. some sitemaps are like 50MB
    # and we only need a handful of urls out of them
    try:
        with session.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None
            buf = bytearray()
            for chunk in r.iter_content(65_536):
                buf += chunk
                if len(buf) >= max_bytes:
                    break
    except requests.RequestException:
        return None

    data = bytes(buf)
    # it's a .xml.gz sitemap, decompress whatever we got
    if data[:2] == b"\x1f\x8b":
        try:
            data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data, max_bytes * 5)
        except zlib.error:
            return None
    return data.decode("utf-8", errors="replace")


def sitemap_urls(domain, k):
    # most sites publish a sitemap.xml so search engines can find their pages,
    # and robots.txt usually says where it is. each site answers in under a
    # second and only gets 2-6 requests from us, so we can hit a lot at once
    candidates = []
    robots = fetch_text(f"https://{domain}/robots.txt", max_bytes=300_000)
    if robots:
        candidates += re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots)
    candidates += [f"https://{domain}/sitemap.xml", f"https://www.{domain}/sitemap.xml"]

    queue = deque(candidates)
    seen = set()
    pages = []
    fetched = 0
    while queue and fetched < 6 and len(pages) < k * 20:
        sitemap = queue.popleft()
        if sitemap in seen:
            continue
        seen.add(sitemap)
        fetched += 1

        text = fetch_text(sitemap)
        if not text:
            continue

        children = []
        for loc in LOC_RE.findall(text):
            loc = html.unescape(loc).strip()
            if not loc.lower().startswith(("http://", "https://")) or len(loc) > 500:
                continue
            if "sitemap" in loc.lower() or loc.lower().endswith((".xml", ".xml.gz")):
                # sitemap index, it just points at more sitemaps
                children.append(loc)
            elif not NOT_A_PAGE.search(loc) and registered_domain(loc) == registered_domain(domain):
                pages.append(loc)

        # only dig into a few of the child sitemaps
        random.shuffle(children)
        queue.extendleft(children[:3])

    pages = list(dict.fromkeys(pages))
    random.shuffle(pages)
    return pages[:k]


def collect_legit(source, n_domains, per_domain, top_n, workers=None):
    random.seed(42)
    domains = tranco_domains(top_n)
    sample = random.sample(domains, min(n_domains, len(domains)))

    if source == "commoncrawl":
        api = latest_commoncrawl_index()

        def fetch(d):
            return commoncrawl_urls(api, d, per_domain)

        workers = workers or 4  # CC index is shared and falls over easily
        out_name = "tranco_commoncrawl"
    else:
        def fetch(d):
            return sitemap_urls(d, per_domain)

        workers = workers or 32  # fine here since every request hits a different site
        out_name = "tranco_sitemaps"

    # resume stuff: every finished domain gets written to a progress file right
    # away, so if this crashes or gets ctrl+c'd the next run skips what's done
    progress_file = RAW_DIR / f".progress_{out_name}.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    if progress_file.exists():
        for line in progress_file.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                results[rec["domain"]] = rec["urls"]
            except (ValueError, KeyError):
                pass  # half-written line from a crash, just ignore it

    todo = [d for d in sample if d not in results]
    print(f"{source}: up to {per_domain} URLs each for {len(sample):,} domains "
          f"({len(results):,} already done, {len(todo):,} to go, {workers} at a time)")

    started = time.time()
    done = 0
    hits = 0
    with progress_file.open("a", encoding="utf-8") as prog, ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch, d): d for d in todo}
        try:
            for job in as_completed(jobs):
                domain = jobs[job]
                try:
                    found = job.result()
                except Exception:
                    found = []  # one weird site shouldn't kill the whole run
                results[domain] = found
                if found:
                    hits += 1
                prog.write(json.dumps({"domain": domain, "urls": found}) + "\n")
                prog.flush()

                done += 1
                if done % 100 == 0 or done == len(todo):
                    rate = done / max(time.time() - started, 1e-9)
                    eta = (len(todo) - done) / max(rate, 1e-9)
                    print(f"      {done:,}/{len(todo):,} domains | {hits:,} with deep links | "
                          f"~{eta / 60:.1f} min left")
        except KeyboardInterrupt:
            print("\n  stopping (letting the in-flight requests finish)... progress is saved, "
                  "run the same command again to resume.")
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)

    urls = []
    with_paths = 0
    for domain in sample:
        found = results.get(domain) or []
        if found:
            with_paths += 1
            urls.extend(found)
        else:
            # nothing came back, use the homepage so the domain still shows up
            urls.append(f"https://{domain}/")
    print(f"  {with_paths:,}/{len(sample):,} domains gave real page URLs, the rest fell back to homepages")

    # only clear the progress file once the csv is actually written
    if save(urls, out_name, 0):
        progress_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["phishing", "legit"], help="collect only one class")
    parser.add_argument("--legit-domains", type=int, default=2000,
                        help="legit domains to sample (default 2000)")
    parser.add_argument("--urls-per-domain", type=int, default=5,
                        help="URLs per legit domain (default 5)")
    parser.add_argument("--tranco-top", type=int, default=100_000,
                        help="sample legit domains from the top N of Tranco (default 100k)")
    parser.add_argument("--legit-source", choices=["sitemaps", "commoncrawl"], default="sitemaps",
                        help="where to get real page URLs for legit domains (default sitemaps, way faster)")
    parser.add_argument("--workers", type=int,
                        help="parallel requests (default 32 for sitemaps, 4 for commoncrawl)")
    args = parser.parse_args()

    if args.only is None or args.only == "phishing":
        collect_openphish()
        collect_phishunt()
        collect_phishtank()

    if args.only is None or args.only == "legit":
        collect_legit(args.legit_source, args.legit_domains, args.urls_per_domain,
                      args.tranco_top, args.workers)


if __name__ == "__main__":
    main()
