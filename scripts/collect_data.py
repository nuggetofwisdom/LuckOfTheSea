"""
collect_data.py - step 1, grab the raw data

Pulls known phishing URLs and known legit URLs and dumps each source into
data/raw/<source>_<date>.csv

columns: url, label, source, collected_at   (label 1 = phishing, 0 = legit)

run from the repo root:
    python -m scripts.collect_data
    python -m scripts.collect_data --only phishing
    python -m scripts.collect_data --only legit --legit-domains 3000
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import requests

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


def collect_legit(n_domains, per_domain, top_n):
    random.seed(42)
    domains = tranco_domains(top_n)
    sample = random.sample(domains, min(n_domains, len(domains)))
    api = latest_commoncrawl_index()

    print(f"Common Crawl ({api.split('/')[-1]}): fetching up to {per_domain} URLs "
          f"each for {len(sample):,} domains. This takes a while...")

    urls = []
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(commoncrawl_urls, api, d, per_domain): d for d in sample}
        for job in as_completed(jobs):
            domain = jobs[job]
            found = job.result()
            if found:
                urls.extend(found)
            else:
                # nothing came back, use the homepage so the domain still shows up
                urls.append(f"https://{domain}/")
            done += 1
            if done % 100 == 0:
                print(f"      {done:,}/{len(sample):,} domains, {len(urls):,} URLs so far")

    save(urls, "tranco_commoncrawl", 0)


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
    args = parser.parse_args()

    if args.only is None or args.only == "phishing":
        collect_openphish()
        collect_phishunt()
        collect_phishtank()

    if args.only is None or args.only == "legit":
        collect_legit(args.legit_domains, args.urls_per_domain, args.tranco_top)


if __name__ == "__main__":
    main()
