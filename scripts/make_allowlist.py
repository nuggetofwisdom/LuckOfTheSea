"""
make_allowlist.py - download the Tranco top sites and save them as the allowlist

writes seeds/top_domains.txt, minus link shorteners and sites where anyone can
host a page (see SHARED_HOSTS in phishdetect/allowlist.py). predict.py skips
the model for anything on the list.

run from the repo root:
    python -m scripts.make_allowlist
    python -m scripts.make_allowlist --top 5000
"""
from __future__ import annotations

import argparse

from phishdetect.allowlist import DEFAULT_ALLOWLIST_PATH, is_shared_host
from phishdetect.features import URL_SHORTENERS
from scripts.collect_data import tranco_domains


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=10_000,
                        help="how many of the top sites to trust (default 10,000). smaller = safer")
    args = parser.parse_args()

    domains = []
    skipped = 0
    for d in tranco_domains(args.top):
        d = d.lower()
        if d in URL_SHORTENERS or is_shared_host(d):
            skipped += 1
            continue
        domains.append(d)

    header = (f"# Tranco top {args.top:,} sites, made by scripts/make_allowlist.py\n"
              "# shorteners and shared hosting sites taken out\n")
    DEFAULT_ALLOWLIST_PATH.write_text(header + "\n".join(domains) + "\n", encoding="utf-8")
    print(f"Saved {len(domains):,} domains -> {DEFAULT_ALLOWLIST_PATH} (skipped {skipped})")


if __name__ == "__main__":
    main()
