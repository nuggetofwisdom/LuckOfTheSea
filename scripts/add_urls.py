"""
add_urls.py - add my own list of urls to the raw data

the feeds only give me one kind of legit url (normal pages on big sites) so the
model never sees a real login page and decides "login" = phishing. this lets me
add urls from anywhere. text file, one url per line, blank lines and lines
starting with # get skipped

saves to data/raw/<source>_<date>.csv same as collect_data.py, so
build_dataset.py picks it up on its own

run from the repo root:
    python -m scripts.add_urls seeds/legit_logins.txt --label legit --source legit_logins
    python -m scripts.add_urls inbox_links.txt --label legit --source inbox
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.collect_data import save


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="text file with one url per line")
    parser.add_argument("--label", choices=["legit", "phishing"], required=True)
    parser.add_argument("--source", required=True,
                        help="short name for where these came from, goes in the source column")
    args = parser.parse_args()

    urls = []
    for line in args.file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)

    label = 1 if args.label == "phishing" else 0
    save(urls, args.source, label)


if __name__ == "__main__":
    main()
