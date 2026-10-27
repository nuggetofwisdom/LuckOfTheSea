"""
build_dataset.py - step 2, clean up the raw data and mash it into one file

Reads every csv in data/raw/ and then:
  1. drops duplicate urls (the feeds overlap a lot and repeat day to day)
  2. drops urls that show up as both phishing AND legit, can't trust those labels
  3. caps how many urls one domain can add, so one big phishing campaign with
     500 nearly identical urls doesn't take over the whole dataset
  4. balances phishing/legit 50/50 (unless you pass --no-balance)

then writes everything to data/processed/dataset.csv

run from the repo root:
    python -m scripts.build_dataset
    python -m scripts.build_dataset --max-per-domain 10 --no-balance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phishdetect.features import registered_domain

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/dataset.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-per-domain", type=int, default=20,
                        help="max urls per domain per label (default 20)")
    parser.add_argument("--no-balance", action="store_true",
                        help="keep the natural phishing/legit ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for shuffling (default 42)")
    args = parser.parse_args()

    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("nothing in data/raw/ yet, run: python -m scripts.collect_data")

    frames = []
    for f in files:
        frames.append(pd.read_csv(f, dtype={"url": str}))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} files")

    df["url"] = df["url"].str.strip()
    df = df.dropna(subset=["url"])

    # sort by time first so we keep the FIRST time we saw each url.
    # matters later if I split train/test by date
    df = df.sort_values("collected_at")
    df = df.drop_duplicates(subset=["url", "label"], keep="first")

    # if the same url got labeled phishing in one feed and legit in another,
    # just toss it. no way to know which one's right
    labels_per_url = df.groupby("url")["label"].nunique()
    conflicted = labels_per_url[labels_per_url > 1].index
    if len(conflicted) > 0:
        print(f"Dropping {len(conflicted):,} URLs that showed up with both labels")
        df = df[~df["url"].isin(conflicted)]

    df["reg_domain"] = df["url"].map(registered_domain)

    # shuffle before the cap, otherwise .head() would always grab the same
    # (oldest) urls from each domain instead of a random handful
    df = df.sample(frac=1, random_state=args.seed)

    before = len(df)
    df = df.groupby(["reg_domain", "label"]).head(args.max_per_domain)
    print(f"Per-domain cap of {args.max_per_domain}: {before:,} -> {len(df):,} rows")

    counts = df["label"].value_counts()
    n_phish = counts.get(1, 0)
    n_legit = counts.get(0, 0)
    print(f"Class counts: phishing={n_phish:,}  legit={n_legit:,}")

    # cut the bigger class down to the size of the smaller one.
    # skip it if one class is missing entirely, nothing to balance against
    if not args.no_balance and counts.size == 2:
        n = counts.min()
        df = df.groupby("label").head(n)
        print(f"Balanced to {n:,} of each class")

    # one last shuffle so phishing and legit rows are mixed together in the file
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
