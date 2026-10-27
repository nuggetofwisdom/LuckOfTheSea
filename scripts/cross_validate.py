"""
cross_validate.py - check if one model is actually better or just got lucky

train.py only does one split, so a 0.01 gap between two models could just be
which urls happened to land in test. this splits the data 5 different ways
(by domain, same as train.py), trains every model on each, and prints the
average and how much it moves around. if two models are within each other's
+/- they're basically tied.

uses the default settings from build_models() (no tuning), so the numbers will
be a little lower than train.py. it's for comparing models, not for the README
headline number.

run from the repo root:
    python -m scripts.cross_validate
    python -m scripts.cross_validate --folds 10 --models XGBoost "Char n-grams"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GroupKFold

from phishdetect.features import model_input, registered_domain
from scripts.train import DATASET, build_models, low_fpr_recall, recall_at_fpr

AVERAGED = "XGBoost + Char n-grams (averaged)"


def all_models():
    models = build_models()
    del models["Baseline (always guess majority)"]
    # same as the averaged one in train.py, just with default settings
    extra = build_models()
    models[AVERAGED] = VotingClassifier([
        ("xgboost", extra["XGBoost"]),
        ("chars", extra["Char n-grams"]),
    ], voting="soft")
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5, help="how many splits (default 5)")
    parser.add_argument("--models", nargs="+", help="only these models (default all of them)")
    args = parser.parse_args()

    if not DATASET.exists():
        raise SystemExit("no dataset yet, run: python -m scripts.build_dataset")

    df = pd.read_csv(DATASET, dtype={"url": str})
    groups = df["url"].map(registered_domain)
    print(f"Extracting features for {len(df):,} URLs...")
    X = model_input(df.url)
    y = df.label.values

    names = args.models or list(all_models())
    scores = {name: {"low_fpr": [], "at_1pct": []} for name in names}

    folds = GroupKFold(n_splits=args.folds)
    for i, (train_idx, test_idx) in enumerate(folds.split(X, y, groups), start=1):
        print(f"Fold {i}/{args.folds}...")
        models = all_models()  # fresh copies every fold
        for name in names:
            model = models[name]
            model.fit(X.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            scores[name]["low_fpr"].append(low_fpr_recall(y[test_idx], prob))
            scores[name]["at_1pct"].append(recall_at_fpr(y[test_idx], prob)[0])

    print(f"\n{'Model':<36}{'avg recall @ 0-5% FPR':>24}{'recall @ 1% FPR':>20}")
    rows = sorted(names, key=lambda n: np.mean(scores[n]["low_fpr"]), reverse=True)
    for name in rows:
        low = np.array(scores[name]["low_fpr"])
        one = np.array(scores[name]["at_1pct"])
        print(f"{name:<36}{low.mean():>16.3f} ± {low.std():.3f}{one.mean():>12.3f} ± {one.std():.3f}")


if __name__ == "__main__":
    main()
