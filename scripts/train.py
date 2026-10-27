"""
train.py -  train a few models, test them fairly, keep the best one

what it does:
  1. loads data/processed/dataset.csv and turns every url into features
  2. splits it into a TRAIN set (what the model learns from) and a TEST set
     (the model doesn't see these until the very end, like a final exam)
  3. trains four models, from dumbest to smartest
  4. scores each one on the test set
  5. saves the best one to models/model.joblib and dumps charts, a results
     table, and a list of every mistake into reports/

split options (--split):
  domain  (default) no website shows up in both train and test. otherwise the
          model can just memorize domain names and look way better than it is
  time    train on urls collected before a cutoff date, test on ones collected
          after. most realistic test there is ("can last week's model catch
          this week's phishing?") but you need ~2+ weeks of daily collection

run from the repo root:
    python -m scripts.train
    python -m scripts.train --split time --cutoff 2026-11-10
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts straight to files, don't try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from phishdetect.features import FEATURE_NAMES, features_dataframe
from phishdetect.model_io import DEFAULT_MODEL_PATH, save_model

DATASET = Path("data/processed/dataset.csv")
REPORTS = Path("reports")
SEED = 42


def build_models():
    return {
        # always guesses whichever class is more common. if a real model
        # can't beat this, something's broken
        "Baseline (always guess majority)": DummyClassifier(strategy="most_frequent"),

        # draws one straight line between phishing and legit. simple, and easy
        # to see which features it cares about. needs scaled features to work well
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),

        # a few hundred decision trees that each vote
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=SEED,
        ),

        # trees built one at a time, each one trying to fix what the last one got wrong
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def split(df, how, cutoff):
    if how == "time":
        if not cutoff:
            raise SystemExit("--split time needs --cutoff YYYY-MM-DD")
        dates = pd.to_datetime(df["collected_at"], utc=True)
        cut = pd.Timestamp(cutoff, tz="UTC")

        # legit urls mostly get collected once in one big run, so splitting
        # them by date doesn't really work. split those by domain instead and
        # only split the phishing ones by time
        legit = df[df.label == 0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        legit_train_idx, legit_test_idx = next(gss.split(legit, groups=legit["reg_domain"]))

        phish = df[df.label == 1]
        phish_dates = dates[df.label == 1]

        train = pd.concat([legit.iloc[legit_train_idx], phish[phish_dates < cut]])
        test = pd.concat([legit.iloc[legit_test_idx], phish[phish_dates >= cut]])

        # if a phishing domain was already in training, it's not really a new
        # site for the test, so pull it out
        seen_before = (test.label == 1) & test.reg_domain.isin(train.reg_domain)
        test = test[~seen_before]
    else:
        # group by domain so every url from one site lands on the same side
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(gss.split(df, groups=df["reg_domain"]))
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

    if train.label.nunique() < 2 or test.label.nunique() < 2:
        raise SystemExit("train or test set is missing a class. collect more data or change --cutoff")

    return train.reset_index(drop=True), test.reset_index(drop=True)


def score(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # roc auc blows up if the test set only has one class in it
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / max(fp + tn, 1),
        "roc_auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def recall_at_fpr(y_true, y_prob, max_fpr=0.01):
    # how much phishing can we catch if we're only allowed to wrongly flag
    # 1% of legit urls? way more useful number than plain accuracy for this
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    allowed = fpr <= max_fpr
    i = np.argmax(tpr * allowed)
    return float(tpr[i]), float(min(thresholds[i], 1.0))


def feature_importance(model):
    # if it's a pipeline, the actual model is the last step
    if hasattr(model, "steps"):
        est = model.steps[-1][1]
    else:
        est = model

    if hasattr(est, "feature_importances_"):  # tree models
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):  # logistic regression
        vals = np.abs(est.coef_[0])
    else:
        return None  # the baseline doesn't have any
    return pd.Series(vals, index=FEATURE_NAMES).sort_values(ascending=False)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["domain", "time"], default="domain",
                        help="how to split train/test (default domain)")
    parser.add_argument("--cutoff", help="YYYY-MM-DD, only for --split time")
    args = parser.parse_args()

    if not DATASET.exists():
        raise SystemExit("no dataset yet, run: python -m scripts.build_dataset")

    df = pd.read_csv(DATASET, dtype={"url": str})
    train, test = split(df, args.split, args.cutoff)
    print(f"Train: {len(train):,} URLs ({train.label.mean():.0%} phishing) | "
          f"Test: {len(test):,} URLs ({test.label.mean():.0%} phishing)")

    print("Extracting features...")
    X_train = features_dataframe(train.url)
    X_test = features_dataframe(test.url)
    y_train = train.label.values
    y_test = test.label.values

    REPORTS.mkdir(exist_ok=True)
    results = {}
    probs = {}
    fitted = {}
    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]  # chance each url is phishing

        results[name] = score(y_test, prob)
        results[name]["recall_at_1pct_fpr"], _ = recall_at_fpr(y_test, prob)
        probs[name] = prob
        fitted[name] = model

    # --- results table (this is the one that goes in the README) ---
    cols = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc", "recall_at_1pct_fpr"]
    table = pd.DataFrame(results).T[cols]
    print("\n" + table.round(3).to_string())

    md = [
        "| Model | Accuracy | Precision | Recall | F1 | False positive rate | ROC AUC | Recall @ 1% FPR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in table.iterrows():
        cells = " | ".join(f"{row[c]:.3f}" for c in cols)
        md.append(f"| {name} | {cells} |")

    header = f"Split: `{args.split}` | Train: {len(train):,} URLs | Test: {len(test):,} URLs\n\n"
    (REPORTS / "results.md").write_text(header + "\n".join(md) + "\n")

    # pick the winner by roc auc. the baseline obviously isn't allowed to win
    real_models = [n for n in results if not n.startswith("Baseline")]
    best = max(real_models, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model by ROC AUC: {best}")

    # --- charts ---

    # roc curves for every real model on one plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob in probs.items():
        if name.startswith("Baseline"):
            continue  # it's just a flat diagonal line, not worth drawing
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guessing")
    ax.set(xlabel="False positive rate (legit URLs wrongly flagged)",
           ylabel="True positive rate (phishing caught)",
           title="ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "roc_curves.png")

    # confusion matrix for the best model
    fig, ax = plt.subplots(figsize=(5, 4.5))
    best_preds = (probs[best] >= 0.5).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_preds,
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix: {best}")
    save_fig(fig, "confusion_matrix.png")

    # which features the best model leaned on the most
    imp = feature_importance(fitted[best])
    if imp is not None:
        top = imp.head(15)[::-1]  # flip it so the biggest bar ends up on top
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(top.index, top.values)
        ax.set(title=f"Top 15 features: {best}", xlabel="Importance")
        save_fig(fig, "feature_importance.png")
        imp.to_csv(REPORTS / "feature_importance.csv", header=["importance"])

    # --- every url the best model got wrong ---
    # honestly the most useful file this spits out. reading through these is
    # how you figure out what features to add next
    errors = test.assign(phishing_probability=probs[best])
    errors["predicted"] = (errors.phishing_probability >= 0.5).astype(int)
    errors = errors[errors.predicted != errors.label]
    errors["error_type"] = np.where(errors.label == 1, "missed phishing (FN)", "false alarm (FP)")
    errors = errors.sort_values("phishing_probability", ascending=False)
    errors[["error_type", "url", "phishing_probability", "source"]].to_csv(
        REPORTS / "errors.csv", index=False)
    print(f"Wrote {len(errors):,} mistakes to reports/errors.csv, go read through them")

    # --- save the model and its numbers ---
    meta = {
        "model_name": best,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "n_train": len(train),
        "n_test": len(test),
        "test_metrics": results[best],
    }
    save_model(fitted[best], DEFAULT_MODEL_PATH, meta)

    # default=float handles the numpy number types json can't deal with on its own
    metrics = {"best": meta, "all_models": results}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved {best} -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
