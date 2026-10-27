"""
train.py -  train a few models, test them fairly, keep the best one

what it does:
  1. loads data/processed/dataset.csv and turns every url into features
  2. splits it 3 ways:
       TRAIN       what the models learn from
       VALIDATION  for picking the best model and its threshold
       TEST        not touched until the very end, like a final exam. it
                   wasn't used to pick anything so its score is honest
  3. trains six models, from dumbest to smartest
  4. picks a threshold for each model on validation (the cutoff where about
     1% of legit urls get flagged) and picks the best model
  5. scores every model on the test set once
  6. saves the best one to models/model.joblib and dumps charts, a results
     table, and a list of every mistake into reports/

split options (--split):
  domain  (default) no website shows up in more than one set. otherwise the
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from phishdetect.features import FEATURE_NAMES, model_input, registered_domain
from phishdetect.model_io import DEFAULT_MODEL_PATH, save_model

DATASET = Path("data/processed/dataset.csv")
REPORTS = Path("reports")
SEED = 42
MAX_FPR = 0.01  # flagging 1 in 100 legit urls is about as many false alarms as anyone would put up with


def only_features():
    # every model gets the 34 features + the raw url. this only passes the
    # features through, for the models that need numbers
    return ColumnTransformer([("features", "passthrough", FEATURE_NAMES)],
                             verbose_feature_names_out=False)


def char_ngrams():
    # cuts each url into overlapping 3-5 letter chunks ("pay", "payp",
    # "paypa", "-ver", "ify.") and counts them. the model figures out which
    # chunks mean phishing by itself, none of my features involved
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, lowercase=True)


def build_models():
    return {
        # always guesses whichever class is more common. if a real model
        # can't beat this, something's broken
        "Baseline (always guess majority)": DummyClassifier(strategy="most_frequent"),

        # draws one straight line between phishing and legit. simple, and easy
        # to see which features it cares about. needs scaled features to work well
        "Logistic Regression": make_pipeline(
            only_features(),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),

        # a few hundred decision trees that each vote
        "Random Forest": make_pipeline(
            only_features(),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=SEED),
        ),

        # trees built one at a time, each one trying to fix what the last one got wrong
        "XGBoost": make_pipeline(
            only_features(),
            XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                          subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                          n_jobs=-1, random_state=SEED),
        ),

        # ignores my 34 features completely and only reads the url text
        "Char n-grams": make_pipeline(
            ColumnTransformer([("chars", char_ngrams(), "url")], verbose_feature_names_out=False),
            LogisticRegression(max_iter=3000, C=4.0, random_state=SEED),
        ),

        # both: the url chunks and my 34 features together
        "Char n-grams + features": make_pipeline(
            ColumnTransformer([
                ("chars", char_ngrams(), "url"),
                ("features", StandardScaler(), FEATURE_NAMES),
            ], verbose_feature_names_out=False),
            LogisticRegression(max_iter=3000, C=4.0, random_state=SEED),
        ),
    }


def split_by_domain(df, test_size):
    # every url from one site lands on the same side
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    a, b = next(gss.split(df, groups=df["reg_domain"]))
    return df.iloc[a], df.iloc[b]


def split(df, how, cutoff):
    if how == "time":
        if not cutoff:
            raise SystemExit("--split time needs --cutoff YYYY-MM-DD")
        dates = pd.to_datetime(df["collected_at"], utc=True)
        cut = pd.Timestamp(cutoff, tz="UTC")

        # legit urls mostly get collected once in one big run, so splitting
        # them by date doesn't really work. split those by domain instead and
        # only split the phishing ones by time
        legit_train, legit_test = split_by_domain(df[df.label == 0], 0.2)
        phish = df[df.label == 1]
        phish_dates = dates[df.label == 1]

        train = pd.concat([legit_train, phish[phish_dates < cut]])
        test = pd.concat([legit_test, phish[phish_dates >= cut]])

        # if a phishing domain was already in training, it's not really a new
        # site for the test, so pull it out
        seen_before = (test.label == 1) & test.reg_domain.isin(train.reg_domain)
        test = test[~seen_before]
    else:
        train, test = split_by_domain(df, 0.2)

    # take validation out of train. 25% of the 80% = 20% overall
    train, val = split_by_domain(train, 0.25)

    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if part.label.nunique() < 2:
            raise SystemExit(f"{name} set is missing a class. collect more data or change --cutoff")

    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


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


def recall_at_fpr(y_true, y_prob, max_fpr=MAX_FPR):
    # how much phishing can we catch if we're only allowed to wrongly flag
    # 1% of legit urls? way more useful number than plain accuracy for this.
    # also gives back the threshold that does it
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    thresholds = np.minimum(thresholds, 1.0)  # the first one is infinity
    allowed = np.flatnonzero(fpr <= max_fpr)
    best_recall = tpr[allowed].max()

    # usually only one threshold gets the best recall, but if the data splits
    # really cleanly there's a whole range of them. the top of that range barely
    # flags anything and the bottom flags borderline legit stuff, so use the middle
    best = allowed[tpr[allowed] == best_recall]
    threshold = (thresholds[best[0]] + thresholds[best[-1]]) / 2
    return float(best_recall), float(threshold)


def feature_importance(model):
    # if it's a pipeline, the actual model is the last step and the names of
    # what went into it come from the first step
    if not hasattr(model, "steps"):
        return None  # the baseline doesn't have any
    est = model.steps[-1][1]
    names = model.steps[0][1].get_feature_names_out()

    if hasattr(est, "feature_importances_"):  # tree models
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):  # logistic regression, bigger weight = matters more
        vals = np.abs(est.coef_[0])
    else:
        return None
    return pd.Series(vals, index=names).sort_values(ascending=False)


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
    # redo this instead of using the csv column, in case I change registered_domain()
    df["reg_domain"] = df["url"].map(registered_domain)
    train, val, test = split(df, args.split, args.cutoff)
    for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"{name}: {len(part):,} URLs ({part.label.mean():.0%} phishing)")

    print("Extracting features...")
    X_train = model_input(train.url)
    X_val = model_input(val.url)
    X_test = model_input(test.url)
    y_train = train.label.values
    y_val = val.label.values
    y_test = test.label.values

    REPORTS.mkdir(exist_ok=True)
    val_results = {}
    results = {}
    probs = {}
    fitted = {}
    thresholds = {}
    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # everything gets decided on validation
        val_prob = model.predict_proba(X_val)[:, 1]
        val_recall, threshold = recall_at_fpr(y_val, val_prob)
        if val_recall == 0:
            threshold = 0.5  # can't get under 1% fpr at all (the baseline), just use 0.5
        val_results[name] = {"roc_auc": score(y_val, val_prob)["roc_auc"],
                             "recall_at_1pct_fpr": val_recall}
        thresholds[name] = threshold

        # then score test once, using the threshold from validation
        prob = model.predict_proba(X_test)[:, 1]  # chance each url is phishing
        results[name] = score(y_test, prob, threshold)
        results[name]["recall_at_1pct_fpr"], _ = recall_at_fpr(y_test, prob)
        results[name]["threshold"] = threshold
        probs[name] = prob
        fitted[name] = model

    # pick the winner on VALIDATION, whoever catches the most phishing at 1%
    # false alarms. roc auc if it's a tie. the baseline obviously isn't allowed to win
    real_models = [n for n in results if not n.startswith("Baseline")]
    best = max(real_models, key=lambda n: (val_results[n]["recall_at_1pct_fpr"], val_results[n]["roc_auc"]))
    best_threshold = thresholds[best]

    # --- results table (this is the one that goes in the README) ---
    cols = ["accuracy", "precision", "recall", "f1", "false_positive_rate", "roc_auc", "recall_at_1pct_fpr"]
    table = pd.DataFrame(results).T[cols]
    table.insert(0, "val_recall_at_1pct_fpr", [val_results[n]["recall_at_1pct_fpr"] for n in table.index])
    print("\nTest set, each model at the threshold picked on validation:")
    print(table.round(3).to_string())

    md = [
        "| Model | Val recall @ 1% FPR | Accuracy | Precision | Recall | F1 | False positive rate | ROC AUC | Recall @ 1% FPR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in table.iterrows():
        cells = " | ".join(f"{row[c]:.3f}" for c in table.columns)
        label = f"**{name}** (selected)" if name == best else name
        md.append(f"| {label} | {cells} |")

    header = (f"Split: `{args.split}` | Train: {len(train):,} | Validation: {len(val):,} | "
              f"Test: {len(test):,} URLs\n\n"
              "Model and thresholds picked on validation, test scores use each model's threshold.\n\n")
    (REPORTS / "results.md").write_text(header + "\n".join(md) + "\n")
    print(f"\nBest model on validation: {best} (threshold {best_threshold:.3f})")

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
           title="ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "roc_curves.png")

    # confusion matrix for the best model
    fig, ax = plt.subplots(figsize=(5, 4.5))
    best_preds = (probs[best] >= best_threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_preds,
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix: {best}")
    save_fig(fig, "confusion_matrix.png")

    # which inputs the best model leaned on the most. for the char models these
    # are url chunks instead of feature names (repr so you can see spaces/dots)
    imp = feature_importance(fitted[best])
    if imp is not None:
        top = imp.head(15)[::-1]  # flip it so the biggest bar ends up on top
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh([repr(n) if n not in FEATURE_NAMES else n for n in top.index], top.values)
        ax.set(title=f"Top 15 inputs: {best}", xlabel="Importance")
        save_fig(fig, "feature_importance.png")
        imp.head(200).to_csv(REPORTS / "feature_importance.csv", header=["importance"])

    # --- every url the best model got wrong ---
    # honestly the most useful file this spits out. reading through these is
    # how you figure out what features to add next
    errors = test.assign(phishing_probability=probs[best])
    errors["predicted"] = (errors.phishing_probability >= best_threshold).astype(int)
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
        "threshold": best_threshold,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "val_metrics": val_results[best],
        "test_metrics": results[best],
    }
    save_model(fitted[best], DEFAULT_MODEL_PATH, meta)

    # default=float handles the numpy number types json can't deal with on its own
    metrics = {"best": meta, "all_models": results, "validation": val_results}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"Saved {best} -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
