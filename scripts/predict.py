"""
predict.py - throw some urls at the trained model and see what it thinks

prints a verdict and the phishing probability for each url, plus any red
flags it spotted (sketchy tld, ip address instead of a domain, stuff like that)

run from the repo root:
    python -m scripts.predict "http://paypa1-secure-login.xyz/verify.php" "https://github.com/JacobMo28"
    python -m scripts.predict --threshold 0.8 "http://some-sketchy-site.top/login"
"""
from __future__ import annotations

import argparse

from phishdetect.features import red_flags
from phishdetect.model_io import DEFAULT_MODEL_PATH, load_model, predict_proba


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="+", help="one or more urls to check")
    parser.add_argument("--threshold", type=float, default=None,
                        help="how sure the model has to be before calling it phishing. default is "
                             "the one picked in training (about 1%% false alarms). "
                             "bump it up if you're getting too many false alarms")
    args = parser.parse_args()

    bundle = load_model(DEFAULT_MODEL_PATH)
    meta = bundle["metadata"]
    # use the threshold from training unless I gave one
    threshold = args.threshold
    if threshold is None:
        threshold = meta.get("threshold", 0.5)
    print(f"Model: {meta['model_name']} (trained {meta['trained_at']}), threshold {threshold:.3f}\n")

    # score all the urls in one go, way faster than one at a time
    probs = predict_proba(bundle, args.urls)

    for url, prob in zip(args.urls, probs):
        if prob >= threshold:
            verdict = "PHISHING"
        else:
            verdict = "looks legitimate"

        # right-align the verdict so the percentages and urls line up in a column
        print(f"{verdict:>16}  {prob:6.1%}  {url}")

        # list whatever looked off about it, indented under the url
        for flag in red_flags(url):
            print(f"{'':>26}- {flag}")


if __name__ == "__main__":
    main()
