
# LuckOfTheSea
Machine learning phishing URL detector: data pipeline, feature engineering, model evaluation, and a deployed API.



## Phase 1 - Environment setup
Goals: Install tools, create repo, get tests passing.

**Did:**
- Installed Python 3, Git, VS Code
- Created a venv and installed requirements

**Results:**
- `pytest`:   8 passed

**Learned:**
Learned about XGBoost and Logistic Regression, and how I plan on using it for this project.

**Next:**
- Run `collect_data.py` and set up a dataset

---

## Phase 2 · First data collection

**Goal:** Collect all data into dataset

**Did:** 	 Added a script to clean and process URL dataset

**Problems & fixes:**
Common Crawl was a bottleneck, so I switched to sitemaps.

**Learned:** Zoom is a surprisingly common target for phishing.

**Next:** Collecting data from URLs with ``build_dataset.py``

---
## Phase 3 · URL Data Collection

**Goal:**  Use extract_features(url) to turn one URL into a dictionary of 34 numbers. 

``from phishdetect import extract_features``
``extract_features("http://paypal.com.account-verify.xyz/signin/webscr.php?cmd=login")``


**Did:**
- Ran `extract_features` from an interactive Python session in the repo root:
  ```python
  from phishdetect import extract_features
  extract_features("http://paypal.com.account-verify.xyz/signin/webscr.php?cmd=login")
  ```
- Ran a legit URL (`https://github.com/nuggetofwisdom`) through it to compare

**Results:**

Selected features, phishing example vs. legit example:

| Feature | Legit (avg) | Phishing (avg) | What it means |
|---|---|---|---|
| `url_length` | 60.707 | 57.419 | Nearly identical, so length alone doesn't separate the classes in this dataset |
| `host_length` | 14.503 | 21.942 | Phishing hosts are longer, often padded with extra words or subdomains |
| `path_length` | 37.849 | 12.793 | Legit URLs have much longer paths (real articles, docs, product pages) |
| `query_length` | 0.351 | 14.288 | Phishing URLs carry far more query-string data (tracking IDs, redirect params) |
| `domain_label_length` | 8.432 | 9.633 | Phishing domain names run slightly longer |
| `subdomain_length` | 1.525 | 7.596 | Phishing leans heavily on subdomains to fake legitimacy |
| `num_dots` | 1.703 | 2.026 | More dots in phishing URLs, mostly from extra subdomains |
| `num_hyphens` | 3.266 | 1.038 | Legit URLs have more hyphens, mostly from slug-style paths like `/how-to-do-x` |
| `num_hyphens_in_host` | 0.074 | 0.511 | Hyphens in the hostname are a strong phishing signal (`secure-login-bank.com`) |
| `num_underscores` | 0.081 | 0.218 | Slightly more common in phishing |
| `num_slashes` | 4.470 | 3.472 | Legit URLs go deeper into site structure |
| `num_query_params` | 0.023 | 0.333 | Phishing URLs carry parameters about 14× as often |
| `num_special_chars` | 0.872 | 1.074 | Slightly more special characters in phishing |
| `digit_ratio` | 0.042 | 0.082 | Phishing URLs contain about twice the share of digits |
| `domain_has_digit` | 0.067 | 0.210 | 21% of phishing domains contain a digit vs 7% of legit |
| `has_at_symbol` | 0.000 | 0.015 | Rare, but it only appears in phishing (an `@` hides the real host) |
| `has_ip_host` | 0.000 | 0.004 | Raw IP hosts are rare, but they only appear in phishing |
| `has_nonstandard_port` | 0.000 | 0.001 | Almost never seen in either class |
| `uses_https` | 0.984 | 0.845 | Most phishing uses HTTPS now (85%), so a padlock isn't proof of safety |
| `has_punycode` | 0.005 | 0.000 | Slightly more common in legit, likely real international domains |
| `num_subdomains` | 0.500 | 0.688 | Phishing uses more subdomains |
| `path_depth` | 2.003 | 1.042 | Legit pages sit deeper in a site's folder structure |
| `has_double_slash_in_path` | 0.000 | 0.003 | Rare redirect trick, seen only in phishing |
| `has_suspicious_extension` | 0.089 | 0.112 | `.php` and similar extensions are only slightly more common in phishing |
| `has_http_token_in_host_or_path` | 0.000 | 0.002 | Rare; `http` embedded in a host or path to look official |
| `is_url_shortener` | 0.000 | 0.016 | Shorteners hide the destination; seen only in phishing |
| `tld_suspicious` | 0.012 | 0.174 | 17% of phishing uses cheap or abused TLDs vs 1% of legit, one of the strongest signals |
| `host_entropy` | 3.189 | 3.527 | Phishing hostnames are more random-looking |
| `path_entropy` | 3.003 | 1.873 | Legit paths are longer and more varied, so they score higher |
| `num_suspicious_words` | 0.033 | 0.116 | Words like login, verify, and account appear about 3.5× as often |
| `brand_outside_domain` | 0.026 | 0.036 | Only a small difference, so it's weaker than expected |
| `brand_inside_domain_label` | 0.008 | 0.036 | Brand names inside the domain are about 4.5× as common in phishing |
| `min_brand_edit_distance` | 5.931 | 6.990 | Phishing domains are not closer to brand names on average |
| `brand_lookalike` | 0.011 | 0.011 | Identical in both classes, so this feature adds nothing as-is |


`red_flags()` output for the phishing URL:
- Mentions a brand outside the real domain name
- Uses a top-level domain that's frequently abused
- Contains several login/verification keywords
- Not using HTTPS

The GitHub URL returned no red flags.


**Learned:**
- Models can't read text, only numbers. Each feature turns one question about the URL into a number, and the 34 answers together are the "feature vector" the model learns from.
- Each feature is a hypothesis about attacker behavior. The model decides how much each one actually matters.
- Browsers and users read domains left to right, so `paypal.com.account-verify.xyz` *looks* like PayPal. The real registered domain is the part right before the TLD: `account-verify.xyz`.
- `brand_lookalike` only catches typos like `paypa1.com`. This URL uses the real brand name in the wrong spot, which is a different trick caught by `brand_outside_domain`.
- The red flags are hand-written rules for humans to read, not the model's reasoning.

**Next:**
- Run `python3 -m scripts.build_dataset` to clean, dedupe, and balance the collected data into `data/processed/dataset.csv`
- Do the feature comparison exercise in doc 03 (average of each feature for phishing vs. legit) to check which hypotheses hold up on real data

## Learning moment
``(.venv) jacob@desktop:~/LuckOfTheSea/$ python3 -m scripts.predict "[http://paypal.com.secure-verify.xyz/signin.php](http://paypal.com.secure-verify.xyz/signin.php)" "[https://github.com/nuggetofwisdom](https://github.com/nuggetofwisdom)"  
Model: XGBoost (trained 2026-10-27T02:26:13+00:00)``

``PHISHING 100.0% [http://paypal.com.secure-verify.xyz/signin.php](http://paypal.com.secure-verify.xyz/signin.php)  ``
``- Mentions a brand outside the real domain name  ``
``- Uses a top-level domain that's frequently abused  ``
``- Contains several login/verification keywords  ``
``- Not using HTTPS (note: most phishing sites DO use HTTPS now)  
PHISHING 95.0% [https://github.com/nuggetofwisdom](https://github.com/nuggetofwisdom)``


-- - 

    (.venv) jacob@desktop:~/LuckOfTheSea/phishing-url-detector$ python3 - <<'EOF'
    import pandas as pd
    from phishdetect import features_dataframe, registered_domain
    df = pd.read_csv("data/processed/dataset.csv")
    print(df.groupby(["label", "source"]).size(), "\n")
    df["reg"] = df.url.map(registered_domain)
    print("github.com rows by label:", df[df.reg == "github.com"].label.value_counts().to_dict())
    X = features_dataframe(df.url); X["label"] = df.label
    print("\nMedians (0 = legit, 1 = phishing):")
    print(X.groupby("label")[["url_length", "path_length", "path_depth", "uses_https", "num_query_params"]].median())
    print("\nLegit URLs that are bare homepages:", f"{(X[X.label == 0].path_depth == 0).mean():.0%}")
    print("Phishing URLs with path_depth <= 1:", f"{(X[X.label == 1].path_depth <= 1).mean():.0%}")
    print("Legit URLs with path_depth == 1:", f"{(X[X.label == 0].path_depth == 1).mean():.0%}")
    EOF
    head -12 reports/feature_importance.csv
    grep -A8 '"XGBoost"' reports/metrics.json
    label  source
    0      tranco_sitemaps    5411
    1      openphish            44
           phishtank          5156
           phishunt            211
    dtype: int64
    github.com rows by label: {}
    Medians (0 = legit, 1 = phishing):
           url_length  path_length  path_depth  uses_https  num_query_params
    label
    0            54.0         30.0         2.0         1.0               0.0
    1            39.0          6.0         1.0         1.0               0.0
    Legit URLs that are bare homepages: 22%
    Phishing URLs with path_depth <= 1: 76%
    Legit URLs with path_depth == 1: 14%
    ,importance
    subdomain_length,0.25195312
    num_query_params,0.0725311
    num_subdomains,0.06480583
    path_length,0.058433767
    num_hyphens,0.057249945
    has_punycode,0.0563148
    tld_suspicious,0.054610956
    query_length,0.043543108
    num_hyphens_in_host,0.034138963
    uses_https,0.025666717
    brand_lookalike,0.023837961
        "model_name": "XGBoost",
        "trained_at": "2026-10-27T02:26:13+00:00",
        "split": "domain",
        "n_train": 8657,
        "n_test": 2165,
        "test_metrics": {
          "accuracy": 0.9191685912240185,
          "precision": 0.9128521126760564,
         "recall": 0.9317160826594789,
    --
        "XGBoost": {
          "accuracy": 0.9191685912240185,
          "precision": 0.9128521126760564,
          "recall": 0.9317160826594789,
          "f1": 0.9221876389506447,
          "false_positive_rate": 0.094106463878327,
          "roc_auc": 0.9728361500278424,
          "tp": 1037,
          "fp": 99,

Here we see that the #1 feature is `subdomain_length` That’s a red flag, since subdomains shouldn’t matter that much. An even bigger red flag however, is that the false positive rate is 9.4%.

My best guess is that sitemaps list official `www.` URLs, so the model learned "has www = legit" instead of anything about phishing, which is also why a plain `github.com/nuggetofwisdom` looked suspicious. On top of that, 76% of my phishing URLs were only 0–1 folders deep, but only 14% of my legit URLs were exactly 1 deep, so short paths got lumped in with phishing too. To fix it, I'm stripping `www.` before extracting features and re-collecting legit URLs spread evenly across path depths. I saved the old reports in `reports/v1-before-fix/` so I can compare once I retrain, and I expect the accuracy might drop a little, since the model is losing a shortcut it shouldn't have had.

