"""
features.py - turns a url string into a bunch of numbers

The model can't read text, it only sees numbers. So every check in here asks
one question about a url (how long is it? is the host an IP?) and gives back a
number. All the answers together for one url = the feature vector, which is
what the model actually trains on.

Pretty much every feature is based on some trick phishers actually use.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd
import tldextract

# suffix_list_urls=() -> use the public suffix list that ships with tldextract
# instead of downloading it every time. works offline + results don't change
# include_psl_private_domains=True makes free hosting sites (web.app, github.io,
# pages.dev, etc) count as a suffix. so evil-login.web.app is its own site instead
# of every firebase page getting lumped together as "web.app"
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

# site builders that get used for phishing but aren't on that list yet
EXTRA_HOSTING_SUFFIXES = {"weebly.com", "000webhostapp.com", "godaddysites.com", "mystrikingly.com"}

# words that show up a lot on credential harvesting pages
SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "logon", "verify", "verification", "account",
    "update", "secure", "security", "confirm", "password", "wallet", "billing",
    "invoice", "suspend", "unlock", "webscr", "authenticate", "support",
    "recover", "validate", "payment", "banking",
]

# brands that get impersonated a lot. keep adding to this as I go through the data
BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "chase",
    "wellsfargo", "bankofamerica", "citibank", "dhl", "usps", "fedex", "ups",
    "coinbase", "binance", "metamask", "steam", "roblox", "adobe", "dropbox",
    "docusign", "linkedin", "att", "verizon", "irs",
]

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "s.id",
    "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "buff.ly", "rb.gy",
    "t.ly",
}

# cheap / heavily abused TLDs. this is a guess for now, need to check it
# against the actual data
SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "icu", "sbs", "click", "link", "live", "shop",
    "online", "site", "buzz", "cfd", "rest", "monster", "cyou", "tk", "ml",
    "ga", "cf", "gq", "info", "work", "support",
}

# short brands like att, ups, irs, dhl show up inside normal words ("groups",
# "attachments", "first") so those only count as a whole word. longer ones are
# unique enough to match inside a word too, like "securepaypal"
SUBSTRING_BRAND_MIN = 6

# only check edit distance against longer brands. pretty much any short word
# is 1 or 2 letters off from "ups" or "att"
LOOKALIKE_BRANDS = [b for b in BRANDS if len(b) >= 5]

# numbers people swap in for letters (micros0ft, paypa1)
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s"})

# cyrillic letters that look exactly like latin ones. "аpple" with a cyrillic а
# is a totally different domain (xn--pple-43d.com) that looks the same in a browser
CONFUSABLES = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
                             "у": "y", "х": "x", "і": "i", "ӏ": "l"})

SUSPICIOUS_EXTENSIONS = (".php", ".html", ".htm", ".asp", ".aspx", ".js", ".exe", ".zip")


# --- helpers ---

def _normalize(url):
    # urlsplit gets the host wrong if there's no scheme, so tack one on
    url = str(url).strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def _strip_www(url):
    # sitemaps list the official www. version of every site, so "has www" ended
    # up meaning "legit" to the model. drop it, but only at the very start of the host
    return re.sub(r"^((?:[a-zA-Z][a-zA-Z0-9+.-]*://)?)www\.", r"\1", url, flags=re.IGNORECASE)


def _words(text):
    # "secure-paypal.com/login_now" -> ["secure", "paypal", "com", "login", "now"]
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _mentions_brand(text, brand):
    if brand in _words(text):
        return True
    if len(brand) >= SUBSTRING_BRAND_MIN and brand in text:
        return True
    return False


def _skeleton(label):
    # what a domain label looks like to a person: micros0ft -> microsoft,
    # xn--pple-43d -> apple. also rn -> m since "rn" looks like "m" in most fonts
    if label.startswith("xn--"):
        try:
            label = label.encode().decode("idna")
        except UnicodeError:
            pass
    return label.translate(CONFUSABLES).translate(LEET).replace("rn", "m")


def _is_lookalike(label, brand):
    # paypa1 vs paypal is 1 edit. short brands only get 1 edit, longer ones
    # get 2, otherwise "apps" counts as a fake "apple"
    if label == brand:
        return False  # that's just the real one
    if _skeleton(label) == brand:
        return True  # micros0ft, xn--pple-43d

    # typosquats swap a letter (amaz0n) or add one (paypall). a label that's
    # shorter than the brand is almost always just a real word that happens to
    # be close, like "cloud" vs "icloud" (outlook.cloud.microsoft, cloud.com)
    if len(label) < len(brand):
        return False
    dist = levenshtein(label, brand)
    if len(brand) <= 6:
        return 0 < dist <= 1
    return 0 < dist <= 2


def shannon_entropy(text):
    # how random a string looks (bits per char). "google" is low,
    # "x7kq9zp2vb" is high. phishing kits love random subdomains/paths
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def levenshtein(a, b):
    # edit distance, e.g. levenshtein("paypa1", "paypal") == 1
    # good for catching lookalike domains
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        row = [i]
        for j, cb in enumerate(b, start=1):
            delete = prev[j] + 1
            insert = row[j - 1] + 1
            sub = prev[j - 1] + (ca != cb)
            row.append(min(delete, insert, sub))
        prev = row
    return prev[-1]


def _is_ip(host):
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        # hex/integer IPs (0x7f000001 etc) are another way to sneak an IP in
        return bool(re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", host, re.IGNORECASE))


def _split_host(url):
    # "login.paypal.com.evil.co.uk" -> ("login.paypal.com", "evil", "co.uk")
    parts = _EXTRACT(url)
    sub = (parts.subdomain or "").lower()
    label = (parts.domain or "").lower()
    suffix = (parts.suffix or "").lower()

    # abc.weebly.com should be the site "abc" not weebly, same as web.app
    if sub and f"{label}.{suffix}" in EXTRA_HOSTING_SUFFIXES:
        sub_parts = sub.split(".")
        suffix = f"{label}.{suffix}"
        label = sub_parts[-1]
        sub = ".".join(sub_parts[:-1])
    return sub, label, suffix


def registered_domain(url):
    # "https://login.paypal.com.evil.co.uk/x" -> "evil.co.uk"
    _, label, suffix = _split_host(_normalize(url))
    if label and suffix:
        return f"{label}.{suffix}"
    return (label or urlsplit(_normalize(url)).hostname or "").lower()


# --- main feature extraction ---

def extract_features(url) -> dict[str, float]:
    raw = _strip_www(str(url).strip())
    norm = _normalize(raw)

    try:
        parts = urlsplit(norm)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        # usually a garbage port number. retry on whatever's after the @ (if any)
        if "@" in norm:
            parts = urlsplit(norm.split("@")[-1])
        else:
            parts = urlsplit("http://invalid")
        host = (parts.hostname or "").lower()
        port = None

    path = parts.path or ""
    query = parts.query or ""

    subdomain, domain_label, suffix = _split_host(norm)  # "login.secure", "paypal", "com"
    tld = suffix.split(".")[-1]                           # "uk" from co.uk
    if suffix:
        reg_domain = f"{domain_label}.{suffix}"
    else:
        reg_domain = domain_label
    lower = raw.lower()

    # brand stuff - is a brand name showing up somewhere it shouldn't?
    outside = subdomain + "/" + path + "?" + query
    brand_elsewhere = any(b != domain_label and _mentions_brand(outside, b) for b in BRANDS)
    # check the skeleton, but compare against the real label so apple.com itself doesn't count
    skeleton = _skeleton(domain_label)
    brand_in_label = any(b != domain_label and _mentions_brand(skeleton, b) for b in BRANDS)
    min_brand_dist = min((levenshtein(domain_label, b) for b in LOOKALIKE_BRANDS), default=99)
    lookalike = len(domain_label) >= 4 and any(_is_lookalike(domain_label, b) for b in LOOKALIKE_BRANDS)

    f = {
        # lengths
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "domain_label_length": len(domain_label),
        "subdomain_length": len(subdomain),

        # character counts
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_hyphens_in_host": host.count("-"),
        "num_underscores": raw.count("_"),
        "num_slashes": raw.count("/"),
        "num_query_params": len([p for p in query.split("&") if p]),
        "num_special_chars": sum(raw.count(c) for c in "=?&%~+!*,;$"),
        "digit_ratio": sum(c.isdigit() for c in raw) / max(len(raw), 1),
        "domain_has_digit": int(any(c.isdigit() for c in domain_label)),

        # structural red flags
        "has_at_symbol": int("@" in raw),
        "has_ip_host": int(_is_ip(host)),
        "has_nonstandard_port": int(port not in (None, 80, 443)),
        "uses_https": int(parts.scheme == "https"),
        "has_punycode": int("xn--" in host),
        "num_subdomains": len([s for s in subdomain.split(".") if s]),
        "path_depth": len([p for p in path.split("/") if p]),
        "has_double_slash_in_path": int("//" in path),
        "has_suspicious_extension": int(path.lower().endswith(SUSPICIOUS_EXTENSIONS)),
        # stuff like https-paypal-login.com or evil.com/http://paypal.com
        "has_http_token_in_host_or_path": int("https" in host or "http" in path.lower()),

        # domain rep-ish
        "is_url_shortener": int(reg_domain in URL_SHORTENERS),
        "tld_suspicious": int(tld in SUSPICIOUS_TLDS),

        # randomness
        "host_entropy": shannon_entropy(host),
        "path_entropy": shannon_entropy(path),

        # social engineering
        "num_suspicious_words": sum(lower.count(w) for w in SUSPICIOUS_WORDS),
        "brand_outside_domain": int(brand_elsewhere),
        "brand_inside_domain_label": int(brand_in_label),
        "min_brand_edit_distance": min_brand_dist,
        "brand_lookalike": int(lookalike),
    }
    return {k: float(v) for k, v in f.items()}


# column order for the model, just run a dummy url through to get the keys
FEATURE_NAMES: list[str] = list(extract_features("http://example.com").keys())


def features_dataframe(urls):
    # one row per url
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)


def model_input(urls):
    # what every model gets: the 34 features plus the url itself. the feature
    # models ignore the url column and the char model only looks at that
    urls = [str(u) for u in urls]
    X = features_dataframe(urls)
    X["url"] = urls
    return X


def red_flags(url):
    # Plain english reasons a url looks sketchy. These are just rules, NOT
    # what the model is thinking - it's so the API has something readable
    # to show next to the score.
    f = extract_features(url)
    flags = []
    if f["has_ip_host"]:
        flags.append("Uses a raw IP address instead of a domain name")
    if f["has_at_symbol"]:
        flags.append("Contains '@', which can hide the real destination")
    if f["has_punycode"]:
        flags.append("Uses punycode (xn--), which can disguise lookalike characters")
    if f["brand_lookalike"]:
        flags.append("Domain is spelled almost like a well-known brand")
    elif f["brand_inside_domain_label"]:
        flags.append("Has a brand name (or a disguised one) inside the domain name")
    if f["brand_outside_domain"]:
        flags.append("Mentions a brand outside the real domain name")
    if f["is_url_shortener"]:
        flags.append("Uses a link shortener that hides the destination")
    if f["tld_suspicious"]:
        flags.append("Uses a top-level domain that's frequently abused")
    if f["num_subdomains"] >= 3:
        flags.append("Has an unusually deep chain of subdomains")
    if f["num_suspicious_words"] >= 2:
        flags.append("Contains several login/verification keywords")
    if f["has_nonstandard_port"]:
        flags.append("Uses a nonstandard port")
    if not f["uses_https"]:
        flags.append("Not using HTTPS (note: most phishing sites DO use HTTPS now)")
    return flags
