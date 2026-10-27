"""
allowlist.py - domains that are trusted enough to skip the model

the model only sees url text, so it has no idea accounts.google.com is the real
google. real phishing filters check a list of known good domains first and only
run the model on everything else. that's all this is.

NOT a model feature on purpose: my legit training urls come from the Tranco
list, so "is it on Tranco" would basically be the answer key. the model would
score ~100% and it'd mean nothing.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from .features import URL_SHORTENERS, _normalize, registered_domain

DEFAULT_ALLOWLIST_PATH = Path("seeds/top_domains.txt")

# trusted sites that let anyone put pages or files on them. a google form asking
# for your password is still on google.com, so these never get a free pass
SHARED_HOSTS = (
    "sites.google.com", "docs.google.com", "drive.google.com", "forms.gle",
    "storage.googleapis.com", "firebasestorage.googleapis.com", "web.app", "firebaseapp.com",
    "onedrive.live.com", "1drv.ms", "sharepoint.com", "forms.office.com",
    "dropbox.com", "dropboxusercontent.com", "blob.core.windows.net", "s3.amazonaws.com",
    "github.io", "githubusercontent.com", "pages.dev", "workers.dev", "r2.dev",
    "weebly.com", "wixsite.com", "blogspot.com", "netlify.app", "vercel.app",
    "000webhostapp.com", "godaddysites.com", "glitch.me", "ngrok.io", "herokuapp.com",
    "notion.site", "canva.site", "typeform.com", "jotform.com", "ipfs.io",
)


# top level domains that belong to one company, nobody else can register a
# .microsoft or .google domain. outlook.cloud.microsoft is the real outlook
BRAND_TLDS = {"microsoft", "office", "windows", "xbox", "azure", "bing",
              "google", "youtube", "apple", "amazon", "aws"}


def load_allowlist(path=DEFAULT_ALLOWLIST_PATH):
    # one domain per line. no file = empty list, so everything goes to the model
    path = Path(path)
    if not path.exists():
        return set()
    domains = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            domains.add(line)
    return domains


def is_shared_host(domain):
    domain = domain.lower()
    return any(domain == h or domain.endswith("." + h) for h in SHARED_HOSTS)


def is_trusted(url, allowlist):
    try:
        host = urlsplit(_normalize(url)).hostname or ""
    except ValueError:
        return False
    if is_shared_host(host):
        return False
    if host.rsplit(".", 1)[-1] in BRAND_TLDS:
        return True
    domain = registered_domain(url)
    return domain in allowlist and domain not in URL_SHORTENERS
