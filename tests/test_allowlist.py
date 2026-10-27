"""tests for allowlist.py"""
from phishdetect.allowlist import is_trusted, load_allowlist

ALLOW = {"google.com", "microsoftonline.com", "paypal.com", "bit.ly"}


def test_trusted_domains():
    assert is_trusted("https://accounts.google.com/v3/signin/identifier?service=mail", ALLOW)
    assert is_trusted("https://login.microsoftonline.com/common/oauth2/v2.0/authorize", ALLOW)
    assert is_trusted("https://www.paypal.com/signin", ALLOW)


def test_lookalikes_arent_trusted():
    # real domain is signin-verify.net, not paypal.com
    assert not is_trusted("https://paypal.com.signin-verify.net/", ALLOW)
    assert not is_trusted("https://paypa1.com/", ALLOW)
    # browser goes to evil.example, everything before the @ is ignored
    assert not is_trusted("http://paypal.com@evil.example/", ALLOW)


def test_shared_hosting_never_trusted():
    # anyone can make one of these
    assert not is_trusted("https://docs.google.com/forms/d/e/abc/viewform", ALLOW)
    assert not is_trusted("https://sites.google.com/view/paypal-help", ALLOW)
    # shorteners hide where they go, even if they end up on the list
    assert not is_trusted("https://bit.ly/3abcdef", ALLOW)


def test_missing_file_is_empty(tmp_path):
    assert load_allowlist(tmp_path / "nope.txt") == set()
