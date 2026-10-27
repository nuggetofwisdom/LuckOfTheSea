"""
tests for features.py - makes sure the feature code actually does what I think it does

run with:  pytest
"""
from phishdetect.features import (
    FEATURE_NAMES,
    extract_features,
    levenshtein,
    red_flags,
    registered_domain,
    shannon_entropy,
)


def test_every_url_gets_the_same_features():
    # the model needs the same columns in the same order every time,
    # even for junk input
    urls = [
        "https://google.com",
        "paypal.com/login",
        "http://1.2.3.4:8080/a?b=c",
        "not a url at all",
    ]
    for url in urls:
        assert list(extract_features(url).keys()) == FEATURE_NAMES


def test_ip_address_host():
    assert extract_features("http://192.168.10.5/login")["has_ip_host"] == 1
    assert extract_features("https://example.com/login")["has_ip_host"] == 0


def test_at_symbol_and_port():
    # everything before the @ is ignored by the browser, real host is evil.example
    f = extract_features("http://paypal.com@evil.example:8443/x")
    assert f["has_at_symbol"] == 1
    assert f["has_nonstandard_port"] == 1


def test_brand_lookalike():
    assert extract_features("https://paypa1.com/signin")["brand_lookalike"] == 1
    # the real paypal shouldn't get flagged
    assert extract_features("https://paypal.com/signin")["brand_lookalike"] == 0


def test_brand_outside_real_domain():
    assert extract_features("https://paypal.com.secure-check.xyz/")["brand_outside_domain"] == 1
    assert extract_features("https://www.paypal.com/us/home")["brand_outside_domain"] == 0


def test_registered_domain():
    assert registered_domain("https://login.paypal.com.evil.co.uk/x") == "evil.co.uk"
    assert registered_domain("github.com/nuggetofwisdom") == "github.com"


def test_helpers():
    assert levenshtein("paypa1", "paypal") == 1
    assert shannon_entropy("aaaa") == 0  # all the same char, no randomness
    assert shannon_entropy("abcd") == 2  # 4 different chars = 2 bits each


def test_www_is_ignored():
    # www. was just a leftover from how I collected the data, so the
    # features should come out the same with or without it
    assert extract_features("https://www.github.com/x") == extract_features("https://github.com/x")
    assert extract_features("www.paypal.com.evil.xyz") == extract_features("paypal.com.evil.xyz")
    # but only strip it at the very start of the host, not in the middle
    assert extract_features("https://login.www.evil.xyz")["num_subdomains"] == 2


def test_red_flags_are_strings():
    flags = red_flags("http://192.168.0.1/paypal/verify-account/login.php")
    assert flags
    assert all(isinstance(f, str) for f in flags)


def test_short_brands_have_to_be_whole_words():
    # "ups" is in "groups" and "att"/"irs" are in "attachments/first", none of those are brands
    assert extract_features("https://www.reddit.com/r/groups/")["brand_outside_domain"] == 0
    assert extract_features("https://example.com/attachments/first.pdf")["brand_outside_domain"] == 0
    # the real thing should still count
    assert extract_features("https://track.example.xyz/ups/parcel")["brand_outside_domain"] == 1
    assert extract_features("https://secure-ups-delivery.com/")["brand_inside_domain_label"] == 1
    assert extract_features("https://login.securepaypal.com/")["brand_inside_domain_label"] == 1


def test_lookalike_has_to_be_close():
    assert extract_features("https://amaz0n.com/")["brand_lookalike"] == 1
    assert extract_features("https://apps.com/")["brand_lookalike"] == 0  # 2 letters off from apple, too far
