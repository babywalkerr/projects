from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]")
TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-zа-я0-9]+", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)

HOMOGLYPHS = str.maketrans(
    {
        "a": "а",
        "e": "е",
        "o": "о",
        "p": "р",
        "c": "с",
        "x": "х",
        "y": "у",
        "k": "к",
        "m": "м",
        "t": "т",
        "b": "в",
        "h": "н",
    }
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.lower().replace("ё", "е")
    text = text.replace("–", "-").replace("—", "-")
    return text


def normalize_homoglyphs(text: str) -> str:
    return normalize_text(text).translate(HOMOGLYPHS)


def squash_repeated_chars(text: str) -> str:
    return REPEATED_CHAR_RE.sub(r"\1\1", text)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def token_variants(text: str) -> set[str]:
    variants: set[str] = set()
    for candidate in (normalize_text(text), normalize_homoglyphs(text)):
        variants.update(TOKEN_RE.findall(candidate))
        variants.update(TOKEN_RE.findall(squash_repeated_chars(candidate)))
    return {item for item in variants if item}


def compact_text(text: str, *, homoglyphs: bool = False) -> str:
    base = normalize_homoglyphs(text) if homoglyphs else normalize_text(text)
    return NON_WORD_RE.sub("", squash_repeated_chars(base))

