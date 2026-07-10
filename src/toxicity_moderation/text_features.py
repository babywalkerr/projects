from __future__ import annotations

import html
import re

from .normalization import normalize_text, squash_repeated_chars

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    text = html.unescape(text or "")
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = normalize_text(text)
    text = squash_repeated_chars(text)
    return SPACE_RE.sub(" ", text).strip()

