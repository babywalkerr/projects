from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .normalization import compact_text, normalize_homoglyphs, normalize_text, token_variants


COMMON_PREFIXES = (
    "",
    "на",
    "по",
    "за",
    "про",
    "вы",
    "от",
    "под",
    "при",
    "об",
    "до",
    "пере",
)

SAFE_TOKENS = {
    "страхуй",
    "застрахуй",
    "подстрахуй",
    "перестрахуй",
    "страхуйте",
    "страхуют",
}

MIN_TERM_LENGTH = 3

BAD_TERM_STEMS = (
    "хуй",
    "хуи",
    "хуя",
    "хуе",
    "хуё",
    "пизд",
    "еба",
    "еби",
    "ебу",
    "ебл",
    "ебн",
    "ёб",
    "бля",
    "бляд",
    "блят",
    "манд",
    "муд",
    "залуп",
    "уеб",
    "заеб",
    "долбоеб",
    "говн",
    "дерьм",
    "жоп",
    "huy",
    "hui",
    "khuy",
    "pizd",
    "eb",
    "yeb",
    "bly",
    "mudak",
    "zalup",
    "mand",
    "govn",
    "derm",
    "zhop",
)

SAFE_ROOT_PREFIXES = {
    "муд": ("мудр",),
    "манд": ("мандев",),
    "еба": ("ебаст",),
    "урод": ("уродител",),
}


@dataclass(frozen=True)
class LexiconHit:
    blocked: bool
    matched: list[str]
    matched_spans: list[dict] | None = None
    confidence: float = 1.0


class ProfanityFilter:
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.roots: set[str] = set()
        self.terms: set[str] = set()
        self.phrases: set[str] = set()
        self.load()

    def load(self) -> None:
        for path in self.paths:
            if path.exists():
                self._load_path(path)

    def _load_path(self, path: Path) -> None:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            items = [line]
            if " " in line and "*" not in line:
                # Raw external lists are often one-line, space-separated files.
                items = [part.strip() for part in line.split() if part.strip()]
            for item in items:
                self._add_entry(item)

    def _add_entry(self, entry: str) -> None:
        raw_entry = entry.strip()
        entry = normalize_text(raw_entry) if raw_entry.isascii() else normalize_homoglyphs(raw_entry)
        if not entry:
            return
        if entry.endswith("*"):
            self.roots.add(entry[:-1])
        elif " " in entry:
            self.phrases.add(entry)
        elif len(entry) < MIN_TERM_LENGTH:
            return
        elif not any(stem in entry for stem in BAD_TERM_STEMS):
            return
        else:
            self.terms.add(entry)

    def find(self, text: str, *, with_spans: bool = False) -> LexiconHit:
        matched: set[str] = set()
        normalized = normalize_text(text)
        normalized_homoglyphs = normalize_homoglyphs(text)
        compact = compact_text(text)
        compact_homoglyphs = compact_text(text, homoglyphs=True)
        tokens = token_variants(text)

        for phrase in self.phrases:
            if phrase in normalized or phrase in normalized_homoglyphs:
                matched.add(phrase)

        for term in self.terms:
            if term in tokens:
                matched.add(term)

        for root in self.roots:
            if self._root_matches(root, tokens, compact, compact_homoglyphs, normalized, normalized_homoglyphs):
                matched.add(f"{root}*")

        spans = None
        if with_spans and matched:
            spans = self._find_spans(text, matched)

        return LexiconHit(blocked=bool(matched), matched=sorted(matched), matched_spans=spans)

    def _find_spans(self, text: str, matched: set[str]) -> list[dict]:
        """Find the character spans of matched profanity in the original text."""
        spans: list[dict] = []
        seen_ranges: set[tuple[int, int]] = set()

        # Build a list of roots/terms to search for in the original text
        search_roots: list[str] = []
        for entry in matched:
            if entry.endswith("*"):
                search_roots.append(entry[:-1])
            else:
                search_roots.append(entry)

        # Tokenize the original text preserving positions
        original_lower = text.lower().replace("ё", "е")
        for match in re.finditer(r'[a-zа-яё0-9]+', text, re.IGNORECASE):
            word = match.group()
            start = match.start()
            end = match.end()
            word_normalized = normalize_text(word)
            word_homoglyphs = normalize_homoglyphs(word)

            for root in search_roots:
                is_match = False
                # Check with common prefixes
                for prefix in COMMON_PREFIXES:
                    prefixed_root = prefix + root
                    if word_normalized.startswith(prefixed_root) or word_homoglyphs.startswith(prefixed_root):
                        is_match = True
                        break
                if is_match:
                    if word_normalized in SAFE_TOKENS:
                        continue
                    if _is_safe_root_context(root, word_normalized):
                        continue
                    key = (start, end)
                    if key not in seen_ranges:
                        seen_ranges.add(key)
                        spans.append({"start": start, "end": end, "word": word})

        # Sort by position
        spans.sort(key=lambda s: s["start"])
        return spans

    @staticmethod
    def _root_matches(
        root: str,
        tokens: set[str],
        compact: str,
        compact_homoglyphs: str,
        normalized: str,
        normalized_homoglyphs: str,
    ) -> bool:
        for token in tokens:
            if token in SAFE_TOKENS:
                continue
            if _is_safe_root_context(root, token):
                continue
            for prefix in COMMON_PREFIXES:
                if token.startswith(prefix + root):
                    return True

        for value in (compact, compact_homoglyphs):
            if _is_safe_root_context(root, value):
                continue
            if any(value.startswith(prefix + root) for prefix in COMMON_PREFIXES):
                return True

        return False


def _spaced_root_matches(root: str, normalized: str, normalized_homoglyphs: str) -> bool:
    if len(root) < 3:
        return False
    pattern = r"(?<![a-zа-я0-9])" + r"[^a-zа-я0-9]*".join(re.escape(char) for char in root)
    pattern += r"(?![a-zа-я0-9])"
    for value in (normalized, normalized_homoglyphs):
        if re.search(pattern, value, flags=re.IGNORECASE):
            return True
    return False


def _is_safe_root_context(root: str, value: str) -> bool:
    return any(value.startswith(prefix) for prefix in SAFE_ROOT_PREFIXES.get(root, ()))
