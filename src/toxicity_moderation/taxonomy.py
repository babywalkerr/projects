from __future__ import annotations

from .lexicon import LexiconHit
from .normalization import normalize_text


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "threat": (
        "убью",
        "убить",
        "сломаю",
        "найду",
        "уничтожу",
        "зарежу",
        "вреда",
    ),
    "spam": (
        "http://",
        "https://",
        "t.me/",
        "казино",
        "заработок",
        "без вложений",
        "скидка",
        "подписк",
        "промокод",
    ),
    "identity_attack_religion": (
        "из-за веры",
        "религ",
        "верующ",
        "атеист",
    ),
    "identity_attack_nationality": (
        "национальн",
        "народ",
        "страна",
    ),
    "identity_attack_social_status": (
        "нищий",
        "бедняк",
        "бомж",
        "безработ",
    ),
    "insult": (
        "идиот",
        "дебил",
        "тупиц",
        "мраз",
        "ничтож",
        "бездар",
        "заткнись",
        "закрой рот",
    ),
}


def guess_violation_type(text: str, lexicon_hit: LexiconHit | None = None) -> str:
    if lexicon_hit and lexicon_hit.blocked:
        return "profanity"

    normalized = normalize_text(text)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "toxicity"

