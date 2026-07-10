from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_COLUMN_CANDIDATES = ("text", "comment", "comment_text", "message", "content")
LABEL_COLUMN_CANDIDATES = ("label", "toxic", "is_toxic", "target", "score")


@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[int]
    violation_types: list[str]


@dataclass(frozen=True)
class RegressionDataset:
    texts: list[str]
    targets: list[float]
    violation_types: list[str]
    text_column: str
    target_column: str


def _detect_column(fieldnames: Iterable[str], candidates: tuple[str, ...], explicit: str | None) -> str:
    names = list(fieldnames)
    if explicit:
        if explicit not in names:
            raise ValueError(f"Column '{explicit}' was not found. Available columns: {names}")
        return explicit

    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Cannot detect column. Expected one of: {candidates}. Available columns: {names}")


def _to_label(value: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "toxic", "bad"}:
        return 1
    if normalized in {"0", "false", "no", "non-toxic", "normal", "good"}:
        return 0
    try:
        return 1 if float(normalized.replace(",", ".")) >= 0.5 else 0
    except ValueError as exc:
        raise ValueError(f"Cannot convert label value '{value}' to 0/1") from exc


def _to_float_target(value: str) -> float:
    normalized = str(value).strip().replace(",", ".")
    if normalized == "":
        raise ValueError("Target value is empty")
    return float(normalized)


def profile_csv(
    path: str | Path,
    *,
    text_column: str | None = None,
    target_column: str | None = None,
) -> dict:
    path = Path(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    fieldnames = list(rows[0].keys())
    text_col = _detect_column(fieldnames, TEXT_COLUMN_CANDIDATES, text_column)
    target_col = _detect_column(fieldnames, LABEL_COLUMN_CANDIDATES + ("toxicity", "toxic_score"), target_column)

    missing = {name: 0 for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            if (row.get(name) or "").strip() == "":
                missing[name] += 1

    texts = [(row.get(text_col) or "").strip() for row in rows]
    duplicate_texts = len(texts) - len(set(texts))
    targets = []
    for row in rows:
        value = (row.get(target_col) or "").strip()
        if value:
            targets.append(_to_float_target(value))

    unique_targets = sorted(set(targets))
    return {
        "path": str(path),
        "rows": len(rows),
        "columns": fieldnames,
        "detected_text_column": text_col,
        "detected_target_column": target_col,
        "missing_values": missing,
        "duplicate_texts": duplicate_texts,
        "target_min": min(targets) if targets else None,
        "target_max": max(targets) if targets else None,
        "target_unique_count": len(unique_targets),
        "target_unique_preview": unique_targets[:20],
    }


def load_dataset(
    path: str | Path,
    *,
    text_column: str | None = None,
    label_column: str | None = None,
) -> Dataset:
    path = Path(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    fieldnames = rows[0].keys()
    text_col = _detect_column(fieldnames, TEXT_COLUMN_CANDIDATES, text_column)
    label_col = _detect_column(fieldnames, LABEL_COLUMN_CANDIDATES, label_column)
    violation_col = "violation_type" if "violation_type" in rows[0] else None

    texts: list[str] = []
    labels: list[int] = []
    violation_types: list[str] = []
    for row in rows:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        texts.append(text)
        labels.append(_to_label(row.get(label_col, "0")))
        violation_types.append((row.get(violation_col) or "unknown") if violation_col else "unknown")

    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain both classes: toxic=1 and normal=0")
    return Dataset(texts=texts, labels=labels, violation_types=violation_types)


def load_regression_dataset(
    path: str | Path,
    *,
    text_column: str | None = None,
    target_column: str | None = None,
) -> RegressionDataset:
    path = Path(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    fieldnames = rows[0].keys()
    text_col = _detect_column(fieldnames, TEXT_COLUMN_CANDIDATES, text_column)
    target_col = _detect_column(fieldnames, LABEL_COLUMN_CANDIDATES + ("toxicity", "toxic_score"), target_column)
    violation_col = "violation_type" if "violation_type" in rows[0] else None

    texts: list[str] = []
    targets: list[float] = []
    violation_types: list[str] = []
    for row in rows:
        text = (row.get(text_col) or "").strip()
        target_raw = (row.get(target_col) or "").strip()
        if not text or not target_raw:
            continue
        texts.append(text)
        targets.append(_to_float_target(target_raw))
        violation_types.append((row.get(violation_col) or "unknown") if violation_col else "unknown")

    if len(texts) < 10:
        raise ValueError("Dataset is too small after dropping empty text/target rows")
    if len(set(targets)) < 2:
        raise ValueError("Target must contain at least two different values")
    return RegressionDataset(
        texts=texts,
        targets=targets,
        violation_types=violation_types,
        text_column=text_col,
        target_column=target_col,
    )
