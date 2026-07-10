from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import _detect_column, _to_label, LABEL_COLUMN_CANDIDATES, TEXT_COLUMN_CANDIDATES
from toxicity_moderation.text_features import preprocess_text


@dataclass(frozen=True)
class RawRow:
    row_index: int
    text: str
    normalized_text: str
    label: int
    source_label: str
    row: dict[str, str]


def read_rows(path: Path, text_column: str | None, label_column: str | None) -> tuple[list[RawRow], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f"Dataset is empty: {path}")

    fieldnames = list(raw_rows[0].keys())
    text_col = _detect_column(fieldnames, TEXT_COLUMN_CANDIDATES, text_column)
    label_col = _detect_column(fieldnames, LABEL_COLUMN_CANDIDATES, label_column)

    rows: list[RawRow] = []
    empty_text_rows = 0
    bad_label_rows = 0
    for idx, row in enumerate(raw_rows):
        text = (row.get(text_col) or "").strip()
        normalized = preprocess_text(text)
        if not normalized:
            empty_text_rows += 1
            continue
        try:
            label = _to_label(row.get(label_col, "0"))
        except ValueError:
            bad_label_rows += 1
            continue
        rows.append(
            RawRow(
                row_index=idx,
                text=text,
                normalized_text=normalized,
                label=label,
                source_label=(row.get("source_label") or row.get(label_col) or "").strip(),
                row=row,
            )
        )

    profile = {
        "path": str(path),
        "raw_rows": len(raw_rows),
        "usable_rows": len(rows),
        "empty_or_unusable_text_rows": empty_text_rows,
        "bad_label_rows": bad_label_rows,
        "text_column": text_col,
        "label_column": label_col,
        "columns": fieldnames,
        "label_counts_raw_usable": dict(Counter(item.label for item in rows)),
    }
    return rows, profile


def choose_representative(items: list[RawRow], resolved_label: int) -> RawRow:
    same_label = [item for item in items if item.label == resolved_label]
    pool = same_label or items
    return max(pool, key=lambda item: (len(item.text), -item.row_index))


def clean_rows(
    rows: list[RawRow],
    *,
    majority_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[RawRow]] = defaultdict(list)
    for item in rows:
        groups[item.normalized_text].append(item)

    cleaned: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []

    duplicate_groups = 0
    exact_duplicate_extra_rows = 0
    conflict_groups = 0
    resolved_conflict_groups = 0
    ambiguous_conflict_groups = 0

    for normalized_text, items in groups.items():
        label_counts = Counter(item.label for item in items)
        total = len(items)
        if total > 1:
            duplicate_groups += 1
            exact_duplicate_extra_rows += total - 1

        has_conflict = len(label_counts) > 1
        if has_conflict:
            conflict_groups += 1
            positive_count = label_counts.get(1, 0)
            negative_count = label_counts.get(0, 0)
            majority_label, majority_count = label_counts.most_common(1)[0]
            ratio = majority_count / total
            conflict_record = {
                "normalized_text": normalized_text,
                "resolved_label": majority_label if ratio >= majority_ratio else "",
                "majority_ratio": ratio,
                "rows_in_group": total,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "raw_indices": " ".join(str(item.row_index) for item in items),
                "examples": " ||| ".join(item.text for item in items[:5]),
            }
            conflict_rows.append(conflict_record)

            if ratio < majority_ratio:
                ambiguous_conflict_groups += 1
                ambiguous_rows.append(conflict_record)
                continue

            resolved_conflict_groups += 1
            resolved_label = int(majority_label)
            note = "resolved_duplicate_label_conflict_by_majority"
        else:
            resolved_label = int(next(iter(label_counts)))
            note = "deduplicated_same_label" if total > 1 else "kept_unique"

        representative = choose_representative(items, resolved_label)
        cleaned.append(
            {
                "text": representative.text,
                "label": resolved_label,
                "source_label": representative.source_label,
                "clean_note": note,
                "raw_count": total,
                "raw_positive_count": label_counts.get(1, 0),
                "raw_negative_count": label_counts.get(0, 0),
                "raw_indices": " ".join(str(item.row_index) for item in items),
            }
        )

    cleaned.sort(key=lambda row: int(str(row["raw_indices"]).split()[0]))
    report = {
        "input_usable_rows": len(rows),
        "unique_normalized_texts": len(groups),
        "clean_rows": len(cleaned),
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows_removed": exact_duplicate_extra_rows,
        "conflict_groups": conflict_groups,
        "resolved_conflict_groups": resolved_conflict_groups,
        "ambiguous_conflict_groups_removed": ambiguous_conflict_groups,
        "majority_ratio": majority_ratio,
        "label_counts_clean": dict(Counter(int(row["label"]) for row in cleaned)),
        "clean_notes": dict(Counter(str(row["clean_note"]) for row in cleaned)),
    }
    return cleaned, conflict_rows, ambiguous_rows, report


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and clean duplicated/conflicting labels in labeled.csv.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "labeled.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "labeled_clean.csv"))
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--majority-ratio", type=float, default=0.80)
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "outputs" / "dataset_cleaning"))
    args = parser.parse_args()

    rows, profile = read_rows(Path(args.input), args.text_column, args.label_column)
    cleaned, conflicts, ambiguous, cleaning_report = clean_rows(rows, majority_ratio=args.majority_ratio)

    output_path = Path(args.output)
    clean_fields = [
        "text",
        "label",
        "source_label",
        "clean_note",
        "raw_count",
        "raw_positive_count",
        "raw_negative_count",
        "raw_indices",
    ]
    write_csv(output_path, cleaned, clean_fields)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        report_dir / "duplicate_label_conflicts.csv",
        conflicts,
        [
            "normalized_text",
            "resolved_label",
            "majority_ratio",
            "rows_in_group",
            "positive_count",
            "negative_count",
            "raw_indices",
            "examples",
        ],
    )
    write_csv(
        report_dir / "ambiguous_removed.csv",
        ambiguous,
        [
            "normalized_text",
            "resolved_label",
            "majority_ratio",
            "rows_in_group",
            "positive_count",
            "negative_count",
            "raw_indices",
            "examples",
        ],
    )

    full_report = {
        "profile": profile,
        "cleaning": cleaning_report,
        "outputs": {
            "clean_dataset": str(output_path),
            "conflicts": str(report_dir / "duplicate_label_conflicts.csv"),
            "ambiguous_removed": str(report_dir / "ambiguous_removed.csv"),
        },
    }
    (report_dir / "dataset_quality_report.json").write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = f"""# Отчет по честной чистке датасета

Исходный файл: `{Path(args.input).resolve()}`

Очищенный файл: `{output_path.resolve()}`

## Что сделано

- Нормализованы тексты тем же препроцессингом, который используется в модели.
- Полные дубли одного нормализованного текста схлопнуты в одну строку.
- Если у одинакового нормализованного текста были разные метки, конфликт решался только большинством дублей.
- Если у конфликтной группы не было большинства >= `{args.majority_ratio:.2f}`, группа удалялась как неоднозначная.
- Ошибки текущей модели не использовались для автоматической правки меток.

## Итоги

| Показатель | Значение |
|---|---:|
| Исходных строк | {profile['raw_rows']} |
| Используемых строк до чистки | {profile['usable_rows']} |
| Строк после чистки | {cleaning_report['clean_rows']} |
| Удалено дублей сверх первой строки | {cleaning_report['duplicate_extra_rows_removed']} |
| Групп с конфликтующей разметкой | {cleaning_report['conflict_groups']} |
| Конфликты решены большинством | {cleaning_report['resolved_conflict_groups']} |
| Неоднозначные конфликтные группы удалены | {cleaning_report['ambiguous_conflict_groups_removed']} |

## Баланс классов после чистки

| Класс | Количество |
|---|---:|
| 0 / нетоксичный | {cleaning_report['label_counts_clean'].get(0, 0)} |
| 1 / токсичный | {cleaning_report['label_counts_clean'].get(1, 0)} |
"""
    (report_dir / "README.md").write_text(md, encoding="utf-8")

    print(json.dumps(full_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
