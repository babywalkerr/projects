from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import _detect_column, _to_label, LABEL_COLUMN_CANDIDATES, TEXT_COLUMN_CANDIDATES
from toxicity_moderation.lexicon import COMMON_PREFIXES, SAFE_TOKENS, _is_safe_root_context
from toxicity_moderation.normalization import TOKEN_RE, normalize_homoglyphs, normalize_text, squash_repeated_chars


def load_hard_roots(path: Path) -> list[str]:
    roots: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith("*"):
            roots.append(normalize_text(line[:-1]))
        else:
            roots.append(normalize_text(line))
    return sorted(set(root for root in roots if len(root) >= 3))


def load_policy_roots(paths: list[Path]) -> list[str]:
    roots: list[str] = []
    for path in paths:
        if path.exists():
            roots.extend(load_hard_roots(path))
    return sorted(set(roots))


def build_root_pattern(roots: list[str]) -> re.Pattern[str]:
    escaped_prefixes = sorted((re.escape(prefix) for prefix in COMMON_PREFIXES), key=len, reverse=True)
    escaped_roots = sorted((re.escape(root) for root in roots), key=len, reverse=True)
    prefix_part = "|".join(escaped_prefixes)
    root_part = "|".join(escaped_roots)
    return re.compile(rf"^(?:{prefix_part})(?P<root>{root_part})", flags=re.IGNORECASE)


def find_hard_root_hits(text: str, pattern: re.Pattern[str]) -> list[str]:
    matched: set[str] = set()
    candidates = (
        normalize_text(text),
        normalize_homoglyphs(text),
        squash_repeated_chars(normalize_text(text)),
        squash_repeated_chars(normalize_homoglyphs(text)),
    )

    for candidate in candidates:
        for token in TOKEN_RE.findall(candidate):
            if token in SAFE_TOKENS:
                continue
            match = pattern.search(token)
            if not match:
                continue
            root = match.group("root")
            if _is_safe_root_context(root, token):
                continue
            matched.add(f"{root}*")
    return sorted(matched)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a policy-clean dataset aligned with layer-1 moderation rules.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "labeled_clean.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "labeled_policy_clean.csv"))
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "outputs" / "dataset_policy_cleaning"))
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"Dataset is empty: {input_path}")

    text_col = _detect_column(fieldnames, TEXT_COLUMN_CANDIDATES, args.text_column)
    label_col = _detect_column(fieldnames, LABEL_COLUMN_CANDIDATES, args.label_column)

    hard_roots = load_policy_roots(
        [
            PROJECT_ROOT / "data" / "profanity_roots_ru.txt",
            PROJECT_ROOT / "data" / "policy_insult_roots_ru.txt",
        ]
    )
    root_pattern = build_root_pattern(hard_roots)

    output_rows: list[dict[str, Any]] = []
    relabel_rows: list[dict[str, Any]] = []
    input_label_counts: Counter[int] = Counter()
    output_label_counts: Counter[int] = Counter()
    lexicon_hits_by_original_label: Counter[int] = Counter()
    matched_terms: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        text = (row.get(text_col) or "").strip()
        if not text:
            continue
        original_label = _to_label(row.get(label_col, "0"))
        input_label_counts[original_label] += 1

        hits = find_hard_root_hits(text, root_pattern)
        new_label = original_label
        clean_note = row.get("clean_note") or "kept"

        if hits:
            lexicon_hits_by_original_label[original_label] += 1
            matched_terms.update(hits)
            if original_label == 0:
                new_label = 1
                clean_note = "policy_relabel_0_to_1_by_layer1_lexicon"
                relabel_rows.append(
                    {
                        "row_index": idx,
                        "old_label": original_label,
                        "new_label": new_label,
                        "matched": ", ".join(hits),
                        "text": text,
                    }
                )

        out = dict(row)
        out["label"] = new_label
        out["policy_original_label"] = original_label
        out["policy_clean_note"] = clean_note
        out["policy_matched"] = ", ".join(hits) if hits else ""
        output_rows.append(out)
        output_label_counts[new_label] += 1

    output_path = Path(args.output)
    output_fields = list(output_rows[0].keys()) if output_rows else ["text", "label"]
    write_csv(output_path, output_rows, output_fields)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        report_dir / "policy_relabels_0_to_1.csv",
        relabel_rows,
        ["row_index", "old_label", "new_label", "matched", "text"],
    )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "input_label_counts": dict(input_label_counts),
        "output_label_counts": dict(output_label_counts),
        "lexicon_hits_by_original_label": dict(lexicon_hits_by_original_label),
        "relabels_0_to_1": len(relabel_rows),
        "top_matched_terms": matched_terms.most_common(50),
        "rule": "Only original label=0 rows hit by layer-1 profanity lexicon are relabeled to 1.",
    }
    (report_dir / "policy_cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = f"""# Policy-clean датасет

Очищенный от дублей вход: `{input_path.resolve()}`

Policy-clean выход: `{output_path.resolve()}`

## Правило

Метка менялась только в одну сторону: `0 -> 1`, если на тексте срабатывал слой 1 со стоп-словами.
Ошибки текущей регрессионной модели не использовались как основание для автоправки.

## Итоги

| Показатель | Значение |
|---|---:|
| Строк на входе | {len(rows)} |
| Строк на выходе | {len(output_rows)} |
| Исправлено `0 -> 1` | {len(relabel_rows)} |
| Исходных label=0 | {input_label_counts.get(0, 0)} |
| Исходных label=1 | {input_label_counts.get(1, 0)} |
| После чистки label=0 | {output_label_counts.get(0, 0)} |
| После чистки label=1 | {output_label_counts.get(1, 0)} |

Файл с исправленными строками: `policy_relabels_0_to_1.csv`.
"""
    (report_dir / "README.md").write_text(md, encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
