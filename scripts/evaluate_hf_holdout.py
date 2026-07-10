from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.config import Settings
from toxicity_moderation.data import load_dataset
from toxicity_moderation.lexicon import ProfanityFilter


def require_sklearn():
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise SystemExit("Не хватает scikit-learn.") from exc
    return locals()


def metrics(skl: dict[str, Any], y_true: list[int], y_pred: list[int], scores: list[float]) -> dict[str, Any]:
    result = {
        "accuracy": float(skl["accuracy_score"](y_true, y_pred)),
        "precision_toxic": float(skl["precision_score"](y_true, y_pred, zero_division=0)),
        "recall_toxic": float(skl["recall_score"](y_true, y_pred, zero_division=0)),
        "f1_toxic": float(skl["f1_score"](y_true, y_pred, zero_division=0)),
        "tp": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)),
        "tn": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)),
        "fp": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)),
        "fn": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)),
    }
    try:
        result["roc_auc"] = float(skl["roc_auc_score"](y_true, scores))
    except ValueError:
        result["roc_auc"] = None
    return result


def threshold_sweep(skl: dict[str, Any], y_true: list[int], scores: list[float]) -> dict[str, Any]:
    best_f1: dict[str, Any] | None = None
    best_accuracy: dict[str, Any] | None = None
    for idx in range(1, 100):
        threshold = idx / 100
        pred = [1 if score >= threshold else 0 for score in scores]
        row = {"threshold": threshold, **metrics(skl, y_true, pred, scores)}
        if best_f1 is None or row["f1_toxic"] > best_f1["f1_toxic"]:
            best_f1 = row
        if best_accuracy is None or row["accuracy"] > best_accuracy["accuracy"]:
            best_accuracy = row
    return {
        "best_f1": best_f1,
        "best_accuracy": best_accuracy,
        "note": "Diagnostic threshold sweep on held-out scores. Use a validation split for official threshold selection.",
    }


def toxic_score_from_scores(scores: list[dict[str, Any]]) -> float:
    for item in scores:
        if str(item.get("label", "")).lower() == "toxic":
            return float(item.get("score", 0.0))
    for item in scores:
        if str(item.get("label", "")).lower() == "neutral":
            return 1.0 - float(item.get("score", 0.0))
    return max((float(item.get("score", 0.0)) for item in scores), default=0.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local HF toxicity classifier on held-out split.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "labeled_policy_clean.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "hf_holdout"))
    args = parser.parse_args()

    skl = require_sklearn()
    settings = Settings(llm_provider="hf_local")
    dataset = load_dataset(args.data, text_column=args.text_column, label_column=args.label_column)
    indices = list(range(len(dataset.texts)))
    _, x_test, _, y_test, _, idx_test = skl["train_test_split"](
        dataset.texts,
        dataset.labels,
        indices,
        test_size=0.10,
        random_state=42,
        stratify=dataset.labels,
    )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_toxic_model,
        cache_dir=str(settings.hf_cache_dir),
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.hf_toxic_model,
        cache_dir=str(settings.hf_cache_dir),
        local_files_only=True,
    )
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        top_k=None,
    )

    hf_scores: list[float] = []
    for start in range(0, len(x_test), args.batch_size):
        batch = x_test[start : start + args.batch_size]
        raw = classifier(batch, batch_size=args.batch_size)
        hf_scores.extend(toxic_score_from_scores(item) for item in raw)

    hf_pred = [1 if score >= 0.5 else 0 for score in hf_scores]
    hf_metrics = metrics(skl, y_test, hf_pred, hf_scores)

    lexicon = ProfanityFilter(settings.profanity_paths)
    chain_pred: list[int] = []
    chain_scores: list[float] = []
    stages: list[str] = []
    matches: list[str] = []
    for text, score in zip(x_test, hf_scores):
        hit = lexicon.find(text)
        if hit.blocked:
            chain_pred.append(1)
            chain_scores.append(1.0)
            stages.append("layer_1_lexicon")
            matches.append(", ".join(hit.matched))
        else:
            chain_pred.append(1 if score >= 0.5 else 0)
            chain_scores.append(score)
            stages.append("layer_3_hf_local")
            matches.append("")

    chain_metrics = metrics(skl, y_test, chain_pred, chain_scores)
    stage_counts: dict[str, int] = {}
    for stage in stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    chain_metrics["stage_counts"] = stage_counts

    errors = []
    for idx, text, label, pred, score, stage, matched in zip(
        idx_test, x_test, y_test, chain_pred, chain_scores, stages, matches
    ):
        if int(label) == int(pred):
            continue
        errors.append(
            {
                "dataset_index": idx,
                "error_type": "false_positive" if label == 0 else "false_negative",
                "label": int(label),
                "prediction": int(pred),
                "score": float(score),
                "stage": stage,
                "matched": matched,
                "text": text,
            }
        )

    report = {
        "dataset": str(Path(args.data).resolve()),
        "test_rows": len(x_test),
        "hf_model": settings.hf_toxic_model,
        "hf_standalone_metrics": hf_metrics,
        "hf_standalone_threshold_sweep": threshold_sweep(skl, y_test, hf_scores),
        "chain_lexicon_plus_hf_metrics": chain_metrics,
        "chain_lexicon_plus_hf_threshold_sweep": threshold_sweep(skl, y_test, chain_scores),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hf_holdout_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "chain_lexicon_plus_hf_errors.csv", errors)
    write_csv(
        output_dir / "hf_holdout_predictions.csv",
        [
            {
                "dataset_index": idx,
                "label": int(label),
                "hf_score": float(hf_score),
                "chain_score": float(chain_score),
                "chain_stage": stage,
                "chain_matched": matched,
                "text": text,
            }
            for idx, label, hf_score, chain_score, stage, matched, text in zip(
                idx_test, y_test, hf_scores, chain_scores, stages, matches, x_test
            )
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
