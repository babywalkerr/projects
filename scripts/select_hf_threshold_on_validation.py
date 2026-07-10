from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from toxicity_moderation.config import Settings
from toxicity_moderation.data import load_dataset
from toxicity_moderation.lexicon import ProfanityFilter
from evaluate_hf_holdout import threshold_sweep, toxic_score_from_scores


def require_sklearn():
    try:
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise SystemExit("Не хватает scikit-learn.") from exc
    return locals()


def evaluate_saved_test(prediction_path: Path, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = [int(row["label"]) for row in rows]
    scores = [float(row["chain_score"]) for row in rows]
    pred = [1 if score >= threshold else 0 for score in scores]
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_toxic": float(precision_score(y, pred, zero_division=0)),
        "recall_toxic": float(recall_score(y, pred, zero_division=0)),
        "f1_toxic": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "tp": int(sum(1 for yt, yp in zip(y, pred) if yt == 1 and yp == 1)),
        "tn": int(sum(1 for yt, yp in zip(y, pred) if yt == 0 and yp == 0)),
        "fp": int(sum(1 for yt, yp in zip(y, pred) if yt == 0 and yp == 1)),
        "fn": int(sum(1 for yt, yp in zip(y, pred) if yt == 1 and yp == 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select HF threshold on validation and evaluate saved held-out test.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "labeled_policy_clean.csv"))
    parser.add_argument("--test-predictions", default=str(PROJECT_ROOT / "outputs" / "hf_holdout" / "hf_holdout_predictions.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "hf_holdout" / "validation_selected_threshold.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    skl = require_sklearn()
    dataset = load_dataset(args.data, text_column="text", label_column="label")
    x_train_full, _, y_train_full, _ = skl["train_test_split"](
        dataset.texts,
        dataset.labels,
        test_size=0.10,
        random_state=42,
        stratify=dataset.labels,
    )
    _, x_val, _, y_val = skl["train_test_split"](
        x_train_full,
        y_train_full,
        test_size=0.10,
        random_state=42,
        stratify=y_train_full,
    )

    settings = Settings(llm_provider="hf_local")
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
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, top_k=None)

    hf_scores: list[float] = []
    for start in range(0, len(x_val), args.batch_size):
        raw = classifier(x_val[start : start + args.batch_size], batch_size=args.batch_size)
        hf_scores.extend(toxic_score_from_scores(item) for item in raw)

    lexicon = ProfanityFilter(settings.profanity_paths)
    chain_scores = []
    for text, score in zip(x_val, hf_scores):
        chain_scores.append(1.0 if lexicon.find(text).blocked else score)

    sweep = threshold_sweep(
        {
            "accuracy_score": __import__("sklearn.metrics").metrics.accuracy_score,
            "precision_score": __import__("sklearn.metrics").metrics.precision_score,
            "recall_score": __import__("sklearn.metrics").metrics.recall_score,
            "f1_score": __import__("sklearn.metrics").metrics.f1_score,
            "roc_auc_score": __import__("sklearn.metrics").metrics.roc_auc_score,
        },
        y_val,
        chain_scores,
    )
    selected = sweep["best_accuracy"]["threshold"]
    test_metrics = evaluate_saved_test(Path(args.test_predictions), selected)
    report = {
        "validation_rows": len(x_val),
        "selected_threshold_source": "validation best accuracy",
        "validation_best_accuracy": sweep["best_accuracy"],
        "validation_best_f1": sweep["best_f1"],
        "test_metrics_with_validation_threshold": test_metrics,
        "note": "This is the official non-test-tuned threshold estimate for lexicon + HF local.",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
