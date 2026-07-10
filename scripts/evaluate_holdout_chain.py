from __future__ import annotations

import argparse
import csv
import json
import math
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
from toxicity_moderation.pipeline import RegressionModerator


def require_sklearn():
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ModuleNotFoundError as exc:
        raise SystemExit("Не хватает scikit-learn.") from exc
    return locals()


def score_batch(regression: RegressionModerator, texts: list[str]) -> list[float]:
    if not regression.available or regression.pipeline is None or not texts:
        return [0.0 for _ in texts]
    pipeline = regression.pipeline
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        classes = getattr(pipeline, "classes_", None)
        index = list(classes).index(1) if classes is not None and 1 in classes else -1
        return [float(row[index]) for row in proba]
    if hasattr(pipeline, "decision_function"):
        return [1.0 / (1.0 + math.exp(-float(value))) for value in pipeline.decision_function(texts)]
    scores = []
    for value in pipeline.predict(texts):
        try:
            score = float(value)
        except TypeError:
            score = float(value[0])
        scores.append(min(1.0, max(0.0, score)))
    return scores


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


def best_threshold(skl: dict[str, Any], y_true: list[int], scores: list[float]) -> dict[str, Any]:
    best = {"threshold": 0.5, "f1_toxic": -1.0}
    for idx in range(1, 100):
        threshold = idx / 100
        pred = [1 if score >= threshold else 0 for score in scores]
        f1 = float(skl["f1_score"](y_true, pred, zero_division=0))
        if f1 > best["f1_toxic"]:
            best = {"threshold": threshold, "f1_toxic": f1}
    return best


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
    parser = argparse.ArgumentParser(description="Evaluate layer-2 and chain metrics on a held-out split.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "labeled_policy_clean.csv"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "toxicity_model_policy_clean.joblib"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "policy_clean_holdout"))
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    skl = require_sklearn()
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

    settings = Settings(model_path=Path(args.model_path), llm_provider="disabled")
    lexicon = ProfanityFilter(settings.profanity_paths)
    regression = RegressionModerator(Path(args.model_path))
    threshold = regression.threshold or settings.regression_block_threshold
    review_threshold = settings.regression_review_threshold

    layer2_scores: list[float] = []
    for start in range(0, len(x_test), args.batch_size):
        layer2_scores.extend(score_batch(regression, x_test[start : start + args.batch_size]))
    layer2_pred = [1 if score >= threshold else 0 for score in layer2_scores]
    layer2_metrics = metrics(skl, y_test, layer2_pred, layer2_scores)

    chain_pred: list[int] = []
    chain_scores: list[float] = []
    chain_stages: list[str] = []
    chain_matches: list[str] = []
    for text, score in zip(x_test, layer2_scores):
        hit = lexicon.find(text)
        if hit.blocked:
            chain_pred.append(1)
            chain_scores.append(1.0)
            chain_stages.append("layer_1_lexicon")
            chain_matches.append(", ".join(hit.matched))
        elif score >= threshold:
            chain_pred.append(1)
            chain_scores.append(score)
            chain_stages.append("layer_2_regression")
            chain_matches.append("")
        elif score < review_threshold:
            chain_pred.append(0)
            chain_scores.append(score)
            chain_stages.append("layer_2_low_risk")
            chain_matches.append("")
        else:
            chain_pred.append(0)
            chain_scores.append(score)
            chain_stages.append("allow_without_llm")
            chain_matches.append("")

    chain_metrics = metrics(skl, y_test, chain_pred, chain_scores)
    stage_counts: dict[str, int] = {}
    for stage in chain_stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    chain_metrics["stage_counts"] = stage_counts

    errors = []
    for idx, text, label, pred, score, stage, matched in zip(
        idx_test, x_test, y_test, chain_pred, chain_scores, chain_stages, chain_matches
    ):
        if label == pred:
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
        "model_path": str(Path(args.model_path).resolve()),
        "test_rows": len(x_test),
        "threshold": threshold,
        "review_threshold": review_threshold,
        "layer2_heldout_metrics": layer2_metrics,
        "chain_no_llm_heldout_metrics": chain_metrics,
        "oracle_threshold_note": "Diagnostic only: selected directly on test split, not an official metric.",
        "layer2_test_oracle_best_threshold": best_threshold(skl, y_test, layer2_scores),
        "chain_test_oracle_best_threshold": best_threshold(skl, y_test, chain_scores),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "holdout_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "chain_errors.csv", errors)
    write_csv(output_dir / "chain_false_positives.csv", [row for row in errors if row["error_type"] == "false_positive"])
    write_csv(output_dir / "chain_false_negatives.csv", [row for row in errors if row["error_type"] == "false_negative"])

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
