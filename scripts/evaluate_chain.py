from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import load_dataset
from toxicity_moderation.metrics import binary_metrics, roc_auc_pairwise
from toxicity_moderation.pipeline import ModerationPipeline


def score_batch(regression, texts: list[str]) -> list[float]:
    if not regression.available or regression.pipeline is None or not texts:
        return [0.0 for _ in texts]
    pipeline = regression.pipeline
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        classes = getattr(pipeline, "classes_", None)
        if classes is not None and 1 in classes:
            index = list(classes).index(1)
        else:
            index = -1
        return [float(row[index]) for row in proba]
    if hasattr(pipeline, "decision_function"):
        import math

        return [1.0 / (1.0 + math.exp(-float(value))) for value in pipeline.decision_function(texts)]
    scores = []
    for value in pipeline.predict(texts):
        try:
            score = float(value)
        except TypeError:
            score = float(value[0])
        target_min = regression.metadata.get("target_min")
        target_max = regression.metadata.get("target_max")
        if target_min is not None and target_max is not None and float(target_max) > float(target_min):
            score = (score - float(target_min)) / (float(target_max) - float(target_min))
        scores.append(min(1.0, max(0.0, score)))
    return scores


def evaluate_without_llm(dataset, pipeline: ModerationPipeline, *, batch_size: int) -> tuple[list[int], list[float], dict[str, int], list[dict]]:
    predictions: list[int | None] = [None] * len(dataset.texts)
    scores: list[float] = [0.0] * len(dataset.texts)
    stages: dict[str, int] = {}
    errors: list[dict] = []
    pending_indices: list[int] = []
    pending_texts: list[str] = []

    for index, (text, label) in enumerate(zip(dataset.texts, dataset.labels)):
        lexicon_hit = pipeline.lexicon.find(text)
        if lexicon_hit.blocked:
            predictions[index] = 1
            scores[index] = 1.0
            stages["layer_1_lexicon"] = stages.get("layer_1_lexicon", 0) + 1
        else:
            pending_indices.append(index)
            pending_texts.append(text)

    threshold = pipeline.regression.threshold or pipeline.settings.regression_block_threshold
    review_threshold = pipeline.settings.regression_review_threshold

    for start in range(0, len(pending_texts), batch_size):
        batch_texts = pending_texts[start : start + batch_size]
        batch_indices = pending_indices[start : start + batch_size]
        batch_scores = score_batch(pipeline.regression, batch_texts)
        for index, score in zip(batch_indices, batch_scores):
            scores[index] = score
            if score >= threshold:
                predictions[index] = 1
                stages["layer_2_regression"] = stages.get("layer_2_regression", 0) + 1
            elif score < review_threshold:
                predictions[index] = 0
                stages["layer_2_low_risk"] = stages.get("layer_2_low_risk", 0) + 1
            else:
                predictions[index] = 0
                stages["allow_without_llm"] = stages.get("allow_without_llm", 0) + 1

    final_predictions = [int(value or 0) for value in predictions]
    for text, label, pred, score in zip(dataset.texts, dataset.labels, final_predictions, scores):
        if pred != label:
            errors.append({"text": text, "label": label, "prediction": pred, "score": score})
    return final_predictions, scores, stages, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the full moderation chain.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "chain_metrics.json"))
    parser.add_argument("--errors", default=str(PROJECT_ROOT / "outputs" / "chain_errors.jsonl"))
    parser.add_argument("--no-llm", action="store_true", help="Force-disable LLM calls for evaluation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    if args.no_llm:
        os.environ["LLM_PROVIDER"] = "disabled"

    dataset = load_dataset(args.data, text_column=args.text_column, label_column=args.label_column)
    if args.limit:
        dataset = type(dataset)(
            texts=dataset.texts[: args.limit],
            labels=dataset.labels[: args.limit],
            violation_types=dataset.violation_types[: args.limit],
        )
    pipeline = ModerationPipeline()

    if args.no_llm:
        predictions, scores, stages, errors = evaluate_without_llm(dataset, pipeline, batch_size=args.batch_size)
    else:
        predictions = []
        scores = []
        stages = {}
        errors = []
        for text, label in zip(dataset.texts, dataset.labels):
            result = pipeline.moderate(text)
            pred = int(result["final_label"])
            score = float(result["confidence"] if pred == 1 else 1.0 - result["confidence"])
            predictions.append(pred)
            scores.append(score)
            stages[result["stage"]] = stages.get(result["stage"], 0) + 1
            if pred != label:
                errors.append({"text": text, "label": label, "prediction": pred, "result": result})

    metrics = binary_metrics(dataset.labels, predictions)
    metrics["roc_auc"] = roc_auc_pairwise(dataset.labels, scores)
    report = {
        "items": len(dataset.texts),
        "metrics": metrics,
        "stage_counts": stages,
        "model_loaded": pipeline.regression.available,
        "llm_provider": pipeline.llm.provider,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    error_path = Path(args.errors)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in errors),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
