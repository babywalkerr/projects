from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.config import Settings
from toxicity_moderation.data import load_dataset
from toxicity_moderation.lexicon import ProfanityFilter
from toxicity_moderation.pipeline import RegressionModerator


def score_predictions(pipeline, texts):
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        classes = getattr(pipeline, "classes_", None)
        if classes is not None and 1 in classes:
            return [float(row[list(classes).index(1)]) for row in proba]
        return [float(row[-1]) for row in proba]
    if hasattr(pipeline, "decision_function"):
        return [1.0 / (1.0 + math.exp(-float(value))) for value in pipeline.decision_function(texts)]
    return [min(1.0, max(0.0, float(value))) for value in pipeline.predict(texts)]


def metrics(y_true, y_pred, scores):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_toxic": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_toxic": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_toxic": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)),
        "tn": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)),
        "fp": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)),
        "fn": int(sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        result["roc_auc"] = None
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def error_rows(indices, texts, y_true, y_pred, scores, stages=None, matches=None):
    rows = []
    for idx, text, label, pred, score in zip(indices, texts, y_true, y_pred, scores):
        if label == pred:
            continue
        error_type = "false_positive" if label == 0 and pred == 1 else "false_negative"
        row = {
            "dataset_index": idx,
            "error_type": error_type,
            "label": int(label),
            "prediction": int(pred),
            "score": float(score),
            "text": text,
        }
        if stages is not None:
            row["stage"] = stages[len(rows)] if False else ""
        rows.append(row)
    return rows


def main() -> int:
    import joblib
    from sklearn.model_selection import train_test_split

    output_dir = PROJECT_ROOT / "outputs" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(PROJECT_ROOT / "labeled.csv", text_column="text", label_column="label")
    indices = list(range(len(dataset.texts)))
    x_train_full, x_test, y_train_full, y_test, idx_train, idx_test = train_test_split(
        dataset.texts,
        dataset.labels,
        indices,
        test_size=0.10,
        random_state=42,
        stratify=dataset.labels,
    )

    bundle = joblib.load(PROJECT_ROOT / "models" / "toxicity_model.joblib")
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.5))
    layer2_scores = score_predictions(pipeline, x_test)
    layer2_pred = [1 if score >= threshold else 0 for score in layer2_scores]
    layer2_metrics = metrics(y_test, layer2_pred, layer2_scores)

    layer2_errors = []
    for idx, text, label, pred, score in zip(idx_test, x_test, y_test, layer2_pred, layer2_scores):
        if label == pred:
            continue
        layer2_errors.append(
            {
                "dataset_index": idx,
                "error_type": "false_positive" if label == 0 else "false_negative",
                "label": int(label),
                "prediction": int(pred),
                "score": float(score),
                "threshold": threshold,
                "text": text,
            }
        )

    write_csv(output_dir / "production_layer2_heldout_errors.csv", layer2_errors)
    write_csv(
        output_dir / "production_layer2_false_positives.csv",
        [row for row in layer2_errors if row["error_type"] == "false_positive"],
    )
    write_csv(
        output_dir / "production_layer2_false_negatives.csv",
        [row for row in layer2_errors if row["error_type"] == "false_negative"],
    )

    settings = Settings(llm_provider="disabled")
    lexicon = ProfanityFilter(settings.profanity_paths)
    regression = RegressionModerator(settings.model_path)
    review_threshold = settings.regression_review_threshold

    chain_pred = []
    chain_scores = []
    chain_stages = []
    chain_matches = []
    for text in x_test:
        hit = lexicon.find(text)
        if hit.blocked:
            chain_pred.append(1)
            chain_scores.append(1.0)
            chain_stages.append("layer_1_lexicon")
            chain_matches.append(",".join(hit.matched))
            continue
        score = regression.predict_score(text)
        if score >= (regression.threshold or threshold):
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

    chain_metrics = metrics(y_test, chain_pred, chain_scores)
    stage_counts = {}
    for stage in chain_stages:
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    chain_metrics["stage_counts"] = stage_counts

    chain_errors = []
    for idx, text, label, pred, score, stage, matched in zip(
        idx_test, x_test, y_test, chain_pred, chain_scores, chain_stages, chain_matches
    ):
        if label == pred:
            continue
        chain_errors.append(
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

    write_csv(output_dir / "chain_no_llm_heldout_errors.csv", chain_errors)
    write_csv(
        output_dir / "chain_no_llm_false_positives.csv",
        [row for row in chain_errors if row["error_type"] == "false_positive"],
    )
    write_csv(
        output_dir / "chain_no_llm_false_negatives.csv",
        [row for row in chain_errors if row["error_type"] == "false_negative"],
    )

    production_report = json.loads((PROJECT_ROOT / "outputs" / "full_production_report.json").read_text(encoding="utf-8"))
    full_chain_report = json.loads(
        (PROJECT_ROOT / "outputs" / "chain_metrics_full_no_llm.json").read_text(encoding="utf-8")
    )
    experiments_18 = list(csv.DictReader((PROJECT_ROOT / "outputs" / "results_18_completed.csv").open(encoding="utf-8")))
    best_18 = max(experiments_18, key=lambda row: float(row["f1_toxic"]))

    summary = {
        "best_18_experiment": {
            "vectorizer": best_18["vectorizer"],
            "model": best_18["model"],
            "f1_toxic": float(best_18["f1_toxic"]),
            "roc_auc": float(best_18["roc_auc"]),
            "dataset": "first 12000 rows, 90/10 split, fixed threshold 0.5",
        },
        "production_layer2_heldout_recomputed": {
            **layer2_metrics,
            "threshold": threshold,
            "test_rows": len(x_test),
            "dataset": "held-out 10% of full labeled.csv, same split as production training",
        },
        "chain_no_llm_heldout_recomputed": {
            **chain_metrics,
            "test_rows": len(x_test),
            "dataset": "held-out 10% of full labeled.csv, lexicon + layer2, LLM disabled",
        },
        "chain_no_llm_full_dataset_previous": {
            **full_chain_report["metrics"],
            "items": full_chain_report["items"],
            "warning": "This was evaluated on the whole labeled.csv, including train rows; use held-out metrics for fair quality.",
        },
        "production_report_file_metrics": production_report,
        "error_files": {
            "production_layer2_all": "outputs/error_analysis/production_layer2_heldout_errors.csv",
            "production_layer2_fp": "outputs/error_analysis/production_layer2_false_positives.csv",
            "production_layer2_fn": "outputs/error_analysis/production_layer2_false_negatives.csv",
            "chain_no_llm_all": "outputs/error_analysis/chain_no_llm_heldout_errors.csv",
            "chain_no_llm_fp": "outputs/error_analysis/chain_no_llm_false_positives.csv",
            "chain_no_llm_fn": "outputs/error_analysis/chain_no_llm_false_negatives.csv",
        },
    }
    (output_dir / "metric_discrepancy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = f"""# Разбор расхождения метрик

## Почему 0.64 и 0.95/0.97 не совпадают

Это были разные протоколы оценки.

| Что сравнивалось | Данные | Модель / цепочка | F1 toxic | Комментарий |
|---|---|---|---:|---|
| 18 экспериментов | первые 12 000 строк, test 10% | лучшая строка: {best_18['vectorizer']} + {best_18['model']} | {float(best_18['f1_toxic']):.4f} | Сетка из задания: 6 регрессоров x 3 векторизации, fixed threshold 0.5 |
| Production layer 2 | held-out 10% полного датасета | word+char + LogisticRegression | {layer2_metrics['f1_toxic']:.4f} | Честный test split, threshold {threshold:.2f} |
| Chain no LLM | held-out 10% полного датасета | словарь + production layer 2 | {chain_metrics['f1_toxic']:.4f} | Честный test split без LLM |
| Старый full chain файл | весь `labeled.csv` | словарь + production layer 2 | {full_chain_report['metrics']['f1_toxic']:.4f} | Не held-out: содержит train rows, поэтому это нельзя напрямую сравнивать с test F1 |

Главный вывод: скачок не означает, что одна и та же модель внезапно выросла с 0.64 до 0.95. `0.64` относится к лучшей строке из обязательной сетки регрессоров на 12k-эксперименте. Production-качество относится к другой модели, другому размеру датасета и другому threshold.

## Ошибки production layer 2 на held-out test

| Тип ошибки | Количество |
|---|---:|
| False positive | {layer2_metrics['fp']} |
| False negative | {layer2_metrics['fn']} |
| Всего ошибок | {layer2_metrics['fp'] + layer2_metrics['fn']} |

Файл со всеми ошибками: `outputs/error_analysis/production_layer2_heldout_errors.csv`

## Ошибки цепочки без LLM на held-out test

| Тип ошибки | Количество |
|---|---:|
| False positive | {chain_metrics['fp']} |
| False negative | {chain_metrics['fn']} |
| Всего ошибок | {chain_metrics['fp'] + chain_metrics['fn']} |

Файл со всеми ошибками: `outputs/error_analysis/chain_no_llm_heldout_errors.csv`
"""
    (output_dir / "metric_discrepancy_explanation.md").write_text(md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
