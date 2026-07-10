from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toxicity_moderation.data import load_dataset
from toxicity_moderation.text_features import preprocess_text


def require_sklearn():
    try:
        import joblib
        import numpy as np
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ModuleNotFoundError as exc:
        print("Не хватает зависимостей. Установи requirements.txt.", file=sys.stderr)
        raise SystemExit(2) from exc
    return locals()


def make_vectorizer(name: str, skl: dict[str, Any], max_features: int, min_df: int):
    TfidfVectorizer = skl["TfidfVectorizer"]
    if name == "word":
        return TfidfVectorizer(
            preprocessor=preprocess_text,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
    if name == "char":
        return TfidfVectorizer(
            preprocessor=preprocess_text,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )
    if name == "word+char":
        return skl["FeatureUnion"](
            [
                ("word", make_vectorizer("word", skl, max_features // 2, min_df)),
                ("char", make_vectorizer("char", skl, max_features // 2, min_df)),
            ]
        )
    raise ValueError(f"Unknown vectorizer: {name}")


def make_model(name: str, skl: dict[str, Any], max_iter: int):
    if name == "SGDRegressor":
        return skl["SGDRegressor"](
            loss="squared_error",
            penalty="elasticnet",
            alpha=0.00003,
            l1_ratio=0.15,
            max_iter=max_iter,
            tol=1e-3,
            random_state=42,
        )
    if name == "SGDClassifier":
        return skl["SGDClassifier"](
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.00003,
            l1_ratio=0.15,
            max_iter=max_iter,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )
    if name == "LogisticRegression":
        return skl["LogisticRegression"](
            C=2.0,
            solver="saga",
            penalty="l2",
            class_weight="balanced",
            max_iter=max_iter,
            n_jobs=1,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")


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


def find_threshold(y_true, scores, f1_score) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for idx in range(10, 91):
        threshold = idx / 100
        pred = [1 if score >= threshold else 0 for score in scores]
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_threshold = threshold
            best_f1 = float(f1)
    return best_threshold, best_f1


def evaluate(skl, y_true, scores, threshold: float) -> dict[str, float]:
    pred = [1 if score >= threshold else 0 for score in scores]
    return {
        "mae": float(skl["mean_absolute_error"](y_true, scores)),
        "rmse": float(math.sqrt(skl["mean_squared_error"](y_true, scores))),
        "r2": float(skl["r2_score"](y_true, scores)),
        "accuracy": float(skl["accuracy_score"](y_true, pred)),
        "precision_toxic": float(skl["precision_score"](y_true, pred, zero_division=0)),
        "recall_toxic": float(skl["recall_score"](y_true, pred, zero_division=0)),
        "f1_toxic": float(skl["f1_score"](y_true, pred, zero_division=0)),
        "roc_auc": float(skl["roc_auc_score"](y_true, scores)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fast production layer-2 model on the full dataset.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "labeled.csv"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--vectorizers",
        default="word,char,word+char",
        help="Comma-separated vectorizers to train: word,char,word+char",
    )
    parser.add_argument(
        "--models",
        default="SGDRegressor,SGDClassifier,LogisticRegression",
        help="Comma-separated models to train: SGDRegressor,SGDClassifier,LogisticRegression",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "toxicity_model.joblib"))
    args = parser.parse_args()

    skl = require_sklearn()
    dataset = load_dataset(args.data, text_column=args.text_column, label_column=args.label_column)
    train_test_split = skl["train_test_split"]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=0.10,
        random_state=42,
        stratify=dataset.labels,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.10,
        random_state=42,
        stratify=y_train_full,
    )

    vectorizers = [item.strip() for item in args.vectorizers.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for vectorizer_name in vectorizers:
        for model_name in models:
            print(f"Training {vectorizer_name} + {model_name}...")
            pipeline = skl["Pipeline"](
                [
                    ("vectorizer", make_vectorizer(vectorizer_name, skl, args.max_features, args.min_df)),
                    ("model", make_model(model_name, skl, args.max_iter)),
                ]
            )
            start = time.perf_counter()
            try:
                pipeline.fit(x_train, y_train)
                train_seconds = time.perf_counter() - start
                val_scores = score_predictions(pipeline, x_val)
                threshold, _ = find_threshold(y_val, val_scores, skl["f1_score"])
                start = time.perf_counter()
                test_scores = score_predictions(pipeline, x_test)
                inference_seconds = time.perf_counter() - start
                metrics = evaluate(skl, y_test, test_scores, threshold)
                row = {
                    "vectorizer": vectorizer_name,
                    "model": model_name,
                    "status": "ok",
                    "threshold": threshold,
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    **metrics,
                    "notes": "threshold selected on validation split",
                }
            except Exception as exc:
                row = {
                    "vectorizer": vectorizer_name,
                    "model": model_name,
                    "status": "failed",
                    "threshold": None,
                    "train_seconds": None,
                    "inference_seconds": None,
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "accuracy": None,
                    "precision_toxic": None,
                    "recall_toxic": None,
                    "f1_toxic": None,
                    "roc_auc": None,
                    "notes": str(exc),
                }
                rows.append(row)
                print(f"  failed: {exc}")
                continue

            rows.append(row)
            print(
                f"  F1={metrics['f1_toxic']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
                f"MAE={metrics['mae']:.4f}, threshold={threshold:.2f}"
            )
            rank = (metrics["f1_toxic"], metrics["roc_auc"], -metrics["mae"])
            if best is None or rank > best["rank"]:
                best = {
                    "rank": rank,
                    "pipeline": pipeline,
                    "vectorizer": vectorizer_name,
                    "model": model_name,
                    "threshold": threshold,
                    "metrics": metrics,
                }

    if best is None:
        raise SystemExit("No production model trained successfully.")

    # Refit the winning pipeline on all non-test rows after choosing threshold on validation.
    print(f"Refitting best model on full train split: {best['vectorizer']} + {best['model']}")
    final_pipeline = skl["Pipeline"](
        [
            ("vectorizer", make_vectorizer(best["vectorizer"], skl, args.max_features, args.min_df)),
            ("model", make_model(best["model"], skl, args.max_iter)),
        ]
    )
    final_pipeline.fit(x_train_full, y_train_full)
    final_scores = score_predictions(final_pipeline, x_test)
    final_metrics = evaluate(skl, y_test, final_scores, best["threshold"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda item: (item["status"] != "ok", -(item["f1_toxic"] or -1), -(item["roc_auc"] or -1)))
    write_csv(output_dir / "full_production_results.csv", rows_sorted)
    (output_dir / "full_production_results.json").write_text(
        json.dumps(rows_sorted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "dataset_rows": len(dataset.texts),
        "train_rows": len(x_train_full),
        "test_rows": len(x_test),
        "best_vectorizer": best["vectorizer"],
        "best_model": best["model"],
        "threshold": best["threshold"],
        "test_metrics_after_refit": final_metrics,
    }
    (output_dir / "full_production_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_pipeline,
        "threshold": best["threshold"],
        "metadata": {
            "vectorizer_name": best["vectorizer"],
            "model_name": best["model"],
            "model_kind": "production_full",
            "metrics": final_metrics,
            "target_min": 0.0,
            "target_max": 1.0,
            "trained_on": str(Path(args.data).resolve()),
            "dataset_rows": len(dataset.texts),
            "notes": "Full dataset production model; threshold selected on validation split before final refit.",
        },
    }
    skl["joblib"].dump(bundle, model_path)
    skl["joblib"].dump(bundle, output_dir / "best_full_production_model.joblib")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
