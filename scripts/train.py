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

from toxicity_moderation.data import load_regression_dataset, profile_csv
from toxicity_moderation.normalization import tokenize
from toxicity_moderation.text_features import preprocess_text

REGRESSOR_NAMES = (
    "SVR",
    "SGDRegressor",
    "KNeighborsRegressor",
    "GaussianProcessRegressor",
    "PLSRegression",
    "DecisionTreeRegressor",
)


def require_sklearn():
    try:
        import joblib
        import numpy as np
        import sklearn
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        from sklearn.linear_model import SGDRegressor
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
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
    except ModuleNotFoundError as exc:
        print(
            "Не хватает библиотек для обучения. Установи зависимости:\n"
            "  python -m pip install -r requirements.txt\n"
            "  python -m pip install -e .",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    return {
        "joblib": joblib,
        "np": np,
        "sklearn": sklearn,
        "BaseEstimator": BaseEstimator,
        "TransformerMixin": TransformerMixin,
        "PLSRegression": PLSRegression,
        "TruncatedSVD": TruncatedSVD,
        "TfidfVectorizer": TfidfVectorizer,
        "GaussianProcessRegressor": GaussianProcessRegressor,
        "DotProduct": DotProduct,
        "WhiteKernel": WhiteKernel,
        "SGDRegressor": SGDRegressor,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
        "KNeighborsRegressor": KNeighborsRegressor,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
        "DecisionTreeRegressor": DecisionTreeRegressor,
    }


def library_versions(skl: dict[str, Any]) -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": skl["np"].__version__,
        "scikit-learn": skl["sklearn"].__version__,
        "joblib": skl["joblib"].__version__,
    }
    for module_name in ("matplotlib", "fasttext", "torch", "transformers"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "installed")
        except ModuleNotFoundError:
            versions[module_name] = "not installed"
    return versions


def build_fasttext_vectorizer(skl: dict[str, Any], model_path: str):
    BaseEstimator = skl["BaseEstimator"]
    TransformerMixin = skl["TransformerMixin"]
    np = skl["np"]

    class FastTextVectorizer(BaseEstimator, TransformerMixin):
        def __init__(self, path: str):
            self.path = path
            self.model = None

        def fit(self, texts, y=None):
            try:
                import fasttext
            except ModuleNotFoundError as exc:
                raise RuntimeError("fasttext is not installed. Install requirements-optional.txt") from exc
            self.model = fasttext.load_model(self.path)
            return self

        def transform(self, texts):
            if self.model is None:
                self.fit(texts)
            rows = []
            dim = self.model.get_dimension()
            for text in texts:
                words = tokenize(preprocess_text(text))
                if not words:
                    rows.append(np.zeros(dim))
                    continue
                vectors = [self.model.get_word_vector(word) for word in words[:256]]
                rows.append(np.mean(vectors, axis=0))
            return np.vstack(rows)

    return FastTextVectorizer(model_path)


def build_rubert_vectorizer(skl: dict[str, Any], model_name: str = "cointegrated/rubert-tiny2"):
    BaseEstimator = skl["BaseEstimator"]
    TransformerMixin = skl["TransformerMixin"]
    np = skl["np"]

    class RuBertTinyVectorizer(BaseEstimator, TransformerMixin):
        def __init__(self, model_name: str = model_name, batch_size: int = 16, max_length: int = 256):
            self.model_name = model_name
            self.batch_size = batch_size
            self.max_length = max_length
            self.tokenizer = None
            self.model = None
            self.device = None

        def fit(self, texts, y=None):
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise RuntimeError("torch/transformers are not installed. Install requirements-optional.txt") from exc
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            return self

        def transform(self, texts):
            if self.model is None or self.tokenizer is None:
                self.fit(texts)
            outputs = []
            with self.torch.no_grad():
                for start in range(0, len(texts), self.batch_size):
                    batch = [preprocess_text(text) for text in texts[start : start + self.batch_size]]
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    model_out = self.model(**encoded)
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (model_out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    outputs.append(pooled.cpu().numpy())
            return np.vstack(outputs)

    return RuBertTinyVectorizer(model_name=model_name)


def make_regressor(name: str, skl: dict[str, Any], args):
    if name == "SVR":
        return skl["SVR"](kernel="linear", C=1.0)
    if name == "SGDRegressor":
        return skl["SGDRegressor"](
            loss="squared_error",
            penalty="elasticnet",
            alpha=0.0001,
            max_iter=2000,
            tol=1e-3,
            random_state=42,
        )
    if name == "KNeighborsRegressor":
        return skl["KNeighborsRegressor"](n_neighbors=args.knn_neighbors, weights="distance")
    if name == "GaussianProcessRegressor":
        return skl["GaussianProcessRegressor"](
            kernel=skl["DotProduct"]() + skl["WhiteKernel"](),
            random_state=42,
        )
    if name == "PLSRegression":
        return skl["PLSRegression"](n_components=2)
    if name == "DecisionTreeRegressor":
        return skl["DecisionTreeRegressor"](max_depth=args.tree_depth, random_state=42)
    raise ValueError(f"Unknown regressor: {name}")


def needs_dense_features(regressor_name: str) -> bool:
    return regressor_name in {"KNeighborsRegressor", "GaussianProcessRegressor", "PLSRegression"}


def needs_scaling(regressor_name: str) -> bool:
    return regressor_name in {"SVR", "SGDRegressor", "KNeighborsRegressor", "GaussianProcessRegressor", "PLSRegression"}


def build_pipeline(
    skl: dict[str, Any],
    *,
    vectorizer_name: str,
    vectorizer,
    vectorizer_kind: str,
    regressor_name: str,
    train_size: int,
    args,
):
    Pipeline = skl["Pipeline"]
    steps = [("vectorizer", vectorizer)]
    notes: list[str] = []

    if vectorizer_kind == "sparse" and needs_dense_features(regressor_name):
        n_components = min(args.svd_components, max(1, train_size - 2))
        steps.append(("svd", skl["TruncatedSVD"](n_components=n_components, random_state=42)))
        notes.append(f"TF-IDF reduced with TruncatedSVD({n_components})")

    if vectorizer_kind == "dense" and needs_scaling(regressor_name):
        steps.append(("scaler", skl["StandardScaler"]()))
        notes.append("StandardScaler used")
    elif vectorizer_kind == "sparse" and needs_dense_features(regressor_name) and needs_scaling(regressor_name):
        steps.append(("scaler", skl["StandardScaler"]()))
        notes.append("StandardScaler used after SVD")

    steps.append(("regressor", make_regressor(regressor_name, skl, args)))
    return Pipeline(steps), "; ".join(notes) if notes else "-"


def available_vectorizers(args, skl: dict[str, Any]):
    TfidfVectorizer = skl["TfidfVectorizer"]
    vectorizers = [
        (
            "TF-IDF",
            TfidfVectorizer(
                preprocessor=preprocess_text,
                analyzer="word",
                ngram_range=(1, 2),
                min_df=args.min_df,
                max_features=args.max_features,
                sublinear_tf=True,
            ),
            "sparse",
            None,
        )
    ]

    skipped = []
    if args.fasttext_model:
        vectorizers.append(("FastText", build_fasttext_vectorizer(skl, args.fasttext_model), "dense", None))
    else:
        skipped.append(("FastText", "No --fasttext-model path was provided"))

    if args.enable_rubert:
        vectorizers.append(("RuBERT-tiny2", build_rubert_vectorizer(skl), "dense", None))
    else:
        skipped.append(("RuBERT-tiny2", "Run with --enable-rubert to download/use the embedding model"))

    return vectorizers, skipped


def clip_predictions(np, predictions, target_min: float, target_max: float, enabled: bool):
    raw = np.ravel(predictions).astype(float)
    if not enabled:
        return raw.tolist()
    return np.clip(raw, target_min, target_max).tolist()


def optional_binary_metrics(skl, y_true, y_pred_scores, toxic_threshold: float) -> dict[str, float | None]:
    unique = sorted(set(y_true))
    if not set(unique).issubset({0.0, 1.0}):
        return {
            "accuracy": None,
            "precision_toxic": None,
            "recall_toxic": None,
            "f1_toxic": None,
            "roc_auc": None,
        }
    y_true_int = [int(value) for value in y_true]
    y_pred_int = [1 if score >= toxic_threshold else 0 for score in y_pred_scores]
    try:
        roc_auc = float(skl["roc_auc_score"](y_true_int, y_pred_scores))
    except ValueError:
        roc_auc = None
    return {
        "accuracy": float(skl["accuracy_score"](y_true_int, y_pred_int)),
        "precision_toxic": float(skl["precision_score"](y_true_int, y_pred_int, zero_division=0)),
        "recall_toxic": float(skl["recall_score"](y_true_int, y_pred_int, zero_division=0)),
        "f1_toxic": float(skl["f1_score"](y_true_int, y_pred_int, zero_division=0)),
        "roc_auc": roc_auc,
    }


def rounded_accuracy_if_discrete(skl, y_true, y_pred) -> float | None:
    unique = sorted(set(y_true))
    if len(unique) > 20:
        return None
    if any(abs(value - round(value)) > 1e-9 for value in unique):
        return None
    allowed = unique
    rounded = []
    for score in y_pred:
        nearest = min(allowed, key=lambda value: abs(value - score))
        rounded.append(nearest)
    return float(skl["accuracy_score"](y_true, rounded))


def evaluate_candidate(
    skl,
    pipeline,
    x_train,
    y_train,
    x_test,
    y_test,
    *,
    target_min: float,
    target_max: float,
    clip: bool,
    toxic_threshold: float,
):
    np = skl["np"]
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    raw_predictions = pipeline.predict(x_test)
    inference_seconds = time.perf_counter() - start

    predictions = clip_predictions(np, raw_predictions, target_min, target_max, clip)
    mae = float(skl["mean_absolute_error"](y_test, predictions))
    rmse = float(math.sqrt(skl["mean_squared_error"](y_test, predictions)))
    r2 = float(skl["r2_score"](y_test, predictions))
    binary = optional_binary_metrics(skl, y_test, predictions, toxic_threshold)
    rounded_accuracy = rounded_accuracy_if_discrete(skl, y_test, predictions)

    return {
        "predictions": predictions,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "rounded_accuracy": rounded_accuracy,
            **binary,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def generate_report(
    *,
    path: Path,
    profile: dict[str, Any],
    versions: dict[str, str],
    results: list[dict[str, Any]],
    best: dict[str, Any],
    examples: list[dict[str, Any]],
    clip: bool,
) -> None:
    ok_results = [row for row in results if row["status"] == "ok"]
    sorted_results = sorted(ok_results, key=lambda row: row["mae"])

    lines = [
        "# Отчет по регрессионной модели токсичности",
        "",
        "## Данные",
        "",
        f"- Файл: `{profile['path']}`",
        f"- Строк: `{profile['rows']}`",
        f"- Колонки: `{', '.join(profile['columns'])}`",
        f"- Текст: `{profile['detected_text_column']}`",
        f"- Target: `{profile['detected_target_column']}`",
        f"- Дубликаты текста: `{profile['duplicate_texts']}`",
        f"- Диапазон target: `{profile['target_min']}` - `{profile['target_max']}`",
        f"- Уникальных target: `{profile['target_unique_count']}`",
        f"- Clip предсказаний: `{'да' if clip else 'нет'}`",
        "",
        "## Библиотеки",
        "",
    ]
    lines.extend(f"- `{name}`: `{version}`" for name, version in versions.items())
    lines.extend(
        [
            "",
            "## Результаты",
            "",
            "| № | Векторизатор | Регрессионная модель | MAE ↓ | RMSE ↓ | R² ↑ | Время обучения, сек | Время инференса, сек | Примечания |",
            "| - | ------------ | -------------------- | ----: | -----: | ---: | ------------------: | -------------------: | ---------- |",
        ]
    )
    for index, row in enumerate(sorted_results, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    markdown_cell(row["vectorizer"]),
                    markdown_cell(row["model"]),
                    markdown_cell(row["mae"]),
                    markdown_cell(row["rmse"]),
                    markdown_cell(row["r2"]),
                    markdown_cell(row["train_seconds"]),
                    markdown_cell(row["inference_seconds"]),
                    markdown_cell(row["notes"]),
                ]
            )
            + " |"
        )

    failed = [row for row in results if row["status"] != "ok"]
    if failed:
        lines.extend(["", "## Пропущенные или неуспешные эксперименты", ""])
        for row in failed:
            lines.append(f"- `{row['vectorizer']} + {row['model']}`: {row.get('notes') or row.get('error')}")

    lines.extend(
        [
            "",
            "## Лучшая комбинация",
            "",
            f"- Векторизатор: `{best['vectorizer_name']}`",
            f"- Модель: `{best['model_name']}`",
            f"- MAE: `{best['metrics']['mae']:.4f}`",
            f"- RMSE: `{best['metrics']['rmse']:.4f}`",
            f"- R²: `{best['metrics']['r2']:.4f}`",
            "",
            "## Примеры с тестовой выборки",
            "",
            "| Комментарий | Реальная токсичность | Предсказанная токсичность | Абсолютная ошибка |",
            "| ----------- | -------------------: | ------------------------: | ----------------: |",
        ]
    )
    for example in examples:
        comment = markdown_cell(example["text"])
        if len(comment) > 220:
            comment = comment[:217] + "..."
        lines.append(
            f"| {comment} | {example['target']:.4f} | {example['prediction']:.4f} | {example['absolute_error']:.4f} |"
        )

    vectorizer_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for row in sorted_results[: max(1, min(5, len(sorted_results)))]:
        vectorizer_counts[row["vectorizer"]] = vectorizer_counts.get(row["vectorizer"], 0) + 1
        model_counts[row["model"]] = model_counts.get(row["model"], 0) + 1
    strongest_vectorizer = max(vectorizer_counts, key=vectorizer_counts.get) if vectorizer_counts else "-"
    strongest_model = max(model_counts, key=model_counts.get) if model_counts else "-"

    slow = sorted(ok_results, key=lambda row: row["train_seconds"] + row["inference_seconds"], reverse=True)[:3]
    slow_names = ", ".join(f"{row['vectorizer']} + {row['model']}" for row in slow) or "-"
    lines.extend(
        [
            "",
            "## Краткий вывод",
            "",
            f"- Самый сильный векторизатор среди верхних результатов: `{strongest_vectorizer}`.",
            f"- Самая сильная модель среди верхних результатов: `{strongest_model}`.",
            f"- Самые медленные комбинации: {slow_names}.",
            "- Лучшая модель, вероятно, выиграла за счет баланса между устойчивостью признаков и простой регуляризованной регрессией.",
            "- Дальше стоит попробовать калибровку порога токсичности, больше реальных данных, отдельную multi-label классификацию типов нарушений и LLM только для спорных случаев.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def create_metrics_plot(path: Path, results: list[dict[str, Any]]) -> str | None:
    ok_results = sorted([row for row in results if row["status"] == "ok"], key=lambda row: row["mae"])
    if not ok_results:
        return "No successful results to plot"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        return f"matplotlib is not installed: {exc}"

    labels = [f"{row['vectorizer']}\n{row['model']}" for row in ok_results]
    values = [row["mae"] for row in ok_results]
    height = max(4, 0.38 * len(values))
    plt.figure(figsize=(11, height))
    plt.barh(range(len(values)), values, color="#0f766e")
    plt.yticks(range(len(values)), labels, fontsize=8)
    plt.xlabel("MAE")
    plt.title("Сравнение MAE регрессионных моделей")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and compare Russian toxicity regression models.")
    parser.add_argument("--data", default=None, help="CSV with text and toxicity target columns")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--label-column", default=None, help="Alias for --target-column")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--svd-components", type=int, default=200)
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--tree-depth", type=int, default=16)
    parser.add_argument("--max-gpr-rows", type=int, default=1500)
    parser.add_argument("--fasttext-model", default=None)
    parser.add_argument("--enable-rubert", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--toxic-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models" / "toxicity_model.joblib"))
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else PROJECT_ROOT / "labeled.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "sample_comments.csv"

    target_column = args.target_column or args.label_column
    skl = require_sklearn()
    versions = library_versions(skl)
    print("Libraries:")
    for name, version in versions.items():
        print(f"  {name}: {version}")

    profile = profile_csv(data_path, text_column=args.text_column, target_column=target_column)
    dataset = load_regression_dataset(data_path, text_column=args.text_column, target_column=target_column)
    target_min = min(dataset.targets)
    target_max = max(dataset.targets)
    clip_enabled = not args.no_clip

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_test_split = skl["train_test_split"]
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.texts,
        dataset.targets,
        test_size=args.test_size,
        random_state=42,
        shuffle=True,
    )

    vectorizers, skipped_vectorizers = available_vectorizers(args, skl)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for vectorizer_name, reason in skipped_vectorizers:
        for regressor_name in REGRESSOR_NAMES:
            results.append(
                {
                    "vectorizer": vectorizer_name,
                    "model": regressor_name,
                    "status": "skipped",
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "train_seconds": None,
                    "inference_seconds": None,
                    "notes": reason,
                }
            )

    for vectorizer_name, vectorizer, vectorizer_kind, _ in vectorizers:
        for regressor_name in REGRESSOR_NAMES:
            if regressor_name == "GaussianProcessRegressor" and len(x_train) > args.max_gpr_rows:
                results.append(
                    {
                        "vectorizer": vectorizer_name,
                        "model": regressor_name,
                        "status": "skipped",
                        "mae": None,
                        "rmse": None,
                        "r2": None,
                        "train_seconds": None,
                        "inference_seconds": None,
                        "notes": f"Skipped because train size {len(x_train)} > --max-gpr-rows {args.max_gpr_rows}",
                    }
                )
                continue

            print(f"Training {vectorizer_name} + {regressor_name}...")
            try:
                pipeline, notes = build_pipeline(
                    skl,
                    vectorizer_name=vectorizer_name,
                    vectorizer=vectorizer,
                    vectorizer_kind=vectorizer_kind,
                    regressor_name=regressor_name,
                    train_size=len(x_train),
                    args=args,
                )
                evaluation = evaluate_candidate(
                    skl,
                    pipeline,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    target_min=target_min,
                    target_max=target_max,
                    clip=clip_enabled,
                    toxic_threshold=args.toxic_threshold,
                )
            except Exception as exc:
                results.append(
                    {
                        "vectorizer": vectorizer_name,
                        "model": regressor_name,
                        "status": "failed",
                        "mae": None,
                        "rmse": None,
                        "r2": None,
                        "train_seconds": None,
                        "inference_seconds": None,
                        "notes": str(exc),
                    }
                )
                print(f"  failed: {exc}")
                continue

            metrics = evaluation["metrics"]
            row = {
                "vectorizer": vectorizer_name,
                "model": regressor_name,
                "status": "ok",
                **metrics,
                "notes": notes,
            }
            results.append(row)
            print(
                f"  MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                f"R2={metrics['r2']:.4f}, train={metrics['train_seconds']:.2f}s, "
                f"infer={metrics['inference_seconds']:.2f}s"
            )

            if best is None or metrics["mae"] < best["metrics"]["mae"]:
                best = {
                    "vectorizer_name": vectorizer_name,
                    "model_name": regressor_name,
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "predictions": evaluation["predictions"],
                    "notes": notes,
                }

    if best is None:
        raise SystemExit("No model was trained successfully.")

    sorted_results = sorted(results, key=lambda row: (row["status"] != "ok", row["mae"] if row["mae"] is not None else math.inf))
    write_csv(output_dir / "results.csv", sorted_results)
    (output_dir / "results.json").write_text(json.dumps(sorted_results, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_rows = []
    examples = []
    for text, target, prediction in zip(x_test, y_test, best["predictions"]):
        absolute_error = abs(float(target) - float(prediction))
        row = {
            "text": text,
            "target": target,
            "prediction": prediction,
            "absolute_error": absolute_error,
        }
        prediction_rows.append(row)
    write_csv(output_dir / "test_predictions.csv", prediction_rows)
    examples = sorted(prediction_rows, key=lambda row: row["absolute_error"], reverse=True)[:5]

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best["pipeline"],
        "threshold": args.toxic_threshold,
        "metadata": {
            "vectorizer_name": best["vectorizer_name"],
            "model_name": best["model_name"],
            "metrics": best["metrics"],
            "target_min": target_min,
            "target_max": target_max,
            "clip_predictions": clip_enabled,
            "notes": best["notes"],
            "trained_on": str(data_path.resolve()),
        },
    }
    skl["joblib"].dump(bundle, model_path)
    skl["joblib"].dump(bundle, output_dir / "best_model.joblib")

    plot_error = create_metrics_plot(output_dir / "metrics_plot.png", sorted_results)
    if plot_error:
        print(f"Plot warning: {plot_error}")

    generate_report(
        path=output_dir / "report.md",
        profile=profile,
        versions=versions,
        results=sorted_results,
        best=best,
        examples=examples,
        clip=clip_enabled,
    )

    print("\nBest:")
    print(f"  vectorizer={best['vectorizer_name']}")
    print(f"  model={best['model_name']}")
    print(f"  MAE={best['metrics']['mae']:.4f}")
    print(f"  saved_for_api={model_path}")
    print(f"  report={output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
