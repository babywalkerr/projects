from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .lexicon import ProfanityFilter
from .llm import build_llm
from .taxonomy import guess_violation_type


def _clamp_score(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class RegressionModerator:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.available = False
        self.pipeline = None
        self.threshold = None
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.error = f"Model file not found: {self.model_path}"
            return
        try:
            import joblib

            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict):
                self.pipeline = bundle["pipeline"]
                self.threshold = float(bundle.get("threshold", 0.5))
                self.metadata = dict(bundle.get("metadata", {}))
            else:
                self.pipeline = bundle
                self.threshold = 0.5
            self.available = True
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.error = str(exc)

    def predict_score(self, text: str) -> float:
        if not self.available or self.pipeline is None:
            return 0.0
        raw = self._predict_raw_score(text)
        target_min = self.metadata.get("target_min")
        target_max = self.metadata.get("target_max")
        if target_min is not None and target_max is not None and float(target_max) > float(target_min):
            raw = (raw - float(target_min)) / (float(target_max) - float(target_min))
        return _clamp_score(raw)

    def _predict_raw_score(self, text: str) -> float:
        if self.pipeline is None:
            return 0.0
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba([text])[0]
            classes = getattr(self.pipeline, "classes_", None)
            if classes is not None and 1 in classes:
                return float(proba[list(classes).index(1)])
            return float(proba[-1])
        if hasattr(self.pipeline, "decision_function"):
            import math

            margin = float(self.pipeline.decision_function([text])[0])
            return 1.0 / (1.0 + math.exp(-margin))
        raw = self.pipeline.predict([text])[0]
        try:
            return float(raw)
        except TypeError:
            return float(raw[0])


class ModerationPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.lexicon = ProfanityFilter(self.settings.profanity_paths)
        self.regression = RegressionModerator(self.settings.model_path)
        self.llm = build_llm(self.settings)

    def moderate(self, text: str) -> dict[str, Any]:
        layers: list[dict[str, Any]] = []

        lexicon_hit = self.lexicon.find(text, with_spans=True)
        if lexicon_hit.blocked:
            violation_type = guess_violation_type(text, lexicon_hit)
            layers.append(
                {
                    "layer": 1,
                    "name": "lexicon",
                    "action": "block",
                    "confidence": lexicon_hit.confidence,
                    "matched": lexicon_hit.matched,
                    "matched_spans": lexicon_hit.matched_spans or [],
                }
            )
            return self._result(
                blocked=True,
                final_label=1,
                confidence=1.0,
                stage="layer_1_lexicon",
                violation_type=violation_type,
                layers=layers,
                reason="Найдено явное словарное нарушение.",
            )

        layers.append({"layer": 1, "name": "lexicon", "action": "pass", "confidence": 1.0, "matched": [], "matched_spans": []})

        regression_score = 0.0
        regression_threshold = self.settings.regression_block_threshold
        if self.regression.available:
            regression_score = self.regression.predict_score(text)
            regression_threshold = self.regression.threshold or regression_threshold
            action = "block" if regression_score >= regression_threshold else "pass"
            layers.append(
                {
                    "layer": 2,
                    "name": "regression",
                    "action": action,
                    "confidence": regression_score,
                    "threshold": regression_threshold,
                    "model": self.regression.metadata.get("model_name", "unknown"),
                    "vectorizer": self.regression.metadata.get("vectorizer_name", "unknown"),
                }
            )
            if action == "block":
                return self._result(
                    blocked=True,
                    final_label=1,
                    confidence=regression_score,
                    stage="layer_2_regression",
                    violation_type=guess_violation_type(text),
                    layers=layers,
                    reason="Регрессионная модель оценила комментарий как токсичный.",
                )
            if regression_score < self.settings.regression_review_threshold:
                return self._result(
                    blocked=False,
                    final_label=0,
                    confidence=1.0 - regression_score,
                    stage="layer_2_low_risk",
                    violation_type="none",
                    layers=layers,
                    reason="Комментарий прошел словарь и получил низкий score токсичности, поэтому LLM не вызывается.",
                )
        else:
            layers.append(
                {
                    "layer": 2,
                    "name": "regression",
                    "action": "skipped",
                    "confidence": 0.0,
                    "error": self.regression.error,
                }
            )

        llm_result = self.llm.moderate(text)
        if llm_result.skipped:
            layers.append(
                {
                    "layer": 3,
                    "name": self.llm.provider,
                    "action": "skipped",
                    "confidence": 0.0,
                    "error": llm_result.error,
                }
            )
            return self._result(
                blocked=False,
                final_label=0,
                confidence=1.0 - regression_score,
                stage="allow_without_llm",
                violation_type="none",
                layers=layers,
                reason="Явных нарушений не найдено; LLM-слой отключен или недоступен.",
            )

        llm_action = "block" if llm_result.toxic else "allow"
        layers.append(
            {
                "layer": 3,
                "name": llm_result.provider,
                "action": llm_action,
                "confidence": llm_result.confidence,
                "violation_type": llm_result.violation_type,
            }
        )
        return self._result(
            blocked=llm_result.toxic,
            final_label=1 if llm_result.toxic else 0,
            confidence=llm_result.confidence,
            stage="layer_3_llm",
            violation_type=llm_result.violation_type if llm_result.toxic else "none",
            layers=layers,
            reason="Итоговый вердикт получен от LLM-слоя.",
        )

    @staticmethod
    def _result(
        *,
        blocked: bool,
        final_label: int,
        confidence: float,
        stage: str,
        violation_type: str,
        layers: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "blocked": blocked,
            "final_label": final_label,
            "confidence": _clamp_score(confidence),
            "stage": stage,
            "violation_type": violation_type,
            "reason": reason,
            "layers": layers,
        }
