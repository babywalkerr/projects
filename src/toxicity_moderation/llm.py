from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass(frozen=True)
class LLMResult:
    skipped: bool
    toxic: bool = False
    confidence: float = 0.0
    violation_type: str = "none"
    provider: str = "disabled"
    raw: dict[str, Any] | None = None
    error: str | None = None


class DisabledLLM:
    provider = "disabled"

    def moderate(self, text: str) -> LLMResult:
        return LLMResult(skipped=True, provider=self.provider)


class OpenAIModerationClient:
    provider = "openai_moderation"

    def __init__(self, *, model: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def moderate(self, text: str) -> LLMResult:
        if not self.api_key:
            return LLMResult(skipped=True, provider=self.provider, error="OPENAI_API_KEY is not set")

        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/moderations",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return LLMResult(skipped=True, provider=self.provider, error=str(exc))

        result = (data.get("results") or [{}])[0]
        categories = result.get("categories") or {}
        scores = result.get("category_scores") or {}
        toxic = bool(result.get("flagged"))
        confidence = max([float(value) for value in scores.values()] or [0.0])
        violation_type = _map_openai_category(categories, scores)
        return LLMResult(
            skipped=False,
            toxic=toxic,
            confidence=confidence,
            violation_type=violation_type,
            provider=self.provider,
            raw=data,
        )


class OpenAIPromptClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        provider: str = "openai_prompt",
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)

    def moderate(self, text: str) -> LLMResult:
        if not self.api_key:
            return LLMResult(skipped=True, provider=self.provider, error=f"{self.api_key_env} is not set")

        prompt = (
            "Ты модератор русскоязычных комментариев. "
            "Верни только JSON с полями toxic:boolean, confidence:number от 0 до 1, "
            "violation_type:string. Возможные violation_type: none, profanity, insult, threat, "
            "spam, identity_attack_religion, identity_attack_nationality, "
            "identity_attack_social_status, toxicity.\n\n"
            f"Комментарий: {text}"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Отвечай только валидным JSON без Markdown."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            verdict = _extract_json(content)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return LLMResult(skipped=True, provider=self.provider, error=str(exc))

        return LLMResult(
            skipped=False,
            toxic=bool(verdict.get("toxic", False)),
            confidence=float(verdict.get("confidence", 0.0)),
            violation_type=str(verdict.get("violation_type", "toxicity")),
            provider=self.provider,
            raw=data,
        )


class HuggingFaceLocalClassifier:
    provider = "hf_local"

    def __init__(self, *, model: str, cache_dir: str | None = None):
        self.model = model
        self.cache_dir = cache_dir
        self.pipeline = None
        self.error: str | None = None

    def _load(self) -> bool:
        if self.pipeline is not None:
            return True
        try:
            # Force full offline mode to avoid proxy/network errors when
            # the model is already cached locally.
            import os
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            kwargs = {"local_files_only": True}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            tokenizer = AutoTokenizer.from_pretrained(self.model, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(self.model, **kwargs)
            self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)
            return True
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.error = str(exc)
            return False

    def moderate(self, text: str) -> LLMResult:
        if not self._load() or self.pipeline is None:
            return LLMResult(skipped=True, provider=self.provider, error=self.error)
        try:
            raw_result = self.pipeline(text[:5000])
            if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], list):
                raw_result = raw_result[0]
            scores = raw_result if isinstance(raw_result, list) else [raw_result]
            toxic_score = _toxic_score_from_hf(scores)
        except Exception as exc:  # pragma: no cover - runtime model failure
            return LLMResult(skipped=True, provider=self.provider, error=str(exc))

        toxic = toxic_score >= 0.5
        return LLMResult(
            skipped=False,
            toxic=toxic,
            confidence=toxic_score if toxic else 1.0 - toxic_score,
            violation_type="toxicity" if toxic else "none",
            provider=self.provider,
            raw={"model": self.model, "scores": scores},
        )


class YandexGPTClient:
    provider = "yandexgpt"

    def __init__(self, *, model: str, folder_id: str):
        self.model = model
        self.folder_id = folder_id
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.iam_token = os.getenv("YANDEX_IAM_TOKEN")
        self.api_key_env = "YANDEX_API_KEY or YANDEX_IAM_TOKEN"

    def moderate(self, text: str) -> LLMResult:
        if not self.folder_id:
            return LLMResult(skipped=True, provider=self.provider, error="YANDEX_FOLDER_ID is not set")
        if not self.api_key and not self.iam_token:
            return LLMResult(skipped=True, provider=self.provider, error="YANDEX_API_KEY/YANDEX_IAM_TOKEN is not set")

        headers = {"Content-Type": "application/json"}
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        else:
            headers["Authorization"] = f"Api-Key {self.api_key}"

        payload = json.dumps(
            {
                "modelUri": f"gpt://{self.folder_id}/{self.model}/latest",
                "completionOptions": {"stream": False, "temperature": 0, "maxTokens": 200},
                "messages": [
                    {
                        "role": "system",
                        "text": (
                            "Ты модератор русскоязычных комментариев. "
                            "Отвечай только JSON с полями toxic, confidence, violation_type."
                        ),
                    },
                    {"role": "user", "text": f"Комментарий: {text}"},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["result"]["alternatives"][0]["message"]["text"]
            verdict = _extract_json(content)
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return LLMResult(skipped=True, provider=self.provider, error=str(exc))

        return LLMResult(
            skipped=False,
            toxic=bool(verdict.get("toxic", False)),
            confidence=float(verdict.get("confidence", 0.0)),
            violation_type=str(verdict.get("violation_type", "toxicity")),
            provider=self.provider,
            raw=data,
        )


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _toxic_score_from_hf(scores: list[dict[str, Any]]) -> float:
    normalized = []
    for item in scores:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        normalized.append((label, score))

    for label, score in normalized:
        if any(token in label for token in ("toxic", "tox", "insult", "threat", "obscene", "abuse")):
            return min(1.0, max(0.0, score))
    for label, score in normalized:
        if label in {"label_1", "1", "__label__1"}:
            return min(1.0, max(0.0, score))
    for label, score in normalized:
        if any(token in label for token in ("normal", "neutral", "non", "clean", "label_0", "0")):
            return min(1.0, max(0.0, 1.0 - score))
    return min(1.0, max(0.0, max((score for _, score in normalized), default=0.0)))


def _map_openai_category(categories: dict[str, Any], scores: dict[str, Any]) -> str:
    active = [name for name, value in categories.items() if value]
    if active:
        name = active[0]
    elif scores:
        name = max(scores, key=lambda key: float(scores[key]))
    else:
        return "none"

    if name.startswith("harassment"):
        return "insult"
    if name.startswith("hate"):
        return "identity_attack"
    if name.startswith("violence"):
        return "threat"
    if name.startswith("sexual"):
        return "sexual"
    if name.startswith("self-harm"):
        return "self_harm"
    return name.replace("/", "_")


def build_llm(settings: Settings):
    if settings.llm_provider == "openai_moderation":
        return OpenAIModerationClient(model=settings.openai_moderation_model, base_url=settings.openai_base_url)
    if settings.llm_provider == "openai_prompt":
        return OpenAIPromptClient(
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
            api_key_env="OPENAI_API_KEY",
            provider="openai_prompt",
        )
    if settings.llm_provider in {"groq", "groq_prompt"}:
        return OpenAIPromptClient(
            model=settings.groq_model,
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            provider="groq",
        )
    if settings.llm_provider in {"gemini", "gemini_prompt"}:
        return OpenAIPromptClient(
            model=settings.gemini_model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key_env="GEMINI_API_KEY",
            provider="gemini",
        )
    if settings.llm_provider in {"hf_local", "huggingface_local"}:
        return HuggingFaceLocalClassifier(model=settings.hf_toxic_model, cache_dir=str(settings.hf_cache_dir))
    if settings.llm_provider == "gigachat":
        return OpenAIPromptClient(
            model=settings.gigachat_model,
            base_url="https://gigachat.devices.sberbank.ru/api/v1",
            api_key_env="GIGACHAT_ACCESS_TOKEN",
            provider="gigachat",
        )
    if settings.llm_provider == "yandexgpt":
        return YandexGPTClient(model=settings.yandex_model, folder_id=settings.yandex_folder_id)
    if settings.llm_provider in {"openai_compatible", "compatible"} and settings.compat_base_url and settings.compat_model:
        return OpenAIPromptClient(
            model=settings.compat_model,
            base_url=settings.compat_base_url,
            api_key_env="COMPAT_API_KEY",
            provider="openai_compatible",
        )
    return DisabledLLM()
