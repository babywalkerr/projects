from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ModuleNotFoundError:
    pass


def _split_paths(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part.strip()) for part in value.split(";") if part.strip()]


def _resolve_path(path: Path) -> Path:
    """Resolve a path against PROJECT_ROOT if it is relative."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    model_path: Path = field(
        default_factory=lambda: _resolve_path(Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "toxicity_model.joblib")))
    )
    regression_block_threshold: float = field(
        default_factory=lambda: float(os.getenv("REGRESSION_BLOCK_THRESHOLD", "0.72"))
    )
    regression_review_threshold: float = field(
        default_factory=lambda: float(os.getenv("REGRESSION_REVIEW_THRESHOLD", "0.35"))
    )
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "disabled").strip().lower())
    openai_moderation_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
    )
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-5.5"))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    )
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-safeguard-20b"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    hf_toxic_model: str = field(
        default_factory=lambda: os.getenv("HF_TOXIC_MODEL", "s-nlp/russian_toxicity_classifier")
    )
    hf_cache_dir: Path = field(default_factory=lambda: _resolve_path(Path(os.getenv("HF_HOME", PROJECT_ROOT / "models" / "huggingface"))))
    gigachat_model: str = field(default_factory=lambda: os.getenv("GIGACHAT_MODEL", "GigaChat"))
    yandex_model: str = field(default_factory=lambda: os.getenv("YANDEX_MODEL", "yandexgpt-lite"))
    yandex_folder_id: str = field(default_factory=lambda: os.getenv("YANDEX_FOLDER_ID", ""))
    compat_model: str = field(default_factory=lambda: os.getenv("COMPAT_MODEL", ""))
    compat_base_url: str = field(default_factory=lambda: os.getenv("COMPAT_BASE_URL", "").rstrip("/"))

    @property
    def profanity_paths(self) -> list[Path]:
        configured = _split_paths(os.getenv("PROFANITY_FILES"))
        defaults = [
            self.data_dir / "profanity_roots_ru.txt",
            self.data_dir / "policy_insult_roots_ru.txt",
            self.data_dir / "profanity_external_ru.txt",
        ]
        return configured + defaults


def get_settings() -> Settings:
    return Settings()
