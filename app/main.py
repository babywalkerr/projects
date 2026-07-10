from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from toxicity_moderation.pipeline import ModerationPipeline


class ModerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class ModerationResponse(BaseModel):
    blocked: bool
    final_label: int
    confidence: float
    stage: str
    violation_type: str
    reason: str
    layers: list[dict[str, Any]]


app = FastAPI(title="Russian Toxicity Moderation API", version="0.1.0")
pipeline = ModerationPipeline()

WEB_DIR = PROJECT_ROOT / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    llm_key_env = getattr(pipeline.llm, "api_key_env", None)
    llm_ready = True
    if llm_key_env:
        llm_ready = bool(getattr(pipeline.llm, "api_key", None))
    return {
        "ok": True,
        "model_loaded": pipeline.regression.available,
        "model_error": pipeline.regression.error,
        "llm_provider": pipeline.llm.provider,
        "llm_ready": llm_ready,
        "llm_key_env": llm_key_env,
    }


@app.get("/api/config")
def config():
    return {
        "regression_block_threshold": pipeline.settings.regression_block_threshold,
        "regression_review_threshold": pipeline.settings.regression_review_threshold,
        "model_path": str(pipeline.settings.model_path),
        "model_loaded": pipeline.regression.available,
        "model_threshold": pipeline.regression.threshold,
        "model_metadata": pipeline.regression.metadata,
        "llm_provider": pipeline.llm.provider,
    }


def _read_json(path: Path, fallback: Any):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


@app.get("/api/metrics")
def metrics():
    return {
        "production": _read_json(PROJECT_ROOT / "outputs" / "full_production_report.json", {}),
        "chain_no_llm": _read_json(PROJECT_ROOT / "outputs" / "chain_metrics_full_no_llm.json", {}),
        "chain_hf_sample": _read_json(PROJECT_ROOT / "outputs" / "chain_metrics_hf_sample_5000.json", {}),
        "dataset_profile": _read_json(PROJECT_ROOT / "outputs" / "dataset_profile.json", {}),
    }


@app.post("/api/moderate", response_model=ModerationResponse)
def moderate(request: ModerationRequest):
    return pipeline.moderate(request.text)
