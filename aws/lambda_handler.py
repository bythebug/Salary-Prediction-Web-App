"""AWS Lambda entrypoint for salary predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from salary_prediction import config


_PIPELINE_CACHE = None


def _model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)
    return config.MODEL_PATH


def _load_pipeline():
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        _PIPELINE_CACHE = joblib.load(_model_path())
    return _PIPELINE_CACHE


def _predict(features: Dict[str, Any]) -> float:
    pipeline = _load_pipeline()
    frame = pd.DataFrame([features])
    return float(pipeline.predict(frame)[0])


def _success(body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error(message: str, status_code: int = 400) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    if event.get("httpMethod") == "GET":
        return _success({"message": "Salary prediction endpoint is live."})

    if event.get("httpMethod") not in {"POST", None}:
        return _error("Unsupported method", 405)

    body = event.get("body") or event
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return _error("Invalid JSON payload")

    features = body.get("features") if isinstance(body, dict) else None
    if not isinstance(features, dict):
        return _error("Missing 'features' object in request body")

    try:
        prediction = _predict(features)
    except Exception as exc:  # pylint: disable=broad-except
        return _error(f"Failed to score request: {exc}", 500)

    return _success({"prediction": prediction, "currency": "USD"})

