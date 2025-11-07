"""Model utilities for training, evaluation, and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, data


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    metrics: Dict[str, float]


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    return pipeline


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_percent": mape,
    }


def train_and_evaluate(
    dataset: data.Dataset | None = None,
) -> Tuple[Pipeline, Dict[str, float]]:
    if dataset is None:
        dataset = data.load_dataset()

    X_train, X_test, y_train, y_test = data.train_test_split(dataset)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    return pipeline, metrics


def save_artifacts(artifacts: ModelArtifacts) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.pipeline, config.MODEL_PATH)
    with config.METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(artifacts.metrics, fp, indent=2)


def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not path.exists():
        dataset = data.load_dataset()
        pipeline, metrics = train_and_evaluate(dataset)
        save_artifacts(ModelArtifacts(pipeline=pipeline, metrics=metrics))
    return joblib.load(path)


def train_pipeline() -> ModelArtifacts:
    dataset = data.load_dataset()
    pipeline, metrics = train_and_evaluate(dataset)
    save_artifacts(ModelArtifacts(pipeline=pipeline, metrics=metrics))
    return ModelArtifacts(pipeline=pipeline, metrics=metrics)


def load_metrics(path: Path = config.METRICS_PATH) -> Dict[str, float]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    _, metrics = train_and_evaluate()
    return metrics


def get_grouped_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Return feature importances aggregated by original feature."""

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model_step: RandomForestRegressor = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model_step.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError("Feature names and importances length mismatch")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})

    def aggregate_name(raw_name: str) -> str:
        if raw_name.startswith("num__"):
            return raw_name.split("num__", 1)[1]
        if raw_name.startswith("cat__"):
            group = raw_name.split("cat__", 1)[1]
            return group.split("_", 1)[0]
        return raw_name

    df["group"] = df["feature"].map(aggregate_name)

    grouped = (
        df.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    )

    total = grouped["importance"].sum()
    if total > 0:
        grouped["importance_percent"] = grouped["importance"] / total * 100
    else:
        grouped["importance_percent"] = 0

    return grouped

