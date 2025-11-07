"""CLI entrypoint to train the salary prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from salary_prediction import config, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the salary prediction model")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Regenerate the synthetic dataset before training",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=config.METRICS_PATH,
        help="Optional path to write evaluation metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.refresh_data and config.PROCESSED_DATA_PATH.exists():
        config.PROCESSED_DATA_PATH.unlink()

    artifacts = model.train_pipeline()
    metrics_payload = json.dumps(artifacts.metrics, indent=2)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(metrics_payload, encoding="utf-8")

    print("Model trained successfully. Metrics:")
    print(metrics_payload)
    print(f"Artifacts saved to: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()

