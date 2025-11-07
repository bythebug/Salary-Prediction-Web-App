"""Configuration values for the salary prediction project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "salary_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "salary_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


RANDOM_SEED = 42
TARGET_COLUMN = "salary"


NUMERIC_FEATURES = [
    "experience_years",
    "years_with_company",
    "company_size",
    "age",
]

CATEGORICAL_FEATURES = [
    "education_level",
    "job_title",
    "country",
    "employment_type",
    "remote_ratio",
]

