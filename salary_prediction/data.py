"""Data utilities: synthetic data generation and loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


EDUCATION_LEVELS = ["High School", "Bachelor's", "Master's", "PhD"]
JOB_TITLES = [
    "Software Engineer",
    "Data Scientist",
    "ML Engineer",
    "Backend Developer",
    "Frontend Developer",
    "DevOps Engineer",
]
COUNTRIES = ["United States", "Canada", "Germany", "India", "United Kingdom", "Australia"]
EMPLOYMENT_TYPES = ["Full-time", "Contract", "Intern"]
REMOTE_RATIO_VALUES = ["On-site", "Hybrid", "Remote"]


@dataclass
class Dataset:
    features: pd.DataFrame
    target: pd.Series


def _generate_base_salary(job_title: str) -> float:
    base_map = {
        "Software Engineer": 105_000,
        "Data Scientist": 115_000,
        "ML Engineer": 130_000,
        "Backend Developer": 110_000,
        "Frontend Developer": 100_000,
        "DevOps Engineer": 120_000,
    }
    return base_map.get(job_title, 100_000)


def _education_multiplier(education_level: str) -> float:
    return {
        "High School": 0.9,
        "Bachelor's": 1.0,
        "Master's": 1.08,
        "PhD": 1.15,
    }[education_level]


def _remote_multiplier(remote_ratio: str) -> float:
    return {
        "On-site": 0.98,
        "Hybrid": 1.0,
        "Remote": 1.05,
    }[remote_ratio]


def _country_adjustment(country: str) -> float:
    return {
        "United States": 1.0,
        "Canada": 0.9,
        "Germany": 0.95,
        "India": 0.45,
        "United Kingdom": 0.88,
        "Australia": 0.92,
    }[country]


def _emp_type_multiplier(emp_type: str) -> float:
    return {
        "Full-time": 1.0,
        "Contract": 1.12,
        "Intern": 0.45,
    }[emp_type]


def generate_synthetic_salary_data(
    n_samples: int = 600,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a reproducible synthetic salary dataset."""

    rng = np.random.default_rng(seed)

    experience_years = rng.uniform(0, 20, size=n_samples).round(2)
    years_with_company = np.clip(experience_years - rng.uniform(0, 5, size=n_samples), 0, None)
    company_size = rng.choice([50, 200, 1000, 5000, 10_000], size=n_samples, p=[0.15, 0.25, 0.35, 0.2, 0.05])
    age = np.clip(experience_years + rng.normal(25, 3, size=n_samples), 20, 65).round(1)
    education_level = rng.choice(EDUCATION_LEVELS, size=n_samples, p=[0.15, 0.4, 0.3, 0.15])
    job_title = rng.choice(JOB_TITLES, size=n_samples)
    country = rng.choice(COUNTRIES, size=n_samples, p=[0.45, 0.1, 0.12, 0.2, 0.08, 0.05])
    employment_type = rng.choice(EMPLOYMENT_TYPES, size=n_samples, p=[0.85, 0.1, 0.05])
    remote_ratio = rng.choice(REMOTE_RATIO_VALUES, size=n_samples, p=[0.45, 0.3, 0.25])

    salary = []
    for idx in range(n_samples):
        base = _generate_base_salary(job_title[idx])
        edu_mul = _education_multiplier(education_level[idx])
        remote_mul = _remote_multiplier(remote_ratio[idx])
        country_mul = _country_adjustment(country[idx])
        emp_mul = _emp_type_multiplier(employment_type[idx])
        experience_bonus = 2_800 * experience_years[idx]
        tenure_bonus = 900 * years_with_company[idx]
        company_adjustment = (np.log1p(company_size[idx]) - 4.0) * 5_000
        age_penalty = -200 * max(age[idx] - (experience_years[idx] + 30), 0)
        noise = rng.normal(0, 4_000)

        annual_salary = (
            base * edu_mul * remote_mul * country_mul * emp_mul
            + experience_bonus
            + tenure_bonus
            + company_adjustment
            + age_penalty
            + noise
        )

        salary.append(max(40_000, annual_salary))

    df = pd.DataFrame(
        {
            "experience_years": experience_years,
            "years_with_company": years_with_company,
            "company_size": company_size,
            "age": age,
            "education_level": education_level,
            "job_title": job_title,
            "country": country,
            "employment_type": employment_type,
            "remote_ratio": remote_ratio,
            "salary": np.round(salary, -2),
        }
    )

    return df


def ensure_dataset(path: Path = config.PROCESSED_DATA_PATH) -> Path:
    """Ensure the processed dataset exists on disk, generating it if necessary."""

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        df = generate_synthetic_salary_data()
        df.to_csv(path, index=False)

    return path


def load_dataset(path: Path = config.PROCESSED_DATA_PATH) -> Dataset:
    """Load the dataset, generating it first if needed."""

    csv_path = ensure_dataset(path)
    data = pd.read_csv(csv_path)
    features = data.drop(columns=[config.TARGET_COLUMN])
    target = data[config.TARGET_COLUMN]
    return Dataset(features=features, target=target)


def train_test_split(
    dataset: Dataset,
    test_size: float = 0.2,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split as sk_split

    return sk_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

