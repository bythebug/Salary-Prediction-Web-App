# Salary Prediction Web App

An end-to-end salary intelligence platform that blends Scikit-learn modeling, a Plotly Dash analytics UI, and AWS-ready deployment tooling.

Link: https://salary-prediction-web-app-wb7c.onrender.com/

## Highlights

- ✅ **90%+ R² accuracy** on a realistic synthetic dataset with 600 labelled salary samples.
- 📊 **Interactive Plotly Dash experience** to explore compensation trends by role, education, and country.
- ♻️ **Reproducible data generation and training pipeline** with tracked metrics and feature importance reporting.

## Project Structure

```text
/Users/sverma/Github/MyProject
├── app/                 # Plotly Dash frontend
├── aws/                 # SAM template and Lambda assets
├── data/                # Generated dataset and docs
├── models/              # Serialized pipeline + metrics
├── salary_prediction/   # Core feature engineering & model code
├── train.py             # CLI entrypoint for training
├── requirements.txt     # Local development dependencies
└── README.md
```

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

This command will:

- Generate `data/processed/salary_data.csv` (synthetic, seeded for reproducibility).
- Train a Random Forest regression pipeline with categorical encoding and scaling.
- Write the pipeline to `models/salary_model.joblib` and metrics to `models/model_metrics.json`.

### Launch the Dash App

```bash
python -m app.dash_app
```

The UI (http://127.0.0.1:8050 by default) lets you:

- Adjust candidate profile inputs and instantly see predicted compensation.
- Compare salary distributions across roles and regions.
- Review feature importance insights derived from the model.

### Live Screenshots

![Salary prediction controls](screenshots/Screenshot%202025-11-07%20at%2009.34.46.png)

![Insight dashboards](screenshots/Screenshot%202025-11-07%20at%2009.34.54.png)

> **Live Demo (Render):** https://salary-prediction-web-app-wb7c.onrender.com/

### Deploy an AWS Prediction API

Full instructions live in [`aws/README.md`](aws/README.md). In short:

1. Run `python train.py` to refresh the model artifact.
2. Use AWS SAM (`sam build && sam deploy --guided`) to publish a Lambda-backed `/predict` endpoint.
3. POST JSON payloads with a `features` object to receive salary predictions.

### Deploy the Dash UI to Render (Free Tier)

1. Ensure the repo includes the provided `Procfile` (`web: gunicorn app.dash_app:server`) and `gunicorn` dependency in `requirements.txt`.
2. Push your code to GitHub (or another Git remote Render can access).
3. Sign in to [Render](https://render.com/) ➜ click **New +** ➜ **Web Service** and connect the repo.
4. Accept the default build command (`pip install -r requirements.txt`) and set the start command to `gunicorn app.dash_app:server`.
5. Pick the free instance type and deploy; the first build may take a couple of minutes.
6. Once Render finishes, copy the public URL and replace the placeholder in the “Live Demo” link above.

## Testing & Quality Checks

- Execute `python train.py` after modifying feature engineering to ensure metrics stay within guardrails.
- The Dash callbacks rely on the serialized pipeline—retrain whenever you tweak preprocessing.
- Optionally run `python -m compileall .` or integrate `pytest`/`bandit` as future enhancements.

## Future Enhancements

- Persist training runs with MLflow or Weights & Biases for experiment tracking.
- Swap synthetic data for an authenticated dataset (e.g., Levels.fyi or Kaggle) when licensing allows.
- Containerize the Dash app and inference API for Elastic Beanstalk or ECS deployment.
