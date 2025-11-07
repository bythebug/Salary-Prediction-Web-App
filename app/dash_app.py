"""Plotly Dash application for the salary prediction project."""

from __future__ import annotations

from typing import Dict

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html

from salary_prediction import config, data as data_utils, model as model_utils

# Lazily load artifacts once so the Dash callbacks stay fast.
DATASET = data_utils.load_dataset()
PIPELINE = model_utils.load_pipeline()
METRICS = model_utils.load_metrics()
FEATURE_IMPORTANCES = model_utils.get_grouped_feature_importances(PIPELINE)


def _default_feature_values() -> Dict[str, object]:
    numeric_defaults = DATASET.features[config.NUMERIC_FEATURES].median().to_dict()
    categorical_defaults = {
        feature: DATASET.features[feature].mode()[0]
        for feature in config.CATEGORICAL_FEATURES
    }
    return {**numeric_defaults, **categorical_defaults}


DEFAULTS = _default_feature_values()


def _predict(payload: Dict[str, object]) -> float:
    frame = pd.DataFrame([payload])
    prediction = float(PIPELINE.predict(frame)[0])
    return prediction


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _confidence_range(prediction: float, mae: float) -> str:
    lower = max(40_000, prediction - mae)
    upper = prediction + mae
    return f"{_format_currency(lower)} - {_format_currency(upper)}"


DEFAULT_PREDICTION = _predict(DEFAULTS)


def _metrics_cards() -> dbc.Row:
    card_style = {"textAlign": "center"}
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("R²"),
                        dbc.CardBody(html.H4(f"{METRICS.get('r2', 0):.3f}", className="card-title")),
                    ],
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("RMSE"),
                        dbc.CardBody(html.H4(_format_currency(METRICS.get("rmse", 0)), className="card-title")),
                    ],
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("MAE"),
                        dbc.CardBody(html.H4(_format_currency(METRICS.get("mae", 0)), className="card-title")),
                    ],
                    style=card_style,
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("MAPE"),
                        dbc.CardBody(html.H4(f"{METRICS.get('mape_percent', 0):.2f}%", className="card-title")),
                    ],
                    style=card_style,
                ),
                md=3,
            ),
        ],
        className="gy-3",
    )


def _feature_controls() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Candidate Profile", className="card-title"),
                html.Div(
                    [
                        dbc.Label("Years of Experience"),
                        dcc.Slider(
                            id="experience-years",
                            min=0,
                            max=30,
                            step=0.5,
                            value=float(DEFAULTS["experience_years"]),
                            marks={0: "0", 10: "10", 20: "20", 30: "30"},
                        ),
                        dbc.Label("Years with Current Company", className="mt-3"),
                        dcc.Slider(
                            id="tenure-years",
                            min=0,
                            max=15,
                            step=0.25,
                            value=float(DEFAULTS["years_with_company"]),
                            marks={0: "0", 5: "5", 10: "10", 15: "15"},
                        ),
                        dbc.Label("Age", className="mt-3"),
                        dcc.Slider(
                            id="age",
                            min=20,
                            max=65,
                            step=1,
                            value=float(DEFAULTS["age"]),
                            marks={20: "20", 35: "35", 50: "50", 65: "65"},
                        ),
                        dbc.Label("Company Size (employees)", className="mt-3"),
                        dcc.Dropdown(
                            id="company-size",
                            options=[
                                {"label": "50", "value": 50},
                                {"label": "200", "value": 200},
                                {"label": "1,000", "value": 1_000},
                                {"label": "5,000", "value": 5_000},
                                {"label": "10,000", "value": 10_000},
                            ],
                            value=int(DEFAULTS["company_size"]),
                            clearable=False,
                        ),
                        dbc.Label("Education Level", className="mt-3"),
                        dcc.Dropdown(
                            id="education-level",
                            options=[{"label": label, "value": label} for label in data_utils.EDUCATION_LEVELS],
                            value=str(DEFAULTS["education_level"]),
                            clearable=False,
                        ),
                        dbc.Label("Job Title", className="mt-3"),
                        dcc.Dropdown(
                            id="job-title",
                            options=[{"label": title, "value": title} for title in data_utils.JOB_TITLES],
                            value=str(DEFAULTS["job_title"]),
                            clearable=False,
                        ),
                        dbc.Label("Country", className="mt-3"),
                        dcc.Dropdown(
                            id="country",
                            options=[{"label": country, "value": country} for country in data_utils.COUNTRIES],
                            value=str(DEFAULTS["country"]),
                            clearable=False,
                        ),
                        dbc.Label("Employment Type", className="mt-3"),
                        dcc.Dropdown(
                            id="employment-type",
                            options=[{"label": label, "value": label} for label in data_utils.EMPLOYMENT_TYPES],
                            value=str(DEFAULTS["employment_type"]),
                            clearable=False,
                        ),
                        dbc.Label("Remote Ratio", className="mt-3"),
                        dcc.Dropdown(
                            id="remote-ratio",
                            options=[{"label": label, "value": label} for label in data_utils.REMOTE_RATIO_VALUES],
                            value=str(DEFAULTS["remote_ratio"]),
                            clearable=False,
                        ),
                    ]
                ),
                dbc.Button("Predict Salary", id="predict-button", color="primary", className="mt-4 w-100"),
            ]
        ),
        className="shadow-sm",
    )


def _insight_tabs() -> dbc.Tabs:
    return dbc.Tabs(
        [
            dbc.Tab(label="Experience vs Salary", tab_id="experience"),
            dbc.Tab(label="Job Title Distribution", tab_id="distribution"),
            dbc.Tab(label="Feature Importances", tab_id="importances"),
        ],
        id="insight-tabs",
        active_tab="experience",
    )


def _insights_card() -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(_insight_tabs()),
                html.Div(
                    [
                        dbc.Label("Filter by Job Title", className="mt-3"),
                        dcc.Dropdown(
                            id="job-filter",
                            options=[{"label": title, "value": title} for title in ["All"] + data_utils.JOB_TITLES],
                            value="All",
                            clearable=False,
                        ),
                        dcc.Graph(id="insight-graph", className="mt-3"),
                    ]
                ),
            ]
        ),
        className="shadow-sm",
    )


def create_app() -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )
    app.title = "Salary Prediction Studio"

    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H1("Salary Prediction Studio", className="display-5"),
                            html.P(
                                "Experiment with role, experience, and location to forecast compensation",
                                className="lead",
                            ),
                        ]
                    ),
                    width=12,
                ),
                className="py-4",
            ),
            _metrics_cards(),
            dbc.Row(
                [
                    dbc.Col(_feature_controls(), md=4),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Predicted Salary", className="card-title"),
                                    html.Div(
                                        id="prediction-output",
                                        className="display-4 fw-bold text-primary",
                                        children=_format_currency(DEFAULT_PREDICTION),
                                    ),
                                    html.Hr(),
                                    html.Small("Estimated range (± MAE):", className="text-muted"),
                                    html.Div(
                                        id="prediction-interval",
                                        className="fs-5",
                                        children=_confidence_range(DEFAULT_PREDICTION, METRICS.get("mae", 8_000)),
                                    ),
                                    html.Div(
                                        html.Small(
                                            "* Model metrics come from the held-out test set (80/20 split)",
                                            className="text-muted",
                                        ),
                                        className="mt-3",
                                    ),
                                ]
                            ),
                            className="shadow-sm h-100",
                        ),
                        md=8,
                    ),
                ],
                className="gy-4 mt-1",
            ),
            dbc.Row(
                dbc.Col(_insights_card(), width=12),
                className="py-4",
            ),
        ],
    )

    register_callbacks(app)

    return app


def register_callbacks(app: Dash) -> None:
    mae = METRICS.get("mae", 8_000)

    @app.callback(
        Output("prediction-output", "children"),
        Output("prediction-interval", "children"),
        Input("predict-button", "n_clicks"),
        State("experience-years", "value"),
        State("tenure-years", "value"),
        State("age", "value"),
        State("company-size", "value"),
        State("education-level", "value"),
        State("job-title", "value"),
        State("country", "value"),
        State("employment-type", "value"),
        State("remote-ratio", "value"),
        prevent_initial_call=True,
    )
    def _update_prediction(
        _n_clicks,
        experience_years,
        tenure_years,
        age,
        company_size,
        education_level,
        job_title,
        country,
        employment_type,
        remote_ratio,
    ):
        features = {
            "experience_years": experience_years,
            "years_with_company": tenure_years,
            "age": age,
            "company_size": company_size,
            "education_level": education_level,
            "job_title": job_title,
            "country": country,
            "employment_type": employment_type,
            "remote_ratio": remote_ratio,
        }

        prediction = _predict(features)
        return _format_currency(prediction), _confidence_range(prediction, mae)

    @app.callback(
        Output("insight-graph", "figure"),
        Input("insight-tabs", "active_tab"),
        Input("job-filter", "value"),
    )
    def _update_graph(active_tab: str, job_filter: str):
        df = DATASET.features.copy()
        df[config.TARGET_COLUMN] = DATASET.target

        if job_filter != "All":
            df = df[df["job_title"] == job_filter]

        if df.empty:
            return px.scatter(title="No data for selected filter")

        if active_tab == "experience":
            fig = px.scatter(
                df,
                x="experience_years",
                y=config.TARGET_COLUMN,
                color="job_title",
                trendline="ols",
                labels={"experience_years": "Years of Experience", config.TARGET_COLUMN: "Salary"},
                hover_data=["education_level", "country"],
            )
            fig.update_layout(legend=dict(orientation="h"))
            return fig

        if active_tab == "distribution":
            fig = px.box(
                df,
                x="job_title",
                y=config.TARGET_COLUMN,
                color="job_title",
                labels={config.TARGET_COLUMN: "Salary", "job_title": "Job Title"},
            )
            fig.update_layout(showlegend=False)
            return fig

        # Feature importances tab
        if not FEATURE_IMPORTANCES.empty:
            fig = px.bar(
                FEATURE_IMPORTANCES,
                x="group",
                y="importance_percent",
                labels={"group": "Feature", "importance_percent": "Importance (%)"},
                text=FEATURE_IMPORTANCES["importance_percent"].map(lambda x: f"{x:.1f}%"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_range=[0, FEATURE_IMPORTANCES["importance_percent"].max() * 1.1])
            return fig

        return px.bar(title="Feature importance unavailable")


app = create_app()
server = app.server


def main(debug: bool = True, host: str = "0.0.0.0", port: int = 8050) -> None:
    app.run_server(debug=debug, host=host, port=port)


if __name__ == "__main__":
    main()

