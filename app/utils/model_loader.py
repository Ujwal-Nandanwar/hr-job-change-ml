from pathlib import Path
import json
import pickle

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
METADATA_DIR = PROJECT_ROOT / "artifacts" / "metadata"
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
TUNING_DIR = PROJECT_ROOT / "artifacts" / "tuning"
ENSEMBLE_DIR = PROJECT_ROOT / "artifacts" / "ensemble"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "KNN": "knn.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
}


CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "company_size",
    "company_type",
]

def _find_artifact(filename):
    """
    Search the entire artifacts directory for a specific artifact.
    """
    matches = list(
        (PROJECT_ROOT / "artifacts").rglob(filename)
    )

    if not matches:
        return None

    return matches[0]


@st.cache_resource
def load_models():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        with open(model_path, "rb") as file:
            models[model_name] = pickle.load(file)

    return models

@st.cache_resource
def load_ensemble_bundle():
    """Load the saved Soft Voting and Stacking components."""
    bundle_path = ENSEMBLE_DIR / "ensemble_bundle.pkl"

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Ensemble bundle not found: {bundle_path}"
        )

    with open(bundle_path, "rb") as file:
        return pickle.load(file)

def _find_encoder(estimator):

    if isinstance(estimator, OneHotEncoder):
        return estimator

    if isinstance(estimator, Pipeline):

        for _, step in estimator.steps:
            encoder = _find_encoder(step)

            if encoder is not None:
                return encoder

    if isinstance(estimator, ColumnTransformer):

        for _, transformer, _ in estimator.transformers_:

            if transformer in ("drop", "passthrough"):
                continue

            encoder = _find_encoder(transformer)

            if encoder is not None:
                return encoder

    return None


@st.cache_resource
def load_feature_categories():

    models = load_models()

    reference_model = models["Logistic Regression"]

    encoder = _find_encoder(reference_model)

    if encoder is None:
        raise RuntimeError(
            "Could not find the fitted OneHotEncoder "
            "inside the saved model."
        )

    if len(encoder.categories_) != len(CATEGORICAL_FEATURES):
        raise RuntimeError(
            "Encoder category count does not match "
            "the expected categorical features."
        )

    categories = {}

    for feature_name, feature_values in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_
    ):
        categories[feature_name] = [
            str(value) for value in feature_values
        ]

    return categories


@st.cache_data
def load_selected_thresholds():

    path = _find_artifact("selected_thresholds.json")

    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    thresholds = {}

    # Handle:
    # {"Logistic Regression": 0.30, ...}
    # and:
    # {"thresholds": {"Logistic Regression": 0.30, ...}}
    raw_data = data.get("thresholds", data)

    for model_name, value in raw_data.items():

        # Direct numeric value
        if isinstance(value, (int, float)):
            thresholds[model_name] = float(value)
            continue

        # Dictionary value
        if isinstance(value, dict):

            for key in [
                "threshold",
                "selected_threshold",
                "best_threshold",
                "value"
            ]:

                if key in value and value[key] is not None:
                    thresholds[model_name] = float(value[key])
                    break

    return thresholds

@st.cache_data
def load_test_metrics():

    path = _find_artifact("final_test_metrics.csv")

    if path is None:
        return pd.DataFrame()

    return _normalize_metric_columns(
        pd.read_csv(path)
    )

@st.cache_data
def load_tuning_summary():

    path = _find_artifact("tuning_summary.csv")

    if path is None:
        return pd.DataFrame()

    return _normalize_metric_columns(
        pd.read_csv(path)
    )


@st.cache_data
def load_best_parameters():

    path = _find_artifact("best_parameters.json")

    if path is None:
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_baseline_metrics():

    path = _find_artifact("baseline_cv_metrics.csv")

    if path is None:
        return pd.DataFrame()

    return _normalize_metric_columns(
        pd.read_csv(path))


def _normalize_metric_columns(df):

    if df.empty:
        return df

    df = df.copy()

    df.columns = [
        str(column).strip().lower().replace("-", "_").replace(" ", "_")
        for column in df.columns
    ]

    return df