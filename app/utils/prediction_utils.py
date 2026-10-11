import numpy as np
import pandas as pd


MODEL_NUMERIC_FEATURES = [
    "city_development_index",
    "training_hours",
    "experience_missing",
    "experience_numeric",
    "last_new_job_missing",
    "last_new_job_numeric",
]


MODEL_CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "relevent_experience",
    "enrolled_university",
    "education_level",
    "major_discipline",
    "company_size",
    "company_type",
]


def convert_experience(value):

    if value in (None, "", "Unknown"):
        return np.nan

    if value == "<1":
        return 0.0

    if value == ">20":
        return 21.0

    return float(value)


def convert_last_new_job(value):

    if value in (None, "", "Unknown"):
        return np.nan

    if value == "never":
        return 0.0

    if value == ">4":
        return 5.0

    return float(value)


def build_model_input(candidate):

    experience_numeric = convert_experience(
        candidate["experience"]
    )

    last_new_job_numeric = convert_last_new_job(
        candidate["last_new_job"]
    )

    experience_missing = int(
        pd.isna(experience_numeric)
    )

    last_new_job_missing = int(
        pd.isna(last_new_job_numeric)
    )

    row = {
        "city": candidate["city"],
        "city_development_index": candidate[
            "city_development_index"
        ],
        "gender": candidate["gender"],
        "relevent_experience": candidate[
            "relevent_experience"
        ],
        "enrolled_university": candidate[
            "enrolled_university"
        ],
        "education_level": candidate[
            "education_level"
        ],
        "major_discipline": candidate[
            "major_discipline"
        ],
        "company_size": candidate[
            "company_size"
        ],
        "company_type": candidate[
            "company_type"
        ],
        "training_hours": candidate[
            "training_hours"
        ],
        "experience_missing": experience_missing,
        "experience_numeric": experience_numeric,
        "last_new_job_missing": last_new_job_missing,
        "last_new_job_numeric": last_new_job_numeric,
    }

    return pd.DataFrame([row])



def predict_all_models(
    models,
    model_input,
    thresholds,
    ensemble_bundle=None,
):
    """Generate predictions from individual and ensemble models."""

    results = []

    # Original four models: preserve the existing prediction logic.
    for model_name, model in models.items():
        probability = float(
            model.predict_proba(model_input)[0, 1]
        )

        threshold = float(thresholds[model_name])
        prediction = int(probability >= threshold)

        results.append({
            "model": model_name,
            "prediction": prediction,
            "probability": probability,
            "threshold": threshold,
            "label": (
                "Likely to seek job change"
                if prediction == 1
                else "Likely to stay"
            ),
        })

    # Add Soft Voting and Stacking when the bundle is available.
    if ensemble_bundle is not None:
        base_models = ensemble_bundle["base_models"]
        meta_model = ensemble_bundle["stacking_meta_model"]
        probability_columns = ensemble_bundle["probability_columns"]
        base_column_names = ensemble_bundle["base_model_column_names"]
        ensemble_thresholds = ensemble_bundle["thresholds"]

        base_probabilities = {}

        for model_name, model in base_models.items():
            column = base_column_names[model_name]
            base_probabilities[column] = float(
                model.predict_proba(model_input)[0, 1]
            )

        # Soft Voting: average the four base-model probabilities.
        soft_probability = float(
            sum(
                base_probabilities[column]
                for column in probability_columns
            ) / len(probability_columns)
        )

        # Stacking: use the base-model probabilities as meta-features.
        meta_input = pd.DataFrame(
            [[base_probabilities[column]
              for column in probability_columns]],
            columns=probability_columns,
        )

        stacking_probability = float(
            meta_model.predict_proba(meta_input)[0, 1]
        )

        ensemble_probabilities = {
            "Soft Voting": soft_probability,
            "Stacking": stacking_probability,
        }

        for model_name, probability in ensemble_probabilities.items():
            threshold = float(ensemble_thresholds[model_name])
            prediction = int(probability >= threshold)

            results.append({
                "model": model_name,
                "prediction": prediction,
                "probability": probability,
                "threshold": threshold,
                "label": (
                    "Likely to seek job change"
                    if prediction == 1
                    else "Likely to stay"
                ),
            })

    return results