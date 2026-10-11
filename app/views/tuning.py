import json

import pandas as pd
import plotly.express as px
import streamlit as st

from pathlib import Path

from utils.model_loader import (
    load_tuning_summary,
    load_best_parameters,
    load_baseline_metrics,
    load_selected_thresholds,
    load_test_metrics,
    load_ensemble_bundle,
)


def _find_column(df, names):

    normalized = {
        str(column).strip().lower().replace("_", " ").replace("-", " "): column
        for column in df.columns
    }

    for name in names:

        key = (
            name.strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if key in normalized:
            return normalized[key]

    return None


def _format_parameter_value(value):

    if isinstance(value, float):
        return f"{value:.6g}"

    if isinstance(value, dict):
        return json.dumps(value)

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    return str(value)


def render_tuning():

    # --------------------------------------------------
    # Heading
    # --------------------------------------------------

    st.title("Hyperparameter Tuning")

    st.caption(
        "Tuning results from the official model-selection process. "
        "The frontend does not retrain any model."
    )

    tuning_summary = load_tuning_summary()
    best_parameters = load_best_parameters()
    baseline_metrics = load_baseline_metrics()
    thresholds = load_selected_thresholds()
    
    # Include the selected thresholds for Soft Voting and Stacking.
    ensemble_bundle = load_ensemble_bundle()

    if isinstance(ensemble_bundle, dict):
        ensemble_thresholds = ensemble_bundle.get("thresholds", {})

        if isinstance(ensemble_thresholds, dict):
            thresholds.update(ensemble_thresholds)

    # --------------------------------------------------
    # Check data
    # --------------------------------------------------

    if tuning_summary.empty:

        st.warning(
            "tuning_summary.csv was not found or contains no data."
        )

    # --------------------------------------------------
    # Best tuned PR-AUC overview
    # --------------------------------------------------

    st.subheader("Best Cross-Validation Results")

    if not tuning_summary.empty:

        model_col = _find_column(
            tuning_summary,
            ["model"]
        )

        score_col = _find_column(
            tuning_summary,
            [
                "best pr auc",
                "best_pr_auc",
                "pr auc",
                "pr_auc",
            ]
        )

        if model_col is not None and score_col is not None:

            best_row = tuning_summary.loc[
                tuning_summary[score_col].idxmax()
            ]

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Best Model",
                    str(best_row[model_col])
                )

            with c2:
                st.metric(
                    "Best CV PR-AUC",
                    f"{best_row[score_col]:.4f}"
                )

    # --------------------------------------------------
    # Tuning comparison
    # --------------------------------------------------

    if not tuning_summary.empty:

        st.subheader("Tuned Model Comparison")

        model_col = _find_column(
            tuning_summary,
            ["model"]
        )

        score_col = _find_column(
            tuning_summary,
            [
                "best pr auc",
                "best_pr_auc",
                "pr auc",
                "pr_auc",
            ]
        )

        if model_col is not None and score_col is not None:

            chart_data = tuning_summary[
                [model_col, score_col]
            ].copy()

            chart_data.columns = [
                "Model",
                "CV PR-AUC"
            ]

            chart_data = chart_data.sort_values(
                "CV PR-AUC",
                ascending=False
            )

            fig = px.bar(
                chart_data,
                x="Model",
                y="CV PR-AUC",
                text="CV PR-AUC",
                title="Best Cross-Validated PR-AUC"
            )

            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_range=[
                    0,
                    min(
                        1.0,
                        chart_data["CV PR-AUC"].max() + 0.08
                    )
                ],
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------
    # Best parameters
    # --------------------------------------------------

    st.subheader("Best Hyperparameters")

    if best_parameters:

        # Handle common formats:
        # {"Logistic Regression": {...}, ...}
        # or {"models": {"Logistic Regression": {...}, ...}}

        parameter_data = best_parameters

        if isinstance(parameter_data, dict):
            parameter_data = parameter_data.get(
                "models",
                parameter_data
            )

        for model_name, parameters in parameter_data.items():

            if not isinstance(parameters, dict):
                continue

            with st.expander(
                str(model_name),
                expanded=True
            ):

                parameter_rows = []

                for parameter, value in parameters.items():

                    parameter_rows.append({
                        "Parameter": parameter,
                        "Best Value": _format_parameter_value(value)
                    })

                parameter_df = pd.DataFrame(
                    parameter_rows
                )

                st.dataframe(
                    parameter_df,
                    use_container_width=True,
                    hide_index=True
                )

    else:

        st.info(
            "best_parameters.json was not found."
        )

    # --------------------------------------------------
    # Baseline vs tuned
    # --------------------------------------------------

    st.subheader("Baseline vs Tuned Performance")

    if not tuning_summary.empty and not baseline_metrics.empty:

        tuned_model_col = _find_column(
            tuning_summary,
            ["model"]
        )

        tuned_score_col = _find_column(
            tuning_summary,
            [
                "best pr auc",
                "best_pr_auc",
                "pr auc",
                "pr_auc"
            ]
        )

        baseline_model_col = _find_column(
            baseline_metrics,
            ["model"]
        )

        baseline_score_col = _find_column(
            baseline_metrics,
            [
                "pr auc",
                "pr_auc",
                "pr-auc"
            ]
        )

        if (
            tuned_model_col is not None
            and tuned_score_col is not None
            and baseline_model_col is not None
            and baseline_score_col is not None
        ):

            tuned_data = tuning_summary[
                [tuned_model_col, tuned_score_col]
            ].copy()

            tuned_data.columns = [
                "Model",
                "Tuned PR-AUC"
            ]

            baseline_data = baseline_metrics[
                [baseline_model_col, baseline_score_col]
            ].copy()

            baseline_data.columns = [
                "Model",
                "Baseline PR-AUC"
            ]

            comparison = pd.merge(
                baseline_data,
                tuned_data,
                on="Model",
                how="inner"
            )

            comparison_long = comparison.melt(
                id_vars="Model",
                var_name="Stage",
                value_name="PR-AUC"
            )

            fig = px.bar(
                comparison_long,
                x="Model",
                y="PR-AUC",
                color="Stage",
                barmode="group",
                text="PR-AUC",
                title="Baseline vs Tuned PR-AUC"
            )

            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_range=[
                    0,
                    min(
                        1.0,
                        comparison_long["PR-AUC"].max() + 0.08
                    )
                ],
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Show numeric improvement
            comparison["Improvement"] = (
                comparison["Tuned PR-AUC"]
                - comparison["Baseline PR-AUC"]
            )

            st.dataframe(
                comparison.style.format({
                    "Baseline PR-AUC": "{:.4f}",
                    "Tuned PR-AUC": "{:.4f}",
                    "Improvement": "{:+.4f}",
                }),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "The expected model/PR-AUC columns could not be "
                "identified in the saved metric files."
            )

    else:

        st.warning(
            "Baseline or tuning results could not be loaded."
        )

    
    # --------------------------------------------------
    # Ensemble configuration
    # --------------------------------------------------

    st.divider()
    st.subheader("Ensemble Model Configuration")

    st.caption(
        "The ensemble models combine predictions from the existing "
        "base models. This page displays their saved configuration "
        "and results; it does not retrain the models."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Soft Voting**")
        st.write(
            "Combines the predicted probabilities from the four "
            "individual models by averaging them."
        )

    with col2:
        st.markdown("**Stacking**")
        st.write(
            "Uses out-of-fold probability predictions from the "
            "base models as inputs to a Logistic Regression "
            "meta-model."
        )

    ensemble_dir = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "ensemble"
    )

    ensemble_parameters_path = (
        ensemble_dir / "ensemble_parameters.json"
    )

    ensemble_metrics_path = (
        ensemble_dir / "ensemble_metrics.csv"
    )

    if ensemble_parameters_path.exists():
        try:
            with open(
                ensemble_parameters_path,
                "r",
                encoding="utf-8",
            ) as file:
                ensemble_parameters = json.load(file)

            with st.expander(
                "View saved ensemble parameters",
                expanded=False,
            ):
                st.json(ensemble_parameters)

        except (OSError, json.JSONDecodeError) as error:
            st.warning(
                f"Could not read ensemble parameters: {error}"
            )
    else:
        st.info(
            "The saved ensemble_parameters.json file was not found."
        )

    # --------------------------------------------------
    # Ensemble vs individual model results
    # --------------------------------------------------

    st.subheader("Ensemble vs Individual Models")

    st.caption(
        "The following comparison uses held-out final test results "
        "for all six models. These are evaluation results, not "
        "cross-validation tuning scores."
    )

    if ensemble_metrics_path.exists():
        try:
            individual_metrics = load_test_metrics().copy()
            ensemble_metrics = pd.read_csv(ensemble_metrics_path)

            def standardize_metric_columns(df):
                df = df.copy()

                df.columns = [
                    str(column)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                    .replace(".", "")
                    for column in df.columns
                ]

                aliases = {
                    "model_name": "model",
                    "f1_score": "f1",
                    "balanced_acc": "balanced_accuracy",
                    "roc_auc_score": "roc_auc",
                    "pr_auc_score": "pr_auc",
                    "log_loss": "logloss",
                    "brier_score": "brier",
                }

                return df.rename(
                    columns={
                        old: new
                        for old, new in aliases.items()
                        if old in df.columns
                    }
                )

            individual_metrics = standardize_metric_columns(
                individual_metrics
            )
            ensemble_metrics = standardize_metric_columns(
                ensemble_metrics
            )

            if (
                "model" in individual_metrics.columns
                and "model" in ensemble_metrics.columns
            ):
                comparison = pd.concat(
                    [individual_metrics, ensemble_metrics],
                    ignore_index=True,
                    sort=False,
                )

                metric_labels = {
                    "Accuracy": "accuracy",
                    "Precision": "precision",
                    "Recall": "recall",
                    "F1-score": "f1",
                    "Balanced Accuracy": "balanced_accuracy",
                    "ROC-AUC": "roc_auc",
                    "PR-AUC": "pr_auc",
                }

                display_labels = {
                    "model": "Model",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1-score",
                    "balanced_accuracy": "Balanced Accuracy",
                    "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC",
                }

                display_columns = [
                    column
                    for column in display_labels
                    if column in comparison.columns
                ]

                comparison_display = comparison[
                    display_columns
                ].rename(columns=display_labels)

                if "F1-score" in comparison_display.columns:
                    comparison_display = comparison_display.sort_values(
                        "F1-score",
                        ascending=False,
                    )

                numeric_columns = [
                    column
                    for column in comparison_display.columns
                    if column != "Model"
                ]

                st.dataframe(
                    comparison_display.style.format(
                        {
                            column: "{:.4f}"
                            for column in numeric_columns
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                available_metrics = {
                    label: column
                    for label, column in metric_labels.items()
                    if (
                        column in comparison.columns
                        and comparison[column].notna().any()
                    )
                }

                if available_metrics:
                    selected_metric = st.selectbox(
                        "Compare models using",
                        list(available_metrics.keys()),
                        index=(
                            list(available_metrics.keys()).index(
                                "F1-score"
                            )
                            if "F1-score" in available_metrics
                            else 0
                        ),
                        key="ensemble_comparison_metric",
                    )

                    metric_column = available_metrics[selected_metric]

                    chart_data = comparison[
                        ["model", metric_column]
                    ].copy()

                    chart_data[metric_column] = pd.to_numeric(
                        chart_data[metric_column],
                        errors="coerce",
                    )

                    chart_data = chart_data.dropna(
                        subset=[metric_column]
                    ).sort_values(
                        metric_column,
                        ascending=False,
                    )

                    if not chart_data.empty:
                        fig = px.bar(
                            chart_data,
                            x="model",
                            y=metric_column,
                            text=metric_column,
                            title=(
                                f"Final Test {selected_metric} "
                                "Across All Models"
                            ),
                        )

                        fig.update_traces(
                            texttemplate="%{text:.4f}",
                            textposition="outside",
                        )

                        fig.update_layout(
                            yaxis_range=[
                                0,
                                min(
                                    1.0,
                                    float(
                                        chart_data[metric_column].max()
                                    ) + 0.08,
                                ),
                            ],
                            xaxis_title="Model",
                            yaxis_title=selected_metric,
                            margin=dict(
                                l=20,
                                r=20,
                                t=60,
                                b=20,
                            ),
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

            else:
                st.warning(
                    "A model-name column was not found in both "
                    "individual and ensemble metrics."
                )

        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.warning(
                f"Could not load the ensemble comparison: {error}"
            )

    else:
        st.info(
            "The saved ensemble_metrics.csv file was not found."
        )  

    # --------------------------------------------------
    # Selected decision thresholds
    # --------------------------------------------------

    st.subheader("Selected Decision Thresholds")

    threshold_rows = []

    for model_name, threshold in thresholds.items():

        threshold_rows.append({
            "Model": str(model_name),
            "Selected Threshold": f"{float(threshold):.2f}"
        })

    if threshold_rows:

        threshold_df = pd.DataFrame(threshold_rows)

        st.table(threshold_df)

        st.caption(
            "The selected threshold is the probability cutoff used "
            "to convert each model's estimated probability into the "
            "final class prediction."
        )

    else:

        st.warning(
            "No selected decision thresholds were found."
        )