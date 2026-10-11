import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from utils.model_loader import (
    load_test_metrics,
    load_selected_thresholds,
    load_ensemble_bundle,
)


def _find_column(df, possible_names):
    """
    Find a column using normalized names.
    """

    normalized = {
        str(column).strip().lower().replace("_", " ").replace("-", " "): column
        for column in df.columns
    }

    for name in possible_names:

        key = (
            name.strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        if key in normalized:
            return normalized[key]

    return None


def render_performance():

    # --------------------------------------------------
    # Page heading
    # --------------------------------------------------

    st.title("Model Performance")

    st.caption(
        "Comparison of four individual models and two ensemble "
        "models on the held-out final test set."
    )

    # --------------------------------------------------
    # Load saved results
    # --------------------------------------------------

    
    metrics = load_test_metrics().copy()
    thresholds = load_selected_thresholds().copy()

    # Add Soft Voting and Stacking metrics.
    ensemble_metrics_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "ensemble"
        / "ensemble_metrics.csv"
    )

    if ensemble_metrics_path.exists():
        ensemble_metrics = pd.read_csv(ensemble_metrics_path)

        # Standardize column names before combining the files.
        def standardize_columns(df):
            df = df.copy()
            df.columns = [
                str(column).strip().lower().replace(" ", "_").replace("-", "_")
                for column in df.columns
            ]

            # Keep equivalent metric names consistent.
            aliases = {
                "f1_score": "f1",
                "log_loss": "logloss",
                "brier_score": "brier",
            }
            return df.rename(columns=aliases)

        metrics = standardize_columns(metrics)
        ensemble_metrics = standardize_columns(ensemble_metrics)

        # Ensure both files have compatible metric columns.
        metrics = pd.concat(
            [metrics, ensemble_metrics],
        ignore_index=True,
        sort=False,
        )

        # Include ensemble thresholds in the displayed threshold table.
        ensemble_bundle = load_ensemble_bundle()
        thresholds.update(ensemble_bundle["thresholds"])


    if metrics.empty:
        st.error(
            "Final test metrics could not be loaded from "
            "artifacts/metrics/final_test_metrics.csv."
        )
        return

    # --------------------------------------------------
    # Identify important columns
    # --------------------------------------------------

    model_col = _find_column(
        metrics,
        ["model"]
    )

    accuracy_col = _find_column(
        metrics,
        ["accuracy"]
    )

    precision_col = _find_column(
        metrics,
        ["precision"]
    )

    recall_col = _find_column(
        metrics,
        ["recall"]
    )

    f1_col = _find_column(
        metrics,
        ["f1", "f1 score"]
    )

    balanced_col = _find_column(
        metrics,
        ["balanced accuracy", "balanced_accuracy"]
    )

    roc_auc_col = _find_column(
        metrics,
        ["roc auc", "roc_auc", "roc-auc"]
    )

    pr_auc_col = _find_column(
        metrics,
        ["pr auc", "pr_auc", "pr-auc"]
    )

    logloss_col = _find_column(
        metrics,
        ["logloss", "log loss"]
    )

    brier_col = _find_column(
        metrics,
        ["brier", "brier score"]
    )

    if model_col is None:
        st.error("Model column was not found in the metrics file.")
        return

    # --------------------------------------------------
    # Overview
    # --------------------------------------------------

    st.subheader("Test Set Overview")

    overview_columns = st.columns(4)

    overview_metrics = [
        ("Accuracy", accuracy_col),
        ("F1 Score", f1_col),
        ("ROC-AUC", roc_auc_col),
        ("PR-AUC", pr_auc_col),
    ]

    for column, (label, metric_column) in zip(
        overview_columns,
        overview_metrics
    ):

        with column:

            if metric_column is not None:

                best_value = metrics[metric_column].max()

                st.metric(
                    label=f"Best {label}",
                    value=f"{best_value:.3f}"
                )

            else:

                st.metric(
                    label=label,
                    value="N/A"
                )

    st.write("")

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    st.subheader("Model Ranking")

    if pr_auc_col is not None:

        ranking = metrics.sort_values(
            pr_auc_col,
            ascending=False
        ).reset_index(drop=True)

        st.caption(
            "Default ranking is based on PR-AUC because the "
            "target classes are imbalanced."
        )

        ranking_data = pd.DataFrame({
            "Rank": range(1, len(ranking) + 1),
            "Model": ranking[model_col].values,
            "PR-AUC": ranking[pr_auc_col].values,
        })

        if f1_col is not None:
            ranking_data["F1"] = ranking[f1_col].values

        if roc_auc_col is not None:
            ranking_data["ROC-AUC"] = ranking[roc_auc_col].values

        st.dataframe(
            ranking_data,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------
    # Interactive metric comparison
    # --------------------------------------------------

    st.subheader("Metric Comparison")

    available_metrics = []

    metric_map = {
        "Accuracy": accuracy_col,
        "Precision": precision_col,
        "Recall": recall_col,
        "F1 Score": f1_col,
        "Balanced Accuracy": balanced_col,
        "ROC-AUC": roc_auc_col,
        "PR-AUC": pr_auc_col,
    }

    for label, column in metric_map.items():

        if column is not None:
            available_metrics.append(label)

    if available_metrics:

        selected_metric = st.selectbox(
            "Select metric",
            available_metrics
        )

        selected_column = metric_map[selected_metric]

        chart_data = metrics[
            [model_col, selected_column]
        ].copy()

        chart_data.columns = [
            "Model",
            "Score"
        ]

        chart_data = chart_data.sort_values(
            "Score",
            ascending=False
        )

        fig = px.bar(
            chart_data,
            x="Model",
            y="Score",
            text="Score",
            title=f"{selected_metric} by Model"
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[
                max(0, chart_data["Score"].min() - 0.05),
                min(1.0, chart_data["Score"].max() + 0.08)
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
    # Detailed metrics
    # --------------------------------------------------

    st.subheader("Detailed Test Metrics")

    display_columns = []

    column_pairs = [
        ("Model", model_col),
        ("Accuracy", accuracy_col),
        ("Precision", precision_col),
        ("Recall", recall_col),
        ("F1 Score", f1_col),
        ("Balanced Accuracy", balanced_col),
        ("ROC-AUC", roc_auc_col),
        ("PR-AUC", pr_auc_col),
        ("Log Loss", logloss_col),
        ("Brier Score", brier_col),
    ]

    for display_name, actual_column in column_pairs:

        if actual_column is not None:
            display_columns.append(
                (display_name, actual_column)
            )

    detailed = metrics[
        [actual for _, actual in display_columns]
    ].copy()

    detailed.columns = [
        display for display, _ in display_columns
    ]

    st.dataframe(
        detailed.style.format({
            column: "{:.4f}"
            for column in detailed.columns
            if column != "Model"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Thresholds
    # --------------------------------------------------

    st.subheader("Decision Thresholds")

    threshold_rows = []

    for model_name, threshold in thresholds.items():

        threshold_rows.append({
            "Model": model_name,
            "Selected Threshold": threshold
        })

    if threshold_rows:

        threshold_df = pd.DataFrame(threshold_rows)

        st.dataframe(
            threshold_df.style.format({
                "Selected Threshold": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "A prediction is classified as positive when the "
        "model-estimated probability reaches or exceeds its "
        "selected threshold."
    )

    # --------------------------------------------------
    # Saved evaluation plots
    # --------------------------------------------------

    st.subheader("Evaluation Visualizations")

    project_root = Path(__file__).resolve().parents[2]
    plot_dir = project_root / "artifacts" / "plots"

    if not plot_dir.exists():

        st.info(
            "Saved evaluation plots were not found."
        )
        return

    plot_files = [
        file for file in plot_dir.iterdir()
        if file.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ]

    if not plot_files:

        st.info(
            "No saved evaluation images were found."
        )
        return

    # Display groups
    plot_groups = [
        ("ROC Curves", ["roc"]),
        ("Precision-Recall Curves", ["pr", "precision_recall"]),
        ("Confusion Matrices", ["confusion"]),
        ("Calibration Curves", ["calibration"]),
    ]

    displayed_any = False

    for group_title, keywords in plot_groups:

        matching_files = [
            file for file in plot_files
            if any(
                keyword in file.name.lower()
                for keyword in keywords
            )
        ]

        if not matching_files:
            continue

        displayed_any = True

        st.markdown(f"#### {group_title}")

        columns = st.columns(
            min(2, len(matching_files))
        )

        for index, image_file in enumerate(
            matching_files
        ):

            with columns[index % len(columns)]:

                st.image(
                    str(image_file),
                    caption=image_file.stem,
                    use_container_width=True
                )

    if not displayed_any:

        st.info(
            "No recognized evaluation plot groups were found."
        )