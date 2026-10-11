import pandas as pd
import plotly.express as px
import streamlit as st

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def _find_importance_file():

    candidates = []

    for file in ARTIFACT_DIR.rglob("*.csv"):

        name = file.name.lower()

        if (
            "importance" in name
            or "permutation" in name
        ):
            candidates.append(file)

    if not candidates:
        return None

    return candidates[0]


def _find_column(df, candidates):

    normalized = {
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_"): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def render_insights():

    # --------------------------------------------------
    # Heading
    # --------------------------------------------------

    st.title("Feature Insights")

    st.caption(
        "Feature importance from the official Random Forest "
        "evaluation, together with the reasoning behind the "
        "feature engineering choices."
    )

    # --------------------------------------------------
    # Load permutation importance
    # --------------------------------------------------

    importance_path = _find_importance_file()

    if importance_path is None:

        st.warning(
            "Random Forest permutation importance data "
            "was not found in the artifacts directory."
        )

        st.info(
            "Expected a saved CSV containing permutation "
            "importance results."
        )

        return

    importance_df = pd.read_csv(importance_path)

    # --------------------------------------------------
    # Identify columns
    # --------------------------------------------------

    feature_column = _find_column(
        importance_df,
        [
            "feature",
            "feature_name",
            "features"
        ]
    )

    importance_column = _find_column(
        importance_df,
        [
            "importance",
            "importance_mean",
            "mean_importance",
            "permutation_importance"
        ]
    )

    if feature_column is None or importance_column is None:

        st.error(
            "The saved permutation-importance file does not "
            "contain recognizable feature and importance columns."
        )

        st.write(
            "Columns found:",
            list(importance_df.columns)
        )

        return

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    importance_df = importance_df[
        [feature_column, importance_column]
    ].copy()

    importance_df.columns = [
        "Feature",
        "Importance"
    ]

    importance_df["Importance"] = pd.to_numeric(
        importance_df["Importance"],
        errors="coerce"
    )

    importance_df = importance_df.dropna(
        subset=["Importance"]
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    # --------------------------------------------------
    # Top feature
    # --------------------------------------------------

    top_feature = importance_df.iloc[0]

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Most Important Feature",
            str(top_feature["Feature"])
        )

    with c2:

        st.metric(
            "Permutation Importance",
            f"{top_feature['Importance']:.4f}"
        )

    # --------------------------------------------------
    # Importance chart
    # --------------------------------------------------

    st.subheader(
        "Random Forest Feature Importance"
    )

    chart_data = importance_df.copy()

    fig = px.bar(
        chart_data.sort_values(
            "Importance",
            ascending=True
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Permutation Importance"
    )

    fig.update_layout(
        height=max(
            450,
            len(chart_data) * 32
        ),
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

    st.caption(
        "Higher positive importance means that randomly "
        "permuting the feature tended to reduce model "
        "performance more strongly."
    )

    # --------------------------------------------------
    # Feature interpretation
    # --------------------------------------------------

    st.subheader(
        "What the Important Features Represent"
    )

    explanations = {
        "city_development_index":
            "Represents the development level associated "
            "with the candidate's city. It was the strongest "
            "feature in the saved Random Forest permutation "
            "importance analysis.",

        "company_size":
            "Represents the size category of the candidate's "
            "current or previous company.",

        "city":
            "Represents the candidate's city. The modeling "
            "pipeline handles city as a categorical feature "
            "with rare categories grouped during cleaning.",

        "company_type":
            "Represents the type of company where the "
            "candidate currently or previously worked.",

        "education_level":
            "Represents the candidate's highest recorded "
            "education level.",

        "relevent_experience":
            "Indicates whether the candidate has relevant "
            "experience for the role.",

        "last_new_job_numeric":
            "Represents the recency of the candidate's "
            "previous job change after ordinal conversion.",

        "major_discipline":
            "Represents the candidate's major or academic "
            "discipline.",

        "training_hours":
            "Represents the amount of training completed "
            "by the candidate."
    }

    top_features = importance_df.head(8)

    for _, row in top_features.iterrows():

        feature = str(row["Feature"])

        explanation = explanations.get(
            feature,
            "This feature is included in the trained "
            "model and contributes according to its "
            "measured permutation importance."
        )

        with st.container(border=True):

            st.markdown(
                f"### {feature}"
            )

            st.write(
                explanation
            )

            st.caption(
                f"Permutation importance: "
                f"{row['Importance']:.4f}"
            )

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------

    st.subheader(
        "Feature Engineering Used in the Pipeline"
    )

    engineering_data = pd.DataFrame({
        "Original Feature": [
            "experience",
            "last_new_job",
            "city",
            "Categorical missing values"
        ],
        "Transformation": [
            "<1, 1–20 and >20 converted to ordered numeric values; missingness retained separately.",
            "never, 1–4 and >4 converted to ordered numeric values; missingness retained separately.",
            "Rare cities grouped into Other before one-hot encoding.",
            "Missing categorical values replaced with Unknown."
        ]
    })

    st.dataframe(
        engineering_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The trained pipelines perform numeric imputation, "
        "scaling and categorical one-hot encoding during "
        "prediction. These transformations are kept inside "
        "the saved models so the frontend uses the same "
        "preprocessing as training."
    )

    # --------------------------------------------------
    # Important interpretation note
    # --------------------------------------------------

    st.subheader(
        "How to Interpret Feature Importance"
    )

    st.write(
        "Permutation importance measures how much model "
        "performance changes when a feature's values are "
        "randomly shuffled. A larger positive value indicates "
        "that the model relied more heavily on that feature."
    )

    st.warning(
        "Feature importance does not prove that a feature "
        "causes job-change behavior. It describes the feature's "
        "importance to the trained model."
    )