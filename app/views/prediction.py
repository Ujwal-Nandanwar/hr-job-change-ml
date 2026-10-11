import streamlit as st

from utils.model_loader import (
    load_feature_categories,
    load_models,
    load_selected_thresholds,
    load_test_metrics,
    load_ensemble_bundle,
)

from utils.prediction_utils import (
    build_model_input,
    predict_all_models,
)


def render_prediction():

    # --------------------------------------------------
    # Page heading
    # --------------------------------------------------

    st.html("""
<div style="
    margin-top: 10px;
    margin-bottom: 25px;
">

    <div style="
        font-size: 32px;
        font-weight: 700;
    ">
        Candidate Prediction
    </div>

    <div style="
        font-size: 15px;
        margin-top: 6px;
        opacity: 0.7;
    ">
        Enter candidate information to estimate their likelihood
        of seeking a new job.
    </div>

</div>
""")

    # --------------------------------------------------
    # Load model information
    # --------------------------------------------------

    models = load_models()
    categories = load_feature_categories()
    thresholds = load_selected_thresholds()
    metrics = load_test_metrics()
    ensemble_bundle = load_ensemble_bundle()

    st.subheader("Candidate Information")

    st.caption(
        "Labels show the original dataset feature names used "
        "by the trained models. Experience and Last New Job "
        "are transformed internally before prediction."
    )

    # --------------------------------------------------
    # Candidate form
    # --------------------------------------------------

    with st.form("candidate_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            city = st.selectbox(
                "City (`city`)",
                categories["city"],
                help="Dataset feature: city"
            )

            city_development_index = st.number_input(
                "City Development Index (`city_development_index`)",
                min_value=0.448,
                max_value=0.949,
                value=0.903,
                step=0.001,
                format="%.3f",
                help="Dataset feature: city_development_index"
            )

            gender = st.selectbox(
                "Gender (`gender`)",
                categories["gender"],
                help="Dataset feature: gender"
            )

            relevent_experience = st.selectbox(
                "Relevant Experience (`relevent_experience`)",
                categories["relevent_experience"],
                help="Dataset feature: relevent_experience"
            )

            enrolled_university = st.selectbox(
                "Enrolled University (`enrolled_university`)",
                categories["enrolled_university"],
                help="Dataset feature: enrolled_university"
            )

            education_level = st.selectbox(
                "Education Level (`education_level`)",
                categories["education_level"],
                help="Dataset feature: education_level"
            )

        with col2:

            major_discipline = st.selectbox(
                "Major Discipline (`major_discipline`)",
                categories["major_discipline"],
                help="Dataset feature: major_discipline"
            )

            company_size = st.selectbox(
                "Company Size (`company_size`)",
                categories["company_size"],
                help="Dataset feature: company_size"
            )

            company_type = st.selectbox(
                "Company Type (`company_type`)",
                categories["company_type"],
                help="Dataset feature: company_type"
            )

            experience = st.selectbox(
                "Experience (`experience`)",
                [
                    "<1",
                    *[str(i) for i in range(1, 21)],
                    ">20",
                    "Unknown",
                ],
                help=(
                    "Original dataset feature: experience. "
                    "Converted to experience_numeric internally."
                )
            )

            last_new_job = st.selectbox(
                "Last New Job (`last_new_job`)",
                [
                    "never",
                    "1",
                    "2",
                    "3",
                    "4",
                    ">4",
                    "Unknown",
                ],
                help=(
                    "Original dataset feature: last_new_job. "
                    "Converted to last_new_job_numeric internally."
                )
            )

            training_hours = st.number_input(
                "Training Hours (`training_hours`)",
                min_value=1,
                max_value=336,
                value=47,
                step=1,
                help="Dataset feature: training_hours"
            )

        st.write("")

        submitted = st.form_submit_button(
            "Predict Job Change",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------
    # Run predictions
    # --------------------------------------------------

    if submitted:

        candidate = {
            "city": city,
            "city_development_index": city_development_index,
            "gender": gender,
            "relevent_experience": relevent_experience,
            "enrolled_university": enrolled_university,
            "education_level": education_level,
            "major_discipline": major_discipline,
            "company_size": company_size,
            "company_type": company_type,
            "experience": experience,
            "last_new_job": last_new_job,
            "training_hours": training_hours,
        }

        try:

            model_input = build_model_input(candidate)

            results = predict_all_models(
                models,
                model_input,
                thresholds,
                ensemble_bundle,
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )
            st.stop()

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.divider()

        st.subheader("Model Predictions")

        result_columns = st.columns(2)

        for index, result in enumerate(results):

            with result_columns[index % 2]:

                with st.container(border=True):

                    st.markdown(
                        f"### {result['model']}"
                    )

                    if result["prediction"] == 1:
                        st.warning(
                            "Likely to seek job change"
                        )
                    else:
                        st.success(
                            "Likely to stay"
                        )

                    st.metric(
                        "Model-estimated probability",
                        f"{result['probability']:.1%}"
                    )

                    st.progress(
                        min(
                            max(
                                result["probability"],
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.caption(
                        f"Decision threshold: "
                        f"{result['threshold']:.2f}"
                    )

                    # Show one important test-set metric
                    if not metrics.empty:

                        # Find the model-name column
                        model_column = None

                        for candidate in ["model", "model_name", "modelname"]:
                            if candidate in metrics.columns:
                                model_column = candidate
                                break

                        if model_column is not None:

                            metric_rows = metrics[
                                metrics[model_column].astype(str).str.strip()
                                == result["model"]
                            ]

                            if not metric_rows.empty:

                                row = metric_rows.iloc[0]

                                if "f1" in metrics.columns:
                                    st.caption(
                                        f"Test F1: {row['f1']:.3f}"
                                    )

                                if "roc_auc" in metrics.columns:
                                    st.caption(
                                        f"Test ROC-AUC: {row['roc_auc']:.3f}"
                                    )
        # --------------------------------------------------
        # Agreement summary
        # --------------------------------------------------

        positive_models = sum(
            result["prediction"] == 1
            for result in results
        )

        st.divider()

        if positive_models == len(results):
            st.success(f"All {len(results)} models predict that the candidate is likely to seek a job change.")
        elif positive_models == 0:
            st.info(f"All {len(results)} models predict that the candidate is likely to stay.")
        else:
            st.warning(
                f"{positive_models} out of {len(results)} models predict that "
                "the candidate is likely to seek a job change."
            )

        # --------------------------------------------------
        # Internal input preview
        # --------------------------------------------------

        with st.expander(
            "View model input after preprocessing"
        ):
            st.dataframe(
                model_input,
                use_container_width=True,
                hide_index=True
            )