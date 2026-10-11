import streamlit as st


def render_stat_card(label, value, description=""):

    st.markdown(
        f"""
        <div class="stat-card">

            <div class="stat-label">
                {label}
            </div>

            <div class="stat-value">
                {value}
            </div>

            <div class="stat-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(
    model_name,
    prediction,
    probability,
    threshold,
    metric_name=None,
    metric_value=None,
):

    prediction_text = (
        "Likely to change jobs"
        if prediction == 1
        else "Likely to stay"
    )

    metric_html = ""

    if metric_name is not None and metric_value is not None:

        metric_html = f"""
        <div class="model-metric">
            <span>{metric_name}</span>
            <strong>{metric_value:.3f}</strong>
        </div>
        """

    st.markdown(
        f"""
        <div class="model-card">

            <div class="model-card-header">

                <div class="model-name">
                    {model_name}
                </div>

                <div class="model-status">
                    ML MODEL
                </div>

            </div>

            <div class="model-prediction">
                {prediction_text}
            </div>

            <div class="model-probability">

                <span>
                    Model-estimated probability
                </span>

                <strong>
                    {probability:.1%}
                </strong>

            </div>

            <div class="model-threshold">
                Decision threshold: {threshold:.2f}
            </div>

            {metric_html}

        </div>
        """,
        unsafe_allow_html=True,
    )