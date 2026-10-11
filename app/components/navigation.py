import streamlit as st


NAVIGATION_ITEMS = [
    ("Prediction", "◎"),
    ("Model Performance", "◈"),
    ("Hyperparameter Tuning", "⚙"),
    ("Feature Insights", "◇"),
]


def render_navigation():

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Prediction"

    st.markdown(
        '<div class="navigation-container">',
        unsafe_allow_html=True,
    )

    columns = st.columns(4)

    for column, (page_name, icon) in zip(
        columns,
        NAVIGATION_ITEMS,
    ):

        with column:

            is_active = (
                st.session_state["active_page"] == page_name
            )

            button_label = f"{icon}  {page_name}"

            if st.button(
                button_label,
                key=f"nav_{page_name}",
                use_container_width=True,
            ):

                if st.session_state["active_page"] != page_name:

                    st.session_state["active_page"] = page_name
                    st.rerun()

    st.markdown(
        """
        <div class="navigation-hint">
            Select a section to explore the ML workspace
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    return st.session_state["active_page"]