import streamlit as st


def render_footer():

    st.markdown(
        """
        <div class="footer-divider"></div>

        <div class="app-footer">

            <div class="footer-left">
                <strong>HR Analytics</strong>
                <span> • Job Change Prediction Studio</span>
            </div>

            <div class="footer-right">
                4 ML Models&nbsp;&nbsp;•&nbsp;&nbsp;Interactive Analytics
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )