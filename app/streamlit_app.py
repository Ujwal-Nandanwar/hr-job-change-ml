import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="HR Analytics",
    page_icon="📊",
    layout="wide"
)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Prediction"

APP_DIR = Path(__file__).resolve().parent
STYLE_PATH = APP_DIR / "styles" / "theme.css"

if STYLE_PATH.exists():
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")

if st.session_state["dark_mode"]:
    st.html("""
    <style>
        .stApp, .main, .block-container {
            background-color: #0f172a !important;
            color: #f8fafc !important;
        }

        .stage-card,
        .preview-card,
        .stat-card,
        .model-card,
        .metric-card,
        .insight-card,
        .tuning-card {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }

        .stage-card *,
        .preview-card *,
        .stat-card *,
        .model-card *,
        .metric-card *,
        .insight-card *,
        .tuning-card * {
            color: #f8fafc !important;
        }
    </style>
    """)
else:
    st.html("""
    <style>
        .stApp, .main, .block-container {
            background-color: #f6f8fc !important;
            color: #111827 !important;
        }

        .stage-card,
        .preview-card,
        .stat-card,
        .model-card,
        .metric-card,
        .insight-card,
        .tuning-card {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #e5e7eb !important;
        }

        .stage-card *,
        .preview-card *,
        .stat-card *,
        .model-card *,
        .metric-card *,
        .insight-card *,
        .tuning-card * {
            color: #111827 !important;
        }
    </style>
    """)

from components.header import render_header
from components.navigation import render_navigation

render_header()
render_navigation()

if st.session_state["active_page"] == "Prediction":

    from views.prediction import render_prediction
    render_prediction()

elif st.session_state["active_page"] == "Model Performance":

    from views.performance import render_performance
    render_performance()

elif st.session_state["active_page"] == "Hyperparameter Tuning":

    from views.tuning import render_tuning
    render_tuning()

elif st.session_state["active_page"] == "Feature Insights":

    from views.insights import render_insights
    render_insights()