import streamlit as st


def render_header():

    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False

    header_html = """
    <div class="app-header"
         style="
            background: var(--header-bg);
            color: var(--header-text);
            border: 1px solid var(--header-border);
            border-radius: 18px;
            padding: 22px 26px;
            margin-bottom: 18px;
         ">

        <div style="
            display: flex;
            align-items: center;
            gap: 16px;
        ">

            <div style="
                width: 56px;
                height: 56px;
                border-radius: 15px;
                background: var(--brand-bg);
                color: var(--brand-text);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                font-weight: 700;
                flex-shrink: 0;
            ">
                HR
            </div>

            <div>

                <div style="
                    font-size: 28px;
                    font-weight: 700;
                    line-height: 1.2;
                    color: var(--header-text);
                ">
                    HR Analytics
                </div>

                <div style="
                    font-size: 14px;
                    margin-top: 4px;
                    color: var(--header-muted);
                ">
                    Job Change Prediction Studio
                </div>

            </div>

        </div>

    </div>
    """
    
    # Theme variables for the entire application
    if st.session_state["dark_mode"]:
        st.html("""
        <style>
            :root {
                --bg-primary: #0b1220;
                --bg-secondary: #111827;
                --bg-tertiary: #1f2937;

                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-muted: #94a3b8;

                --border: #334155;

                --accent: #a5b4fc;
                --accent-light: #25264a;

                --success: #34d399;
                --danger: #fb7185;

                --shadow-small: 0 2px 8px rgba(0, 0, 0, 0.20);
                --shadow-medium: 0 10px 30px rgba(0, 0, 0, 0.30);

                /* Streamlit text and background aliases */
                --text-color: var(--text-primary);
                --background-color: var(--bg-primary);
                --secondary-background-color: var(--bg-secondary);
                --primary-color: var(--accent);
                --border-color: var(--border);

                /* Header */
                --header-bg: #111827;
                --header-text: #f8fafc;
                --header-muted: #cbd5e1;
                --header-border: #334155;
                --brand-bg: #f8fafc;
                --brand-text: #111827;
            }
        </style>
        """)
    else:
        st.html("""
        <style>
            :root {
                --bg-primary: #f6f8fc;
                --bg-secondary: #ffffff;
                --bg-tertiary: #f0f2f7;

                --text-primary: #151821;
                --text-secondary: #667085;
                --text-muted: #667085;

                --border: #e4e7ec;

                --accent: #5b5bd6;
                --accent-light: #eeeeff;

                --success: #159570;
                --danger: #d6455d;

                --shadow-small: 0 2px 8px rgba(16, 24, 40, 0.06);
                --shadow-medium: 0 10px 30px rgba(16, 24, 40, 0.08);

                /* Streamlit text and background aliases */
                --text-color: var(--text-primary);
                --background-color: var(--bg-primary);
                --secondary-background-color: var(--bg-secondary);
                --primary-color: var(--accent);
                --border-color: var(--border);

                /* Header */
                --header-bg: #ffffff;
                --header-text: #111827;
                --header-muted: #6b7280;
                --header-border: #e5e7eb;
                --brand-bg: #111827;
                --brand-text: #ffffff;
            }
        </style>
        """)


    st.html(header_html)

    # Dark/light mode button
    if st.session_state["dark_mode"]:
        button_label = "☀️ Light Mode"
    else:
        button_label = "🌙 Dark Mode"

    if st.button(button_label, key="dark_mode_button"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()