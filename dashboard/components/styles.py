"""Custom CSS styles for the Streamlit dashboard."""

CUSTOM_CSS = """
<style>
    /* Main theme */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Headers */
    h1 {
        color: #1a1a2e;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }

    h2, h3 {
        color: #16213e;
    }

    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
    }

    /* Tables */
    .dataframe {
        font-size: 0.85rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
"""


def apply_styles():
    """Apply custom CSS to the Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
