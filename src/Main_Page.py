# src/main_app.py

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="AI Travel Planner",
    page_icon="✈️"
)

st.title("✈️ Welcome to the AI Travel Planner!")

st.markdown(
    """
    Select a planning mode from the sidebar:

    - **🔍 Detailed Planner:** Brainstorm, curate activities, and generate a customizable itinerary.
    - **⚡ Quick Mode:** Provide basic details and let the AI generate a full itinerary suggestion instantly.

    *(Requires Google Gemini API Key and optionally Mapbox Access Token in a `.env` file)*
    """
)

# You could add global setup here if needed, but often it's better
# to do it within each page or dedicated utility modules.