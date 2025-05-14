# src/main_app.py

import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="AI Travel Planner",
    page_icon="✈️"
)

st.title("✈️ Welcome to WayLi! - The future of travel planning")

st.markdown(
    """
    Select a planning mode from the sidebar:

    - **Detailed Planner:** Brainstorm, curate activities, and generate a customizable itinerary.
    - **Quick Mode:** Provide basic details and let the AI generate a full itinerary suggestion instantly.

    """
)
