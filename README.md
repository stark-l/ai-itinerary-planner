 ✈️ AI Travel Planner

This is a Streamlit web application designed to help users plan travel itineraries using the power of Google's Gemini AI models. It offers two modes: a detailed step-by-step planner and a quick, automated planner.

## ✨ Features

**Common:**
*   Powered by Google Gemini (specifically `gemini-1.5-flash-latest`).
*   Uses Mapbox GL JS for interactive map visualizations (requires a Mapbox Access Token for full functionality).
*   Handles API keys securely via a `.env` file.

**1. 🔍 Detailed Planner:**
*   **Trip Definition:** Define destination, duration, activity preferences, and budget style.
*   **AI Brainstorming:** Chat with Gemini to get activity and place suggestions based on your trip criteria.
*   **Activity Curation:** Select activities from AI suggestions to build a personalized list.
*   **Geocoding:** Automatically finds coordinates for curated activities using Nominatim (via `tools.py` and `geopy`).
*   **Location Overview Map:** View your curated, geocoded activities on a 2D Mapbox map.
*   **Detailed Itinerary Generation:** Let Gemini create a timed, day-by-day itinerary using your selected activities, including suggested timings, activity types, descriptions, and map view parameters.
*   **Interactive Itinerary Map:** Explore the generated plan on an interactive 3D Mapbox map with a synchronized sidebar displaying the daily schedule. Click on stops to fly to their location.
*   **Itinerary Editing:** Modify the generated plan by adding available activities from your pool, removing scheduled stops, or reordering stops within a day.

**2. ⚡ Quick Mode Planner:**
*   **Simplified Input:** Provide destination, duration, and a free-text description of your interests/vibe.
*   **One-Click Generation:** The AI performs the following steps automatically:
    *   Brainstorms relevant places based on your input.
    *   Geocodes the suggested places.
    *   Generates a complete, detailed, day-by-day itinerary using the geocoded places.
*   **Interactive Itinerary Map Display:** View the instantly generated itinerary on the same interactive 3D Mapbox map component used in the Detailed Planner.
*   **Chat-based Modification:** Use a chat interface to request changes to the generated itinerary (e.g., "Swap Day 1 and Day 2", "Add a coffee break", "Remove the park visit"). The AI will attempt to update the plan and refresh the map.

## 📁 Project Structure.
├── .env # Stores API keys (!! IMPORTANT: Add to .gitignore !!)
├── .gitignore # Specifies intentionally untracked files that Git should ignore
├── requirements.txt # Python dependencies
└── src
├── pages # Contains individual Streamlit pages (multi-page app)
│ ├── 1_Detailed_Planner.py
│ ├── 2_Quick_Mode_Planner.py
│ └── init.py
├── init.py
├── Main_page.py # The main entry point / landing page for Streamlit
├── itinerary_agent.py # Contains functions calling Gemini for brainstorming/planning
└── tools.py # Utility functions (geocoding, routing - future, POI - future)

## 🚀 Setup and Installation

1.  **Prerequisites:**
    *   Python (version 3.9 or higher recommended)
    *   Git (for cloning the repository)

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```

3.  **Create a Virtual Environment (Recommended):**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Keys:**
    *   Create a file named `.env` in the **root** directory of the project (the same level as `src` and `requirements.txt`).
    *   Add your API keys to the `.env` file like this:
        ```dotenv
        GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
        MAPBOX_ACCESS_TOKEN="YOUR_MAPBOX_ACCESS_TOKEN"
        ```
    *   Replace `"YOUR_GOOGLE_GEMINI_API_KEY"` with your actual key obtained from [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console.
    *   Replace `"YOUR_MAPBOX_ACCESS_TOKEN"` with your actual token obtained from [Mapbox](https://www.mapbox.com/). The Mapbox token is required for the interactive maps; the app will show warnings but may partially function without it.
    *   **Important:** Ensure the `.env` file is listed in your `.gitignore` file to prevent accidentally committing your secret keys.

## ▶️ Running the Application

1.  Make sure your virtual environment is activated.
2.  Navigate to the **root** directory of the project in your terminal (the directory containing the `src` folder).
3.  Run the Streamlit app using the main page script:
    ```bash
    streamlit run src/Main_page.py
    ```
4.  Streamlit will start a local web server and should automatically open the application in your default web browser.

## 📝 Usage

1.  **Select Mode:** Use the sidebar navigation to choose between the "Detailed Planner" and "Quick Mode Planner".
2.  **Detailed Planner Workflow:**
    *   Fill in your trip details in Section 1.
    *   Use the chat interface in Section 2 to ask the AI for activity suggestions.
    *   Select desired activities from the suggestions in Section 3.
    *   Review your curated list in Section 4. Click "Geocode Curated Activities" to find their locations.
    *   View the geocoded locations on the 2D overview map in Section 5.
    *   In Section 6, specify the number of days and click "Generate Detailed Plan".
    *   Explore the generated plan on the interactive 3D map and sidebar.
    *   Optionally, use the "Edit Itinerary Plan" expander below the map to make manual adjustments.
3.  **Quick Mode Planner Workflow:**
    *   Enter your destination, duration, and interests/vibe.
    *   Click "Generate Quick Plan".
    *   Wait for the AI to brainstorm, geocode, and generate the itinerary (progress will be shown).
    *   Explore the generated plan on the interactive 3D map and sidebar.
    *   Use the chat input below the map to ask the AI for modifications to the plan. The map and sidebar will update if the modification is successful.

## ⚙️ Dependencies

All required Python packages are listed in `requirements.txt`. Key libraries include:

*   `streamlit`: The web application framework.
*   `google-generativeai`: For interacting with the Gemini API.
*   `geopy`: For geocoding place names via Nominatim.
*   `pandas`: Used for data handling, particularly for map data.
*   `numpy`: Numerical processing, often a dependency.
*   `scikit-learn`: Used in the (currently unused by Gemini agents) `create_basic_itinerary` function for K-Means clustering.
*   `python-dotenv`: For loading environment variables from `.env`.
*   `requests`: For making HTTP requests (used in `tools.py` for potential future routing/POI features).

---

*Disclaimer: This application uses AI models which may sometimes produce inaccurate or unexpected results. Always double-check critical information like addresses, opening times, and travel details.*