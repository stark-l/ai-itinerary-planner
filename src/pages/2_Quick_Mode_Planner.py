# src/pages/2_Quick_Mode_Planner.py

import streamlit as st
import google.generativeai as genai
import os
import json
import time
import re
import pandas as pd
from dotenv import load_dotenv

# --- 1. Imports & Setup ---
load_dotenv() # Load environment variables (.env file at the project root)

# Import shared tools and agents
# Assumes running with `streamlit run src/Main_page.py` from project root
try:
    from tools import cached_geocode_location # Use the cached version from tools.py
    from itinerary_agent import brainstorm_places_for_quick_mode, generate_detailed_itinerary_gemini, modify_detailed_itinerary_gemini
except ImportError as e:
    st.error(f"Error importing custom modules: {e}. Make sure you are running streamlit from the project root directory and the 'src' folder is correctly structured.")
    st.stop()

import streamlit.components.v1 as components

# --- 2. Initialize Session State for this page ---
# Use unique keys to avoid conflicts with the detailed planner page
if 'quick_mode_location' not in st.session_state: st.session_state.quick_mode_location = ""
if 'quick_mode_duration' not in st.session_state: st.session_state.quick_mode_duration = "3 days"
if 'quick_mode_prefs' not in st.session_state: st.session_state.quick_mode_prefs = "A mix of history, food, and nice views. Not too rushed."
if 'quick_mode_itinerary_data' not in st.session_state: st.session_state.quick_mode_itinerary_data = None
if 'quick_mode_geocoded_places' not in st.session_state: st.session_state.quick_mode_geocoded_places = []
if 'quick_mode_error' not in st.session_state: st.session_state.quick_mode_error = None
if 'quick_mode_generating' not in st.session_state: st.session_state.quick_mode_generating = False
if 'quick_mode_status_msgs' not in st.session_state: st.session_state.quick_mode_status_msgs = [] # Store status messages
if 'quick_mode_chat_messages' not in st.session_state: st.session_state.quick_mode_chat_messages = [] # Store chat messages

# --- Configuration ---
GEMINI_MODEL_ITINERARY = 'gemini-1.5-flash-latest' # Model for final itinerary generation

# --- Helper Function ---
def parse_duration_days(duration_str: str) -> int:
    """Extracts the number of days from a string."""
    match = re.search(r'\d+', duration_str)
    if match:
        return max(1, int(match.group()))
    return 3 # Default if parsing fails

# --- 3. API Configuration ---
# Moved this section up to ensure config happens before potential use
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("🔴 Error: GOOGLE_API_KEY environment variable not found.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"🔴 Error configuring Google AI SDK: {e}")
    st.stop()

MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_ACCESS_TOKEN:
    st.warning("⚠️ Mapbox Access Token missing. Interactive map requires MAPBOX_ACCESS_TOKEN in .env.")
    # Don't stop, but the map component will fail later if token is truly needed by JS

# --- 4. UI Layout ---
st.set_page_config(layout="wide", page_title="Quick Planner") # Page specific config
st.title("AI Assistant: Instant Itinerary")
st.markdown("Provide your destination, duration, and a brief description of your interests. The AI will suggest locations and generate a full itinerary plan.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    st.text_input("Destination:", placeholder="e.g., Rome, Italy", key='quick_mode_location')
    st.text_input("Trip Duration:", placeholder="e.g., 5 days", key='quick_mode_duration')
with col2:
    st.text_area("Your Interests / Trip Vibe:", height=120, placeholder="e.g., Interested in ancient history, great pasta, maybe some art. Like walking around, but not too hectic.", key='quick_mode_prefs')

# Generate Button (placed after inputs)
generate_button = st.button("🚀 Generate Quick Plan", key="quick_generate_button", type="primary", disabled=st.session_state.quick_mode_generating)

# Progress bar placeholder (will be filled during generation)
progress_bar_placeholder = st.empty()
status_text_placeholder = st.empty()

# --- 5. Button Logic (Generation Process Trigger) ---
if generate_button and not st.session_state.quick_mode_generating:
    # Reset previous results/errors/chat
    st.session_state.quick_mode_itinerary_data = None
    st.session_state.quick_mode_geocoded_places = []
    st.session_state.quick_mode_error = None
    st.session_state.quick_mode_status_msgs = []
    st.session_state.quick_mode_chat_messages = [] # Clear chat history on new generation
    st.session_state.quick_mode_generating = True
    st.rerun() # Rerun immediately to show the "Generating..." state and disable button

# --- 6. Generation Process Execution ---
# This section runs *during* the generation process initiated by the rerun above
if st.session_state.quick_mode_generating:
    location = st.session_state.quick_mode_location
    duration = st.session_state.quick_mode_duration
    prefs = st.session_state.quick_mode_prefs
    num_days = parse_duration_days(duration)

    if not location:
        st.error("Please enter a destination.")
        st.session_state.quick_mode_generating = False # Reset state
        st.stop() # Stop this run

    # Use the placeholders defined earlier
    progress_bar = progress_bar_placeholder.progress(0)
    status_text_placeholder.info("Initializing...")

    try:
        # --------- Step 1: Brainstorm Places ----------
        step_text = "🧠 Brainstorming places with Gemini..."
        st.session_state.quick_mode_status_msgs.append(step_text)
        status_text_placeholder.info(step_text)
        progress_bar.progress(10, text=step_text)

        suggested_places = brainstorm_places_for_quick_mode(location, duration, prefs)
        if not suggested_places:
            st.session_state.quick_mode_error = "Failed to brainstorm places. The AI might not have suggestions, or an error occurred. Try adjusting your preferences."
            raise Exception(st.session_state.quick_mode_error)

        # --------- Step 2: Geocode Places ----------
        step_text = f"🗺️ Geocoding {len(suggested_places)} potential places..."
        st.session_state.quick_mode_status_msgs.append(step_text)
        status_text_placeholder.info(step_text)
        progress_bar.progress(30, text=step_text)

        geocoded_places_list = []
        failed_geocode = []
        total_places = len(suggested_places)
        for i, place_name in enumerate(suggested_places):
            # Update progress specifically for geocoding step
            geocode_progress = 30 + int(60 * (i + 1) / total_places)
            geocode_progress_text = f"Geocoding: {place_name[:30]}... ({i+1}/{total_places})"
            progress_bar.progress(geocode_progress, text=geocode_progress_text)
            status_text_placeholder.info(geocode_progress_text) # Keep user updated

            geo_result = cached_geocode_location(place_name) # Use cached tool from tools.py
            if geo_result:
                geocoded_places_list.append({
                    "place_name": place_name, # Use the name Gemini suggested
                    "latitude": geo_result["latitude"],
                    "longitude": geo_result["longitude"],
                    "address": geo_result["address"]
                })
            else:
                failed_geocode.append(place_name)
            # Optional small sleep if Nominatim rate limiting becomes an issue
            # time.sleep(0.05)

        st.session_state.quick_mode_geocoded_places = geocoded_places_list

        if not geocoded_places_list:
            st.session_state.quick_mode_error = "Could not geocode any suggested places. Please check the destination or try again."
            # Add any failed place names to the error message if they exist
            if failed_geocode:
                st.session_state.quick_mode_error += f" Failed attempts: {', '.join(failed_geocode)}"
            raise Exception(st.session_state.quick_mode_error)

        if failed_geocode:
            # Display warning below the status text temporarily
            st.warning(f"Could not find coordinates for: {', '.join(failed_geocode)}. They won't be included in the final plan.")
            time.sleep(2) # Allow user to see the warning

        # --------- Step 3: Generate Detailed Itinerary ----------
        step_text = f"✍️ Generating {num_days}-day detailed itinerary with Gemini..."
        st.session_state.quick_mode_status_msgs.append(step_text)
        status_text_placeholder.info(step_text)
        progress_bar.progress(90, text=step_text)

        # Prepare activities with coordinates for the itinerary generator
        activities_for_gemini = [
            {"place_name": p["place_name"], "latitude": p["latitude"], "longitude": p["longitude"]}
            for p in geocoded_places_list
        ]

        detailed_plan = generate_detailed_itinerary_gemini(
            activities=activities_for_gemini,
            num_days=num_days,
            destination=location,
            prefs=[prefs] if prefs else [], # Pass prefs as a list
            budget="Any" # Quick mode assumes 'Any' budget for now
        )

        if detailed_plan:
            st.session_state.quick_mode_itinerary_data = detailed_plan
            status_text_placeholder.success("✅ Itinerary Generated!")
            progress_bar.progress(100)
            time.sleep(2) # Let user see success message
        else:
            st.session_state.quick_mode_error = "Failed to generate the detailed itinerary using the suggested places. The AI might have encountered an issue or returned invalid data."
            raise Exception(st.session_state.quick_mode_error)

    except Exception as e:
        st.session_state.quick_mode_error = str(e) # Store error message
        status_text_placeholder.error(f"An error occurred during generation: {e}")
        # Keep progress bar showing the error state
        if progress_bar: progress_bar.progress(100) # Or set to a specific error value like 50


    finally:
        # Clear placeholders only if generation finished successfully
        # On error, keep status showing the error message
        if not st.session_state.quick_mode_error:
             progress_bar_placeholder.empty()
             status_text_placeholder.empty()
        # Reset generating flag regardless of outcome
        st.session_state.quick_mode_generating = False
        # Rerun one last time to update the UI state (remove progress, show results/final error)
        st.rerun()

# --- Error display (after generation attempt finishes) ---
# Placed here so it shows *after* the generation attempt is complete and generating flag is false
elif st.session_state.quick_mode_error and not st.session_state.quick_mode_itinerary_data:
    st.error(f"Failed to generate itinerary: {st.session_state.quick_mode_error}")


# --- 7. Results Display (Map & Chat Interface) ---
# This section now runs only when NOT generating and itinerary data IS present
elif not st.session_state.quick_mode_generating and st.session_state.get('quick_mode_itinerary_data'):
    st.markdown("---")
    st.subheader("🗓️ Generated Itinerary & Map")

    # --- Map Display Section ---
    try:
        itinerary_data = st.session_state.quick_mode_itinerary_data # Already checked it exists

        # Basic validation again just in case
        if not isinstance(itinerary_data, list) or not all(isinstance(day, dict) and 'stops' in day and isinstance(day['stops'], list) for day in itinerary_data):
             st.error("Generated itinerary data has an invalid structure. Cannot display map.")
             st.json(itinerary_data) # Show the invalid data
             # Don't proceed to chat if data is bad
        elif not MAPBOX_ACCESS_TOKEN:
            st.warning("Mapbox token missing. Displaying itinerary data as JSON instead of interactive map.")
            st.json(itinerary_data)
            # Allow chat even without map? Let's proceed.
        else:
            # Convert to JSON string for HTML embedding
            try:
                # Ensure itinerary data is JSON serializable (handles potential complex objects if any slipped through)
                itinerary_json = json.dumps(itinerary_data)
            except TypeError as json_error:
                st.error(f"Error converting itinerary data to JSON for map: {json_error}")
                st.json(itinerary_data) # Show data that caused error
                itinerary_json = None # Prevent map rendering

            if itinerary_json: # Only proceed if JSON conversion worked
                # Determine map center robustly
                map_center_lon, map_center_lat = -9.1393, 38.7223 # Default (Lisbon)
                initial_zoom = 11
                try:
                    first_valid_coord = None
                    for day in itinerary_data:
                        if day.get('stops'):
                            for stop in day['stops']:
                                coords = stop.get('coordinates')
                                if isinstance(coords, list) and len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords):
                                     # Basic range check for longitude/latitude
                                     if -180 <= coords[0] <= 180 and -90 <= coords[1] <= 90:
                                         first_valid_coord = coords
                                         break
                                     else:
                                         st.warning(f"Stop '{stop.get('name', 'Unknown')}' has out-of-range coordinates: {coords}")
                        if first_valid_coord:
                            break

                    if first_valid_coord:
                        map_center_lon, map_center_lat = first_valid_coord[0], first_valid_coord[1]
                        initial_zoom = 12
                    else:
                         st.warning("No valid coordinates found in the first day's stops. Using default map center.")

                except Exception as center_err:
                    st.warning(f"Could not determine map center from itinerary ({center_err}). Using default.")

                map_height_detailed = 750

                # --- HTML Component (Contains CSS and JS for MapLibre GL JS) ---
                # This HTML structure includes the sidebar and map container
                interactive_map_html_with_sidebar = f"""
                <!DOCTYPE html><html lang="en"><head>
                    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Interactive Itinerary Map</title>
                    <link href='https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.css' rel='stylesheet' />
                    <script src='https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.js'></script>
                    <style>
                        /* --- CSS for Sidebar & Map --- */
                        body {{ margin: 0; padding: 0; font-family: 'Arial', sans-serif; overflow: hidden; }}
                        #app-container {{ display: flex; height: {map_height_detailed}px; width: 100%; position: relative; border: 1px solid #ddd; box-sizing: border-box; }}
                        #sidebar {{ width: 340px; background-color: #f8f9fa; padding: 15px; box-shadow: 2px 0 5px rgba(0,0,0,0.1); overflow-y: auto; z-index: 10; border-right: 1px solid #dee2e6; height: 100%; box-sizing: border-box; display: flex; flex-direction: column; }}
                        #sidebar h2 {{ margin-top: 0; color: #343a40; border-bottom: 1px solid #ced4da; padding-bottom: 10px; font-size: 1.2em; flex-shrink: 0; }}
                        #itinerary-content {{ overflow-y: auto; flex-grow: 1; }}
                        .day-header {{ background-color: #007bff; color: white; padding: 8px 12px; margin-top: 15px; margin-bottom: 5px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background-color 0.2s ease; }} .day-header:hover {{ background-color: #0056b3; }} .day-header:first-of-type {{ margin-top: 5px; }}
                        .day-stops {{ list-style: none; padding: 0; margin: 0 0 15px 0; border-left: 3px solid transparent; padding-left: 10px; transition: border-left-color 0.3s ease; }} .day-stops.active {{ border-left-color: #007bff; }}
                        .destination-item {{ padding: 8px 5px; cursor: pointer; border-bottom: 1px solid #e9ecef; transition: background-color 0.2s ease; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; }} .destination-item:hover {{ background-color: #e9ecef; }} .destination-item:last-child {{ border-bottom: none; }}
                        .destination-item span.time {{ font-weight: bold; color: #6c757d; margin-right: 8px; flex-shrink: 0; width: 50px; text-align: right; }} .destination-item span.name {{ flex-grow: 1; text-align: left; margin-left: 5px; }} .destination-item span.type-prefix {{ font-style: italic; color: #555; margin-right: 5px; font-size: 0.85em; background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; border: 1px solid #ddd; white-space: nowrap; }}
                        /* Type Colors (Expanded) */
                        .destination-item--lunch, .destination-item--dinner, .destination-item--food, .destination-item--restaurant {{ background-color: #fff3e0; }} .destination-item--lunch:hover, .destination-item--dinner:hover, .destination-item--food:hover, .destination-item--restaurant:hover {{ background-color: #ffe0b2; }}
                        .destination-item--break, .destination-item--cafe, .destination-item--coffee {{ background-color: #e3f2fd; }} .destination-item--break:hover, .destination-item--cafe:hover, .destination-item--coffee:hover {{ background-color: #bbdefb; }}
                        .destination-item--museum, .destination-item--gallery, .destination-item--culture, .destination-item--history {{ background-color: #ede7f6; }} .destination-item--museum:hover, .destination-item--gallery:hover, .destination-item--culture:hover, .destination-item--history:hover {{ background-color: #d1c4e9; }}
                        .destination-item--park, .destination-item--garden, .destination-item--nature {{ background-color: #e8f5e9; }} .destination-item--park:hover, .destination-item--garden:hover, .destination-item--nature:hover {{ background-color: #c8e6c9; }}
                        .destination-item--viewpoint, .destination-item--landmark, .destination-item--sightseeing {{ background-color: #fce4ec; }} .destination-item--viewpoint:hover, .destination-item--landmark:hover, .destination-item--sightseeing:hover {{ background-color: #f8bbd0; }}
                        .destination-item--shopping, .destination-item--market {{ background-color: #fffde7; }} .destination-item--shopping:hover, .destination-item--market:hover {{ background-color: #fff9c4; }}
                        .destination-item--activity, .destination-item--tour, .destination-item--show {{ background-color: #e0f2f1; }} .destination-item--activity:hover, .destination-item--tour:hover, .destination-item--show:hover {{ background-color: #b2dfdb; }}
                        .destination-item--hotel, .destination-item--accommodation {{ background-color: #eceff1; }} .destination-item--hotel:hover, .destination-item--accommodation:hover {{ background-color: #cfd8dc; }}
                        #map {{ flex-grow: 1; height: 100%; }}
                        #info-panel {{ position: absolute; bottom: 20px; left: 360px; /* Adjust if sidebar width changes */ width: 300px; max-height: 40%; background-color: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 20; display: none; overflow-y: auto; font-size: 0.9em; border: 1px solid #ccc; }} #info-panel h4 {{ margin-top: 0; margin-bottom: 10px; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }} #info-panel p {{ margin-bottom: 5px; line-height: 1.4; color: #555; }} #info-panel-close {{ position: absolute; top: 5px; right: 8px; background: none; border: none; font-size: 1.2em; font-weight: bold; cursor: pointer; color: #888; }} #info-panel-close:hover {{ color: #333; }}
                        .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right {{ z-index: 25; }} .mapboxgl-popup-content {{ font-size: 0.9em; max-width: 200px; padding: 8px 12px; }} .mapboxgl-marker {{ width: 20px; height: 20px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 5px rgba(0,0,0,0.5); cursor: pointer; }}
                    </style>
                </head><body>
                    <div id="app-container">
                        <div id="sidebar">
                            <h2>Daily Plan</h2>
                            <div id="itinerary-content"><p style="text-align: center; color: #777; margin-top: 20px;">Loading itinerary...</p></div>
                        </div>
                        <div id="map"></div>
                        <div id="info-panel"><button id="info-panel-close">×</button><h4 id="info-panel-title"></h4><p id="info-panel-description"></p></div>
                    </div>
                    <script>
                        // --- Javascript for Map Interaction (Use double braces for Mapbox objects/methods) ---
                        mapboxgl.accessToken = '{MAPBOX_ACCESS_TOKEN}';
                        const itineraryData = {itinerary_json}; // Parse the JSON string passed from Python
                        const mapCenter = [{map_center_lon}, {map_center_lat}];
                        const initialZoom = {initial_zoom};
                        const routeLayerId = 'route-line-layer';
                        const routeSourceId = 'route-line-source';
                        const markers = []; // Store markers to potentially clear later if needed

                        // --- Initialize Map ---
                        const map = new mapboxgl.Map({{
                            container: 'map',
                            style: 'mapbox://styles/mapbox/standard', // Use a standard style
                            center: mapCenter,
                            zoom: initialZoom,
                            pitch: 50, // Initial pitch for 3D view
                            bearing: -10, // Slight rotation
                            antialias: true // Improves text rendering
                        }});

                        // --- DOM Elements ---
                        const itineraryContentEl = document.getElementById('itinerary-content');
                        const infoPanelEl = document.getElementById('info-panel');
                        const infoPanelTitleEl = document.getElementById('info-panel-title');
                        const infoPanelDescriptionEl = document.getElementById('info-panel-description');
                        const infoPanelCloseBtn = document.getElementById('info-panel-close');

                        // --- Helper Functions ---
                        function populateSidebar() {{
                            itineraryContentEl.innerHTML = ''; // Clear previous content
                            if (!itineraryData || !Array.isArray(itineraryData) || itineraryData.length === 0) {{
                                itineraryContentEl.innerHTML = '<p style="padding:10px; color:#dc3545;">Error: Invalid or empty itinerary data provided to map.</p>';
                                console.error("Invalid itineraryData passed to JS:", itineraryData);
                                return;
                            }}
                            try {{
                                itineraryData.forEach((dayData, dayIndex) => {{
                                    // Basic validation for day data
                                    if (!dayData || typeof dayData !== 'object') {{
                                        console.warn(`Skipping invalid day data at index ${{dayIndex}}`);
                                        return; // Skip this day if data is bad
                                    }}
                                    const dayNum = dayData.day || (dayIndex + 1); // Fallback day number
                                    const dayTitle = dayData.title || `Day ${{dayNum}}`;

                                    const dayHeader = document.createElement('div');
                                    dayHeader.className = 'day-header';
                                    dayHeader.textContent = dayTitle;
                                    dayHeader.setAttribute('data-day', dayNum);
                                    itineraryContentEl.appendChild(dayHeader);

                                    const stopsList = document.createElement('ul');
                                    stopsList.className = 'day-stops';
                                    stopsList.setAttribute('data-day', dayNum);

                                    if (!dayData.stops || !Array.isArray(dayData.stops) || dayData.stops.length === 0) {{
                                        stopsList.innerHTML = '<li style="padding: 5px 0; color: #6c757d; font-style: italic;">_No activities scheduled._</li>';
                                    }} else {{
                                        dayData.stops.forEach((stop, stopIndex) => {{
                                            // Robust validation for each stop
                                            if (!stop || typeof stop !== 'object' || !stop.coordinates || !Array.isArray(stop.coordinates) || stop.coordinates.length !== 2 || typeof stop.coordinates[0] !== 'number' || typeof stop.coordinates[1] !== 'number') {{
                                                console.warn(`Skipping invalid stop data at Day ${{dayNum}}, index ${{stopIndex}}:`, stop);
                                                const errorItem = document.createElement('li');
                                                errorItem.style.color = 'red';
                                                errorItem.style.fontSize = '0.8em';
                                                errorItem.style.padding = '5px';
                                                errorItem.textContent = `[Invalid Stop Data #${{stopIndex+1}}]`;
                                                stopsList.appendChild(errorItem);
                                                return; // Skip this stop
                                            }}

                                            const listItem = document.createElement('li');
                                            listItem.className = 'destination-item';
                                            const stopType = stop.type ? String(stop.type).toLowerCase().replace(/[^a-z0-9\-]/g, '-') : 'activity'; // Sanitize type for CSS class
                                            listItem.classList.add(`destination-item--${{stopType}}`);

                                            // Store data attributes for interaction
                                            listItem.setAttribute('data-lng', stop.coordinates[0]);
                                            listItem.setAttribute('data-lat', stop.coordinates[1]);
                                            listItem.setAttribute('data-zoom', stop.zoom || 16); // Default zoom
                                            listItem.setAttribute('data-pitch', stop.pitch || 50); // Default pitch
                                            listItem.setAttribute('data-bearing', stop.bearing || 0); // Default bearing
                                            const stopName = stop.name || 'Unnamed Stop';
                                            listItem.setAttribute('data-name', stopName);
                                            listItem.setAttribute('data-description', stop.description || '');
                                            listItem.setAttribute('data-type', stop.type || 'activity'); // Store original type if needed

                                            let typePrefixHTML = '';
                                            const defaultTypes = ['sightseeing', 'activity']; // Types not needing a prefix bubble
                                            if (stop.type && !defaultTypes.includes(stopType)) {{
                                                typePrefixHTML = `<span class="type-prefix">${{stop.type}}</span>`;
                                            }}

                                            listItem.innerHTML = `<span class="time">${{stop.time || ''}}</span>${{typePrefixHTML}}<span class="name">${{stopName}}</span>`;
                                            listItem.addEventListener('click', handleStopClick);
                                            stopsList.appendChild(listItem);
                                            addMarker(stop); // Add marker for valid stops
                                        }});
                                    }}
                                    itineraryContentEl.appendChild(stopsList);

                                    // Add click listener to the day header
                                    dayHeader.addEventListener('click', (e) => {{
                                        const day = parseInt(e.currentTarget.getAttribute('data-day'));
                                        if (!isNaN(day)) {{
                                            highlightDay(day);
                                            drawRouteForDay(day);
                                            flyToDayBounds(day);
                                            hideInfoPanel();
                                        }} else {{
                                            console.error("Invalid day number on header:", e.currentTarget);
                                        }}
                                    }});
                                }});
                            }} catch (error) {{
                                console.error("Error populating sidebar:", error);
                                itineraryContentEl.innerHTML = `<p style="padding:10px; color:#dc3545;">Error rendering itinerary details in sidebar.</p>`;
                            }}
                        }}

                        function handleStopClick(e) {{
                            e.stopPropagation(); // Prevent map click event when clicking item
                            const target = e.currentTarget;
                            try {{
                                const lng = parseFloat(target.getAttribute('data-lng'));
                                const lat = parseFloat(target.getAttribute('data-lat'));
                                const zoom = parseFloat(target.getAttribute('data-zoom'));
                                const pitch = parseFloat(target.getAttribute('data-pitch'));
                                const bearing = parseFloat(target.getAttribute('data-bearing'));
                                const name = target.getAttribute('data-name') || 'Location';
                                const description = target.getAttribute('data-description') || 'No details provided.';

                                // Validate parsed numbers
                                if (isNaN(lng) || isNaN(lat) || isNaN(zoom) || isNaN(pitch) || isNaN(bearing)) {{
                                    console.error("Parsing error in handleStopClick data attributes for:", name);
                                    showInfoPanel(name, `Error: Invalid location data associated with this stop.`);
                                    return;
                                }}

                                // Highlight selected item in sidebar
                                document.querySelectorAll('.destination-item').forEach(item => {{
                                    item.style.fontWeight = 'normal';
                                    // Optionally reset background color if type classes don't cover hover state well
                                    // item.style.backgroundColor = '';
                                }});
                                target.style.fontWeight = 'bold';
                                // target.style.backgroundColor = '#d6eaff'; // Optional direct highlight

                                // Fly map to the location
                                map.flyTo({{
                                    center: [lng, lat],
                                    zoom: zoom,
                                    pitch: pitch,
                                    bearing: bearing,
                                    essential: true, // Ensures animation completes
                                    speed: 1.2,
                                    curve: 1.4
                                }});

                                showInfoPanel(name, description); // Show details in info panel
                            }} catch (parseError) {{
                                console.error("Error processing stop click:", parseError);
                                showInfoPanel("Error", "Could not process stop details due to an unexpected error.");
                            }}
                        }}

                        function addMarker(stop) {{
                            // Validation already done in populateSidebar, but double check coords type
                            if (!stop || !stop.coordinates || !Array.isArray(stop.coordinates) || stop.coordinates.length !== 2 || typeof stop.coordinates[0] !== 'number' || typeof stop.coordinates[1] !== 'number') {{
                                console.warn("addMarker: Skipping marker due to invalid coordinates:", stop);
                                return;
                            }}
                            const el = document.createElement('div');
                            el.className = 'mapboxgl-marker'; // Use the styled div
                            el.style.backgroundColor = getTypeColor(stop.type); // Color based on type

                            const popup = new mapboxgl.Popup({{ offset: 25, closeButton: false }})
                                .setHTML(`<b>${{stop.name || 'Unnamed'}}</b><br>${{stop.time || ''}}`);

                            const marker = new mapboxgl.Marker(el)
                                .setLngLat(stop.coordinates)
                                .setPopup(popup)
                                .addTo(map);

                            // Add hover events to the marker itself
                            el.addEventListener('mouseenter', () => marker.togglePopup());
                            el.addEventListener('mouseleave', () => marker.togglePopup());
                            markers.push(marker); // Keep track of markers
                        }}

                        function getTypeColor(type) {{
                            const typeLower = type ? String(type).toLowerCase().replace(/[^a-z0-9\-]/g, '-') : 'activity';
                             // Match more variations
                            switch (typeLower) {{
                                case 'lunch': case 'dinner': case 'food': case 'restaurant': return '#FFA726'; // Orange
                                case 'break': case 'cafe': case 'coffee': return '#42A5F5'; // Blue
                                case 'museum': case 'gallery': case 'culture': case 'history': case 'art': return '#AB47BC'; // Purple
                                case 'park': case 'garden': case 'nature': case 'walk': return '#66BB6A'; // Green
                                case 'viewpoint': case 'landmark': case 'sightseeing': case 'monument': return '#EC407A'; // Pink
                                case 'shopping': case 'market': case 'shop': return '#FFCA28'; // Yellow/Amber
                                case 'activity': case 'tour': case 'show': case 'event': return '#26A69A'; // Teal
                                case 'hotel': case 'accommodation': case 'stay': return '#78909C'; // Blue Grey
                                default: return '#FF5252'; // Red as default fallback
                            }}
                        }}

                        function highlightDay(dayNum) {{
                            // Highlight sidebar list
                            document.querySelectorAll('.day-stops').forEach(ul => ul.classList.remove('active'));
                            const activeList = document.querySelector(`.day-stops[data-day="${{dayNum}}"]`);
                            if (activeList) {{
                                activeList.classList.add('active');
                            }} else {{
                                console.warn("Could not find stops list for day:", dayNum);
                            }}
                            // Highlight header
                            document.querySelectorAll('.day-header').forEach(hdr => hdr.style.backgroundColor = '#007bff'); // Reset others
                            const activeHdr = document.querySelector(`.day-header[data-day="${{dayNum}}"]`);
                            if (activeHdr) {{
                                activeHdr.style.backgroundColor = '#0056b3'; // Darker blue for active
                            }} else {{
                                console.warn("Could not find header for day:", dayNum);
                            }}
                        }}

                        function drawRouteForDay(dayNum) {{
                            const dayData = itineraryData ? itineraryData.find(d => (d.day || (itineraryData.indexOf(d) + 1)) === dayNum) : null;

                            // Check if day data and stops are valid
                            if (!dayData || !dayData.stops || !Array.isArray(dayData.stops) || dayData.stops.length < 1) {{
                                // Remove existing route if no stops or invalid data
                                if (map.getLayer(routeLayerId)) map.removeLayer(routeLayerId);
                                if (map.getSource(routeSourceId)) map.removeSource(routeSourceId);
                                return;
                            }}

                            // Filter only valid coordinates for the route line
                            const coords = dayData.stops
                                .map(s => s.coordinates)
                                .filter(c => c && Array.isArray(c) && c.length === 2 && typeof c[0] === 'number' && typeof c[1] === 'number');

                            if (coords.length < 1) {{ // Need at least one point to draw anything (though line needs 2)
                                if (map.getLayer(routeLayerId)) map.removeLayer(routeLayerId);
                                if (map.getSource(routeSourceId)) map.removeSource(routeSourceId);
                                return;
                            }}

                            const geojson = {{
                                'type': 'Feature',
                                'properties': {{}},
                                'geometry': {{
                                    'type': 'LineString',
                                    'coordinates': coords
                                }}
                            }};

                            const source = map.getSource(routeSourceId);
                            if (source) {{
                                source.setData(geojson); // Update existing source
                            }} else {{
                                map.addSource(routeSourceId, {{
                                    'type': 'geojson',
                                    'data': geojson
                                }});
                                // Add the route layer below labels for better visibility
                                let firstSymbolId;
                                const layers = map.getStyle().layers;
                                for (let i = 0; i < layers.length; i++) {{
                                    if (layers[i].type === 'symbol') {{
                                        firstSymbolId = layers[i].id;
                                        break;
                                    }}
                                }}
                                map.addLayer({{
                                    'id': routeLayerId,
                                    'type': 'line',
                                    'source': routeSourceId,
                                    'layout': {{
                                        'line-join': 'round',
                                        'line-cap': 'round'
                                    }},
                                    'paint': {{
                                        'line-color': '#ff5722', // Orange route line
                                        'line-width': 4,
                                        'line-opacity': 0.8
                                    }}
                                }}, firstSymbolId); // Add layer before the first symbol layer
                            }}
                        }}

                        function flyToDayBounds(dayNum) {{
                             const dayData = itineraryData ? itineraryData.find(d => (d.day || (itineraryData.indexOf(d) + 1)) === dayNum) : null;
                             if (!dayData || !dayData.stops || !Array.isArray(dayData.stops)) {{
                                 console.warn("Invalid data for flyToDayBounds, day:", dayNum);
                                 return;
                             }}
                             const coords = dayData.stops
                                .map(s => s.coordinates)
                                .filter(c => c && Array.isArray(c) && c.length === 2 && typeof c[0] === 'number' && typeof c[1] === 'number');

                             if (coords.length === 0) {{
                                 console.warn("No valid coordinates to fly to for day:", dayNum);
                                 return; // Cannot fly anywhere
                             }}

                             if (coords.length === 1) {{
                                 // Fly to single point
                                 map.flyTo({{ center: coords[0], zoom: 15, pitch: 50, duration: 1500 }});
                             }} else {{
                                 // Calculate bounds for multiple points
                                 const bounds = new mapboxgl.LngLatBounds();
                                 coords.forEach(coord => bounds.extend(coord));
                                 map.fitBounds(bounds, {{
                                     padding: {{ top: 50, bottom: 50, left: 380, right: 50 }}, // Adjust padding for sidebar
                                     maxZoom: 16,
                                     pitch: 45,
                                     duration: 1500
                                 }});
                             }}
                        }}

                        function showInfoPanel(title, description) {{
                            infoPanelTitleEl.textContent = title || "Details";
                            infoPanelDescriptionEl.textContent = description || "No additional details.";
                            infoPanelEl.style.display = 'block';
                        }}

                        function hideInfoPanel() {{
                            infoPanelEl.style.display = 'none';
                            // De-highlight sidebar item
                            document.querySelectorAll('.destination-item').forEach(item => {{
                                item.style.fontWeight = 'normal';
                                // item.style.backgroundColor = ''; // Optional reset
                            }});
                        }}

                        // --- Event Listeners ---
                        infoPanelCloseBtn.addEventListener('click', hideInfoPanel);
                        map.on('click', hideInfoPanel); // Hide panel if map clicked

                        // --- Map Loaded Event ---
                        map.on('style.load', () => {{
                            console.log("Quick Mode: Map style loaded.");
                            // Add 3D terrain if source doesn't exist
                            if (!map.getSource('mapbox-dem')) {{
                                map.addSource('mapbox-dem', {{
                                    'type': 'raster-dem',
                                    'url': 'mapbox://mapbox.mapbox-terrain-dem-v1',
                                    'tileSize': 512,
                                    'maxzoom': 14
                                }});
                            }}
                            map.setTerrain({{ 'source': 'mapbox-dem', 'exaggeration': 1.5 }});

                            // Add sky layer for atmosphere effect
                            if (!map.getLayer('sky')) {{
                                map.addLayer({{
                                    'id': 'sky',
                                    'type': 'sky',
                                    'paint': {{
                                        'sky-type': 'atmosphere',
                                        'sky-atmosphere-sun': [0.0, 0.0], // Sun position [direction, elevation]
                                        'sky-atmosphere-sun-intensity': 5
                                    }}
                                }});
                            }}

                            // Populate sidebar and draw initial route AFTER style loaded
                            populateSidebar();
                            if (itineraryData && itineraryData.length > 0) {{
                                const firstDayNum = itineraryData[0].day || 1;
                                highlightDay(firstDayNum);
                                drawRouteForDay(firstDayNum);
                                // Optionally fly to bounds on load, might be too much zoom initially
                                // flyToDayBounds(firstDayNum);
                            }} else {{
                                console.log("Quick Mode: No valid initial itinerary data to display on map load.");
                            }}
                            console.log("Quick Mode: Sidebar populated, initial setup complete.");
                        }});

                        // --- Add Map Controls ---
                        map.addControl(new mapboxgl.NavigationControl(), 'top-right');
                        map.addControl(new mapboxgl.FullscreenControl(), 'top-right');
                        map.addControl(new mapboxgl.ScaleControl());

                        // --- Handle Resize ---
                        window.addEventListener('resize', () => {{ map.resize(); }});

                        // --- Error Handling ---
                        map.on('error', (e) => console.error("Mapbox GL Error:", e.error?.message || e));

                        console.log("Quick Mode: Interactive map setup initiated.");
                    </script>
                </body></html>
                """

                # Render the map component
                components.html(interactive_map_html_with_sidebar, height=map_height_detailed + 20, scrolling=False)

    # --- End Map Display Section ---

    # --- Catch errors specifically during map display ---
    except Exception as e:
        st.error(f"Error preparing or displaying interactive map: {e}")
        # Attempt to show the data that caused the error
        st.json(st.session_state.get('quick_mode_itinerary_data', {"error": "Could not retrieve itinerary data for display."}))

    # --- >>> Chat Interface for Modifications (Displayed regardless of map success if itinerary_data exists) <<< ---
# Ensure itinerary_data exists before showing chat
if 'itinerary_data' in locals() and itinerary_data: # Check if variable was assigned and is not None/empty
    st.markdown("---") # Separator
    st.subheader("💬 Modify Your Itinerary")
    st.caption("Chat with the AI to make changes to the plan above. (e.g., 'Swap Day 1 and Day 2', 'Add a coffee break after the museum on Day 1', 'Remove the park visit on Day 3')")

    # Display existing chat messages
    if 'quick_mode_chat_messages' in st.session_state:
        for message in st.session_state.quick_mode_chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.session_state.quick_mode_chat_messages = [] # Initialize if somehow missing

    # Chat input
    if user_prompt := st.chat_input("Enter your change request here...", key="quick_mode_chat_input"):
        # Add user message to state and display
        st.session_state.quick_mode_chat_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Prepare data for the modification function
        current_itinerary = st.session_state.quick_mode_itinerary_data # Already checked this exists
        if not current_itinerary:
             # This check is slightly redundant given the outer check, but safe
             st.error("Internal Error: Cannot modify, itinerary data lost.")
             st.stop()

        try:
            current_itinerary_json_str = json.dumps(current_itinerary, indent=2) # Use indent for readability
        except TypeError as e:
            st.error(f"Internal Error: Could not serialize current itinerary to JSON: {e}")
            st.stop()

        # Get original context used for generation (needed by the modification agent)
        original_location = st.session_state.get('quick_mode_location', '')
        original_prefs_str = st.session_state.get('quick_mode_prefs', '')
        original_prefs = [original_prefs_str] if original_prefs_str else []
        original_budget = "Any" # Assuming Quick mode doesn't have explicit budget

        # Call the modification function
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔄 Thinking about your request...")
            # Using st.spinner for visual feedback during the API call
            with st.spinner("Asking AI to modify the itinerary..."):
                try:
                    new_itinerary_json_str, error_msg = modify_detailed_itinerary_gemini(
                        current_itinerary_json=current_itinerary_json_str,
                        user_request=user_prompt,
                        destination=original_location,
                        prefs=original_prefs,
                        budget=original_budget
                    )
                except Exception as agent_error:
                     st.error(f"Error calling modification agent: {agent_error}")
                     new_itinerary_json_str = None
                     error_msg = f"An unexpected error occurred while trying to modify the plan: {agent_error}"


            if error_msg:
                # --- ADDED: Print AI Explanation/Error ---
                print("\n--- DEBUG: AI Modification Error/Explanation ---")
                print(error_msg)
                print("--- END DEBUG ---\n")
                # --- END ADDED ---

                # AI explained why it couldn't modify or an error occurred
                response_content = f"⚠️ {error_msg}" # Prepend warning emoji
                message_placeholder.warning(response_content) # Use warning display for errors/info
                # Add assistant's explanation/error to chat history
                st.session_state.quick_mode_chat_messages.append({"role": "assistant", "content": response_content})

            elif new_itinerary_json_str:
                # --- ADDED: Print the RAW JSON string ---
                print("\n--- DEBUG: Raw JSON Received from Modify Agent ---")
                print(new_itinerary_json_str)
                print("--- END DEBUG ---\n")
                # --- END ADDED ---

                # Attempt to parse the new JSON
                try:
                    new_itinerary_data = json.loads(new_itinerary_json_str)

                    # --- ADDED: Print the PARSED data ---
                    print("\n--- DEBUG: Parsed Itinerary Data ---")
                    import pprint # Make sure to import pprint if not already done at top of file
                    pprint.pprint(new_itinerary_data) # Pretty print the Python dict
                    print("--- END DEBUG ---\n")
                    # --- END ADDED ---

                    # More robust validation of the structure
                    if (isinstance(new_itinerary_data, list) and
                        all(isinstance(d, dict) and 'day' in d and 'stops' in d and isinstance(d['stops'], list) for d in new_itinerary_data) and
                        all(isinstance(stop, dict) and 'name' in stop and 'coordinates' in stop for d in new_itinerary_data for stop in d['stops'])): # Check stops basic fields
                        # Even more robust check for coordinates format specifically (as per JS requirements)
                        valid_coords = True
                        for day in new_itinerary_data:
                            for stop in day['stops']:
                                coords = stop.get('coordinates')
                                if not (isinstance(coords, list) and len(coords) == 2 and
                                        isinstance(coords[0], (int, float)) and
                                        isinstance(coords[1], (int, float))):
                                    valid_coords = False
                                    print(f"DEBUG: Invalid coordinates found in parsed data for stop '{stop.get('name')}': {coords}") # Debug print
                                    break
                            if not valid_coords:
                                break

                        if valid_coords:
                             # Update successful!
                            response_content = "✅ OK, I've updated the itinerary based on your request. The map and plan above should refresh momentarily."
                            message_placeholder.success(response_content) # Use success display
                            # Add assistant's success message to chat history
                            st.session_state.quick_mode_chat_messages.append({"role": "assistant", "content": response_content})

                            # *** Update the main itinerary state ***
                            st.session_state.quick_mode_itinerary_data = new_itinerary_data

                            # *** Rerun to refresh the map/sidebar ***
                            time.sleep(1) # Brief pause allows user to see message before rerun
                            st.rerun()
                        else:
                             # JSON received and parsed, but coordinates are invalid
                            response_content = "⚠️ Sorry, the AI provided an updated plan, but the structure was invalid (specifically, the `coordinates` format was incorrect - must be a list of two numbers like `[longitude, latitude]`). Please try rephrasing your request or regenerating the plan."
                            message_placeholder.warning(response_content)
                            st.session_state.quick_mode_chat_messages.append({"role": "assistant", "content": response_content})
                            # Optionally log the invalid structure (already printed parsed data)
                    else:
                        # JSON received but invalid overall structure
                        response_content = "⚠️ Sorry, the AI provided an updated plan but its overall structure was invalid (missing required fields like 'day', 'stops', 'name', or 'coordinates'). Please try rephrasing your request or regenerating the plan."
                        message_placeholder.warning(response_content)
                        st.session_state.quick_mode_chat_messages.append({"role": "assistant", "content": response_content})
                        # Optionally log the invalid JSON for debugging:
                        # print("--- Invalid JSON Structure Received ---")
                        # print(new_itinerary_json_str)
                        # print("--- End Invalid JSON ---")

                except json.JSONDecodeError as e:
                    # Failed to parse the JSON string from AI
                    response_content = f"⚠️ Sorry, I received an invalid response from the AI and couldn't update the plan. The AI likely provided text explanation instead of JSON. Response received:\n```\n{new_itinerary_json_str}\n```"
                    message_placeholder.error(response_content) # Use error display
                    st.session_state.quick_mode_chat_messages.append({"role": "assistant", "content": response_content})
                    # Optionally log the bad response:
                    # print("--- Non-JSON/Invalid JSON Response Received ---")
                    # print(new_itinerary_json_str)
                    # print("--- End Non-JSON/Invalid JSON Response ---")
            else:
                 # Should not happen if modify func returns one or the other, but handle defensively
                 response_content = "🤔 Something unexpected happened. I didn't receive an update or an error message from the modification agent."
                 message_placeholder.warning(response_content)
                 st.session_state.quick_mode_chat_messages.append({"role": "assistant", "content": response_content})
    # --- <<< END Chat Interface >>> ---
else:
    # This case handles if itinerary_data was found to be invalid before the map attempt
    # Or if the map itself had an error but we still want to *try* showing chat
    # (Currently, the invalid data case above prevents chat, which might be desired)
    # If you wanted chat even with bad map data, you'd add the chat code here too.
    st.info("Cannot display modification chat as itinerary data is missing or invalid.")


