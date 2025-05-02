# src/pages/2_⚡_Quick_Mode.py
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
# Assumes running with `streamlit run src/main_app.py` from project root
from tools import cached_geocode_location # Use the cached version from tools.py
from itinerary_agent import brainstorm_places_for_quick_mode, generate_detailed_itinerary_gemini
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
st.title("⚡ Quick Mode: Instant Itinerary")
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

# --- 5. Button Logic (Generation Process) ---
if generate_button and not st.session_state.quick_mode_generating:
    # Reset previous results/errors
    st.session_state.quick_mode_itinerary_data = None
    st.session_state.quick_mode_geocoded_places = []
    st.session_state.quick_mode_error = None
    st.session_state.quick_mode_status_msgs = []
    st.session_state.quick_mode_generating = True
    st.rerun() # Rerun immediately to show the "Generating..." state and disable button

# --- 6. Status/Error Display ---
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
        # 1. Brainstorm Places
        step_text = "🧠 Brainstorming places with Gemini..."
        st.session_state.quick_mode_status_msgs.append(step_text)
        status_text_placeholder.info(step_text)
        progress_bar.progress(10, text=step_text)
        # No spinner needed here as we update the status text directly
        suggested_places = brainstorm_places_for_quick_mode(location, duration, prefs)

        if not suggested_places:
            st.session_state.quick_mode_error = "Failed to brainstorm places. The AI might not have suggestions, or an error occurred. Try adjusting your preferences."
            raise Exception(st.session_state.quick_mode_error)

        # 2. Geocode Places
        step_text = f"🗺️ Geocoding {len(suggested_places)} potential places..."
        st.session_state.quick_mode_status_msgs.append(step_text)
        status_text_placeholder.info(step_text)
        progress_bar.progress(30, text=step_text)

        geocoded_places_list = []
        failed_geocode = []
        total_places = len(suggested_places)
        for i, place_name in enumerate(suggested_places):
            # Update progress specifically for geocoding step
            geocode_progress_text = f"Geocoding: {place_name[:30]}... ({i+1}/{total_places})"
            progress_bar.progress(30 + int(60 * (i + 1) / total_places), text=geocode_progress_text)
            status_text_placeholder.info(geocode_progress_text) # Keep user updated

            geo_result = cached_geocode_location(place_name) # Use cached tool
            if geo_result:
                geocoded_places_list.append({
                    "place_name": place_name,
                    "latitude": geo_result["latitude"],
                    "longitude": geo_result["longitude"],
                    "address": geo_result["address"]
                })
            else:
                failed_geocode.append(place_name)
            # Optional small sleep if needed, even with caching
            # time.sleep(0.05)

        st.session_state.quick_mode_geocoded_places = geocoded_places_list

        if not geocoded_places_list:
            st.session_state.quick_mode_error = "Could not geocode any suggested places. Please check the destination or try again."
            raise Exception(st.session_state.quick_mode_error)

        if failed_geocode:
            # Display warning below the status text temporarily
            st.warning(f"Could not find coordinates for: {', '.join(failed_geocode)}. They won't be included.")

        # 3. Generate Detailed Itinerary
        step_text = f"✍️ Generating {num_days}-day detailed itinerary with Gemini..."
        st.session_state.quick_mode_status_msgs.append(step_text)
        status_text_placeholder.info(step_text)
        progress_bar.progress(90, text=step_text)

        activities_for_gemini = [
            {"place_name": p["place_name"], "latitude": p["latitude"], "longitude": p["longitude"]}
            for p in geocoded_places_list
        ]
        detailed_plan = generate_detailed_itinerary_gemini(
            activities=activities_for_gemini,
            num_days=num_days,
            destination=location,
            prefs=[prefs],
            budget="Any"
        )

        if detailed_plan:
            st.session_state.quick_mode_itinerary_data = detailed_plan
            status_text_placeholder.success("✅ Itinerary Generated!")
            progress_bar.progress(100)
            time.sleep(2) # Let user see success message
        else:
            st.session_state.quick_mode_error = "Failed to generate the detailed itinerary using the suggested places. The AI might have encountered an issue."
            raise Exception(st.session_state.quick_mode_error)

    except Exception as e:
        st.session_state.quick_mode_error = str(e) # Store error message
        status_text_placeholder.error(f"An error occurred: {e}")
        # Don't clear progress bar on error, keep it showing the error

    finally:
        # Clear placeholders only if generation finished (success or handled error)
        # On actual exception, they might stay showing the error
        if not st.session_state.quick_mode_error:
             progress_bar_placeholder.empty()
             status_text_placeholder.empty()
        # Reset generating flag regardless of outcome
        st.session_state.quick_mode_generating = False
        # Rerun one last time to update the UI state (remove progress, show results/final error)
        st.rerun()

# --- Error display (after generation attempt finishes) ---
# Placed here so it shows *after* the generation attempt is complete
elif st.session_state.quick_mode_error and not st.session_state.quick_mode_itinerary_data:
    st.error(f"Failed to generate itinerary: {st.session_state.quick_mode_error}")


# --- 7. Results Display ---
# This section now runs only when NOT generating and itinerary data IS present
if not st.session_state.quick_mode_generating and st.session_state.get('quick_mode_itinerary_data'):
    st.markdown("---")
    st.subheader("🗓️ Generated Itinerary & Map")
    try:
        itinerary_data = st.session_state.quick_mode_itinerary_data # Already checked it exists

        # Basic validation again just in case
        if not isinstance(itinerary_data, list) or not all(isinstance(day, dict) and 'stops' in day for day in itinerary_data):
             st.error("Generated itinerary data is invalid. Cannot display map.")
             st.json(itinerary_data)
        elif not MAPBOX_ACCESS_TOKEN:
            st.warning("Mapbox token missing. Displaying itinerary data as JSON instead of interactive map.")
            st.json(itinerary_data)
        else:
            # Convert to JSON string for HTML embedding
            try:
                itinerary_json = json.dumps(itinerary_data)
            except TypeError as json_error:
                st.error(f"Error converting itinerary data to JSON: {json_error}")
                st.json(itinerary_data)
                st.stop() # Stop if JSON conversion fails

            # Determine map center
            map_center_lon, map_center_lat = -9.1393, 38.7223 # Default
            initial_zoom = 11
            try: # Add try-except for robust coordinate handling
                if itinerary_data and itinerary_data[0].get('stops') and itinerary_data[0]['stops'][0].get('coordinates'):
                    first_coord = itinerary_data[0]['stops'][0]['coordinates']
                    if isinstance(first_coord, list) and len(first_coord) == 2 and all(isinstance(c, (int, float)) for c in first_coord):
                         if -180 <= first_coord[0] <= 180 and -90 <= first_coord[1] <= 90:
                             map_center_lon, map_center_lat = first_coord[0], first_coord[1]
                             initial_zoom = 12
                         else: st.warning(f"First stop coordinates {first_coord} out of range, using default.")
                    else: st.warning(f"First stop coordinates invalid ({first_coord}), using default.")
            except Exception as center_err:
                 st.warning(f"Could not determine center from first stop ({center_err}), using default.")


            map_height_detailed = 750

            # --- HTML Component ---
            # (Keep the interactive_map_html_with_sidebar f-string exactly as it was in the previous corrected version)
            interactive_map_html_with_sidebar = f"""
            <!DOCTYPE html><html lang="en"><head>
                <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Interactive Itinerary Map</title>
                <link href='https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.css' rel='stylesheet' />
                <script src='https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.js'></script>
                <style>
                    /* --- CSS --- */
                    body {{ margin: 0; padding: 0; font-family: 'Arial', sans-serif; overflow: hidden; }}
                    #app-container {{ display: flex; height: {map_height_detailed}px; width: 100%; position: relative; border: 1px solid #ddd; }}
                    #sidebar {{ width: 340px; background-color: #f8f9fa; padding: 15px; box-shadow: 2px 0 5px rgba(0,0,0,0.1); overflow-y: auto; z-index: 10; border-right: 1px solid #dee2e6; height: 100%; box-sizing: border-box; }}
                    #sidebar h2 {{ margin-top: 0; color: #343a40; border-bottom: 1px solid #ced4da; padding-bottom: 10px; font-size: 1.2em; }}
                    .day-header {{ background-color: #007bff; color: white; padding: 8px 12px; margin-top: 15px; margin-bottom: 5px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background-color 0.2s ease; }} .day-header:hover {{ background-color: #0056b3; }} .day-header:first-of-type {{ margin-top: 5px; }}
                    .day-stops {{ list-style: none; padding: 0; margin: 0 0 15px 0; border-left: 3px solid transparent; padding-left: 10px; transition: border-left-color 0.3s ease; }} .day-stops.active {{ border-left-color: #007bff; }}
                    .destination-item {{ padding: 8px 5px; cursor: pointer; border-bottom: 1px solid #e9ecef; transition: background-color 0.2s ease; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; }} .destination-item:hover {{ background-color: #e9ecef; }} .destination-item:last-child {{ border-bottom: none; }}
                    .destination-item span.time {{ font-weight: bold; color: #6c757d; margin-right: 8px; flex-shrink: 0; width: 50px; text-align: right; }} .destination-item span.name {{ flex-grow: 1; text-align: left; margin-left: 5px; }} .destination-item span.type-prefix {{ font-style: italic; color: #555; margin-right: 5px; font-size: 0.85em; background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; border: 1px solid #ddd; white-space: nowrap; }}
                    .destination-item--lunch, .destination-item--dinner {{ background-color: #fff3e0; }} .destination-item--lunch:hover, .destination-item--dinner:hover {{ background-color: #ffe0b2; }} .destination-item--break {{ background-color: #e3f2fd; }} .destination-item--break:hover {{ background-color: #bbdefb; }} .destination-item--museum {{ background-color: #ede7f6; }} .destination-item--museum:hover {{ background-color: #d1c4e9; }} .destination-item--park {{ background-color: #e8f5e9; }} .destination-item--park:hover {{ background-color: #c8e6c9; }} .destination-item--viewpoint {{ background-color: #fce4ec; }} .destination-item--viewpoint:hover {{ background-color: #f8bbd0; }}
                    #map {{ flex-grow: 1; height: 100%; }}
                    #info-panel {{ position: absolute; bottom: 20px; left: 360px; width: 300px; max-height: 40%; background-color: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); z-index: 20; display: none; overflow-y: auto; font-size: 0.9em; border: 1px solid #ccc; }} #info-panel h4 {{ margin-top: 0; margin-bottom: 10px; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }} #info-panel p {{ margin-bottom: 5px; line-height: 1.4; color: #555; }} #info-panel-close {{ position: absolute; top: 5px; right: 8px; background: none; border: none; font-size: 1.2em; font-weight: bold; cursor: pointer; color: #888; }} #info-panel-close:hover {{ color: #333; }}
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
                    // --- Javascript with doubled braces ---
                    mapboxgl.accessToken = '{MAPBOX_ACCESS_TOKEN}';
                    const itineraryData = {itinerary_json};
                    const mapCenter = [{map_center_lon}, {map_center_lat}];
                    const initialZoom = {initial_zoom};
                    const routeLayerId = 'route-line-layer';
                    const routeSourceId = 'route-line-source';
                    const markers = [];
                    const map = new mapboxgl.Map({{
                        container: 'map', style: 'mapbox://styles/mapbox/standard',
                        center: mapCenter, zoom: initialZoom, pitch: 50, bearing: -10, antialias: true
                    }});
                    const itineraryContentEl = document.getElementById('itinerary-content');
                    const infoPanelEl = document.getElementById('info-panel');
                    const infoPanelTitleEl = document.getElementById('info-panel-title');
                    const infoPanelDescriptionEl = document.getElementById('info-panel-description');
                    const infoPanelCloseBtn = document.getElementById('info-panel-close');

                    function populateSidebar() {{
                        itineraryContentEl.innerHTML = '';
                        if (!itineraryData || !Array.isArray(itineraryData) || itineraryData.length === 0) {{
                             itineraryContentEl.innerHTML = '<p style="padding:10px; color:#dc3545;">Error: Invalid or empty itinerary data provided.</p>'; console.error("Invalid itineraryData:", itineraryData); return;
                        }}
                        try {{
                            itineraryData.forEach((dayData, dayIndex) => {{
                                if (!dayData || typeof dayData !== 'object') {{ console.warn(`Skipping invalid day data at index ${{dayIndex}}`); return; }}
                                const dayNum = dayData.day || (dayIndex + 1); const dayTitle = dayData.title || `Day ${{dayNum}}`;
                                const dayHeader = document.createElement('div'); dayHeader.className = 'day-header'; dayHeader.textContent = dayTitle; dayHeader.setAttribute('data-day', dayNum); itineraryContentEl.appendChild(dayHeader);
                                const stopsList = document.createElement('ul'); stopsList.className = 'day-stops'; stopsList.setAttribute('data-day', dayNum);
                                if (!dayData.stops || !Array.isArray(dayData.stops) || dayData.stops.length === 0) {{
                                    stopsList.innerHTML = '<li style="padding: 5px 0; color: #6c757d;">_No activities scheduled._</li>';
                                }} else {{
                                    dayData.stops.forEach((stop, stopIndex) => {{
                                        if (!stop || typeof stop !== 'object' || !stop.coordinates || !Array.isArray(stop.coordinates) || stop.coordinates.length !== 2 || typeof stop.coordinates[0] !== 'number' || typeof stop.coordinates[1] !== 'number') {{
                                             console.warn(`Skipping invalid stop data at Day ${{dayNum}}, index ${{stopIndex}}:`, stop);
                                             const errorItem = document.createElement('li'); errorItem.style.color = 'red'; errorItem.style.fontSize = '0.8em'; errorItem.textContent = `[Invalid Stop Data ${{stopIndex+1}}]`; stopsList.appendChild(errorItem); return;
                                        }}
                                        const listItem = document.createElement('li'); listItem.className = 'destination-item'; const stopType = stop.type ? String(stop.type).toLowerCase().replace(/\s+/g, '-') : 'sightseeing'; listItem.classList.add(`destination-item--${{stopType}}`);
                                        listItem.setAttribute('data-lng', stop.coordinates[0]); listItem.setAttribute('data-lat', stop.coordinates[1]); listItem.setAttribute('data-zoom', stop.zoom || 16); listItem.setAttribute('data-pitch', stop.pitch || 50); listItem.setAttribute('data-bearing', stop.bearing || 0);
                                        const stopName = stop.name || 'Unnamed Stop'; listItem.setAttribute('data-name', stopName); listItem.setAttribute('data-description', stop.description || ''); listItem.setAttribute('data-type', stopType);
                                        let typePrefixHTML = ''; const defaultTypes = ['sightseeing', 'activity']; if (stop.type && !defaultTypes.includes(stopType)) {{ typePrefixHTML = `<span class="type-prefix">${{stop.type}}</span>`; }}
                                        listItem.innerHTML = `<span class="time">${{stop.time || ''}}</span>${{typePrefixHTML}}<span class="name">${{stopName}}</span>`;
                                        listItem.addEventListener('click', handleStopClick); stopsList.appendChild(listItem); addMarker(stop);
                                    }});
                                }}
                                itineraryContentEl.appendChild(stopsList);
                                dayHeader.addEventListener('click', (e) => {{
                                    const day = parseInt(e.currentTarget.getAttribute('data-day'));
                                    if (!isNaN(day)) {{ highlightDay(day); drawRouteForDay(day); flyToDayBounds(day); hideInfoPanel(); }}
                                    else {{ console.error("Invalid day number on header:", e.currentTarget); }}
                                }});
                            }});
                        }} catch (error) {{ console.error("Error populating sidebar:", error); itineraryContentEl.innerHTML = `<p style="padding:10px; color:#dc3545;">Error rendering itinerary details.</p>`; }}
                    }}
                    function handleStopClick(e) {{
                        e.stopPropagation(); const target = e.currentTarget; try {{
                            const lng = parseFloat(target.getAttribute('data-lng')); const lat = parseFloat(target.getAttribute('data-lat')); const zoom = parseFloat(target.getAttribute('data-zoom')); const pitch = parseFloat(target.getAttribute('data-pitch')); const bearing = parseFloat(target.getAttribute('data-bearing'));
                            const name = target.getAttribute('data-name') || 'Location'; const description = target.getAttribute('data-description') || 'No details provided.';
                            if (isNaN(lng) || isNaN(lat) || isNaN(zoom) || isNaN(pitch) || isNaN(bearing)) {{ console.error("Parsing error in handleStopClick for:", name); showInfoPanel(name, `Error: Invalid location data.`); return; }}
                            document.querySelectorAll('.destination-item').forEach(item => {{ item.style.fontWeight = 'normal'; item.style.backgroundColor = ''; }}); target.style.fontWeight = 'bold'; target.style.backgroundColor = '#d6eaff';
                            map.flyTo({{ center: [lng, lat], zoom: zoom, pitch: pitch, bearing: bearing, essential: true, speed: 1.2, curve: 1.4 }});
                            showInfoPanel(name, description);
                        }} catch (parseError) {{ console.error("Error processing stop click:", parseError); showInfoPanel("Error", "Could not process stop details."); }}
                    }}
                    function addMarker(stop) {{
                        if (!stop || !stop.coordinates || !Array.isArray(stop.coordinates) || stop.coordinates.length !== 2 || typeof stop.coordinates[0] !== 'number' || typeof stop.coordinates[1] !== 'number') {{ console.warn("Skipping marker invalid coords:", stop); return; }}
                        const el = document.createElement('div'); el.className = 'mapboxgl-marker'; el.style.backgroundColor = getTypeColor(stop.type);
                        const popup = new mapboxgl.Popup({{ offset: 25, closeButton: false }}).setHTML(`<b>${{stop.name || 'Unnamed'}}</b><br>${{stop.time || ''}}`);
                        const marker = new mapboxgl.Marker(el).setLngLat(stop.coordinates).setPopup(popup).addTo(map);
                        el.addEventListener('mouseenter', () => marker.togglePopup()); el.addEventListener('mouseleave', () => marker.togglePopup()); markers.push(marker);
                    }}
                    function getTypeColor(type) {{
                        const typeLower = type ? String(type).toLowerCase().replace(/\s+/g, '-') : 'sightseeing'; switch (typeLower) {{
                            case 'lunch': case 'dinner': case 'food': case 'restaurant': return '#FFA726'; case 'break': case 'cafe': case 'coffee': return '#42A5F5'; case 'museum': case 'gallery': case 'culture': case 'history': return '#AB47BC'; case 'park': case 'garden': case 'nature': return '#66BB6A'; case 'viewpoint': case 'landmark': case 'sightseeing': return '#EC407A'; case 'shopping': case 'market': return '#FFCA28'; case 'activity': case 'tour': case 'show': return '#26A69A'; case 'hotel': case 'accommodation': return '#78909C'; default: return '#FF5252';
                        }}
                    }}
                    function highlightDay(dayNum) {{
                        document.querySelectorAll('.day-stops').forEach(ul => ul.classList.remove('active')); const activeList = document.querySelector(`.day-stops[data-day="${{dayNum}}"]`); if (activeList) {{ activeList.classList.add('active'); }} else {{ console.warn("No stops list for day:", dayNum); }}
                        document.querySelectorAll('.day-header').forEach(hdr => hdr.style.backgroundColor = '#007bff'); const activeHdr = document.querySelector(`.day-header[data-day="${{dayNum}}"]`); if (activeHdr) {{ activeHdr.style.backgroundColor = '#0056b3'; }} else {{ console.warn("No header for day:", dayNum); }}
                    }}
                    function drawRouteForDay(dayNum) {{
                        const dayData = itineraryData ? itineraryData.find(d => (d.day || (itineraryData.indexOf(d) + 1)) === dayNum) : null;
                        if (!dayData || !dayData.stops || !Array.isArray(dayData.stops) || dayData.stops.length < 1) {{
                            if (map.getLayer(routeLayerId)) map.removeLayer(routeLayerId); if (map.getSource(routeSourceId)) map.removeSource(routeSourceId); return;
                        }}
                        const coords = dayData.stops.map(s => s.coordinates).filter(c => c && Array.isArray(c) && c.length === 2 && typeof c[0] === 'number' && typeof c[1] === 'number');
                        if (coords.length < 1) {{ if (map.getLayer(routeLayerId)) map.removeLayer(routeLayerId); if (map.getSource(routeSourceId)) map.removeSource(routeSourceId); return; }}
                        const geojson = {{ 'type': 'Feature', 'properties': {{}}, 'geometry': {{ 'type': 'LineString', 'coordinates': coords }} }};
                        const source = map.getSource(routeSourceId); if (source) {{ source.setData(geojson); }}
                        else {{ map.addSource(routeSourceId, {{ 'type': 'geojson', 'data': geojson }}); let firstSymbolId; const layers = map.getStyle().layers; for (let i = 0; i < layers.length; i++) {{ if (layers[i].type === 'symbol') {{ firstSymbolId = layers[i].id; break; }} }} map.addLayer({{ 'id': routeLayerId, 'type': 'line', 'source': routeSourceId, 'layout': {{ 'line-join': 'round', 'line-cap': 'round' }}, 'paint': {{ 'line-color': '#ff5722', 'line-width': 4, 'line-opacity': 0.8 }} }}, firstSymbolId); }}
                    }}
                    function flyToDayBounds(dayNum) {{
                        const dayData = itineraryData ? itineraryData.find(d => (d.day || (itineraryData.indexOf(d) + 1)) === dayNum) : null; if (!dayData || !dayData.stops || !Array.isArray(dayData.stops)) {{ console.warn("Invalid data flyToDayBounds day:", dayNum); return; }}
                        const coords = dayData.stops.map(s => s.coordinates).filter(c => c && Array.isArray(c) && c.length === 2 && typeof c[0] === 'number' && typeof c[1] === 'number'); if (coords.length === 0) {{ console.warn("No valid coords flyToDayBounds day:", dayNum); return; }}
                        if (coords.length === 1) {{ map.flyTo({{ center: coords[0], zoom: 15, pitch: 50, duration: 1500 }}); }}
                        else {{ const bounds = new mapboxgl.LngLatBounds(); coords.forEach(coord => bounds.extend(coord)); map.fitBounds(bounds, {{ padding: {{ top: 50, bottom: 50, left: 380, right: 50 }}, maxZoom: 16, pitch: 45, duration: 1500 }}); }}
                    }}
                    function showInfoPanel(title, description) {{ infoPanelTitleEl.textContent = title || "Details"; infoPanelDescriptionEl.textContent = description || "No additional details."; infoPanelEl.style.display = 'block'; }}
                    function hideInfoPanel() {{ infoPanelEl.style.display = 'none'; document.querySelectorAll('.destination-item').forEach(item => {{ item.style.fontWeight = 'normal'; item.style.backgroundColor = ''; }}); }}

                    infoPanelCloseBtn.addEventListener('click', hideInfoPanel); map.on('click', hideInfoPanel);
                    map.on('style.load', () => {{
                        console.log("Quick Mode: Map style loaded.");
                        if (!map.getSource('mapbox-dem')) {{ map.addSource('mapbox-dem', {{ 'type': 'raster-dem', 'url': 'mapbox://mapbox.mapbox-terrain-dem-v1', 'tileSize': 512, 'maxzoom': 14 }}); }}
                        map.setTerrain({{ 'source': 'mapbox-dem', 'exaggeration': 1.5 }});
                        if (!map.getLayer('sky')) {{ map.addLayer({{ 'id': 'sky', 'type': 'sky', 'paint': {{ 'sky-type': 'atmosphere', 'sky-atmosphere-sun': [0.0, 0.0], 'sky-atmosphere-sun-intensity': 5 }} }}); }}
                        populateSidebar();
                         if (itineraryData && itineraryData.length > 0) {{ const firstDayNum = itineraryData[0].day || 1; highlightDay(firstDayNum); drawRouteForDay(firstDayNum); /* flyToDayBounds(firstDayNum); */ }}
                         else {{ console.log("Quick Mode: No initial itinerary data."); }}
                         console.log("Quick Mode: Sidebar populated, setup complete.");
                    }});
                    map.addControl(new mapboxgl.NavigationControl(), 'top-right'); map.addControl(new mapboxgl.FullscreenControl(), 'top-right'); map.addControl(new mapboxgl.ScaleControl());
                    window.addEventListener('resize', () => {{ map.resize(); }});
                    map.on('error', (e) => console.error("Mapbox GL Error:", e.error?.message || e));
                    console.log("Quick Mode: Interactive map setup initiated.");
                </script>
            </body></html>
            """

            # Render the map component
            components.html(interactive_map_html_with_sidebar, height=map_height_detailed + 20, scrolling=False)

            # Optionally add an expander to view the raw JSON data
            with st.expander("View Raw Itinerary Data (JSON)"):
                st.json(itinerary_data)

    except Exception as e:
        # Catch errors specifically during map display
        st.error(f"Error displaying interactive map: {e}")
        # Attempt to show the data that caused the error
        st.json(st.session_state.get('quick_mode_itinerary_data', {"error": "Could not retrieve itinerary data for display."}))

# --- 8. Default Message ---
# Shown only if not generating, no error is displayed, and no itinerary data exists
elif not st.session_state.quick_mode_generating and not st.session_state.quick_mode_error and not st.session_state.quick_mode_itinerary_data:
     st.info("Enter your trip details above and click 'Generate Quick Plan' to get started.")