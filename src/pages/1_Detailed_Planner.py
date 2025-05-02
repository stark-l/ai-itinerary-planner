# src/app.py

import streamlit as st
import google.generativeai as genai
import re
import time
import pandas as pd
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import json
from tools import geocode_location, cached_geocode_location

# REMOVE basic itinerary import, KEEP detailed one
# from itinerary_agent import create_basic_itinerary, generate_detailed_itinerary_gemini
from itinerary_agent import generate_detailed_itinerary_gemini
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
GEMINI_MODEL = 'gemini-1.5-flash-latest'
GEOCODER_USER_AGENT = "ai_travel_planner_app_v0.4_gemini" # Increment version

# --- Gemini API Configuration ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("🔴 Error: GOOGLE_API_KEY environment variable not found.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"🔴 Error configuring Google AI SDK: {e}")
    st.stop()

# --- Mapbox Configuration ---
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_ACCESS_TOKEN:
    st.warning("⚠️ Mapbox Access Token missing. Interactive itinerary map requires MAPBOX_ACCESS_TOKEN in .env.")
    # Decide if you want to stop or just disable the map feature later
    # st.stop()


# --- Helper Functions ---
def get_pool_items(all_curated_geocoded, itinerary_data):
    """Identifies items not currently scheduled in the itinerary."""
    if not itinerary_data:
        return all_curated_geocoded # If no itinerary, all are available

    scheduled_ids = set()
    # Create a unique ID for comparison (e.g., name + lat/lon)
    for day_plan in itinerary_data:
        for stop in day_plan.get('stops', []):
            # Use a robust identifier - coordinates are good if precise
            stop_id = f"{stop.get('name', '')}_{stop.get('coordinates', [0,0])[0]:.4f}_{stop.get('coordinates', [0,0])[1]:.4f}"
            scheduled_ids.add(stop_id)

    pool = []
    for item in all_curated_geocoded:
        item_id = f"{item.get('place_name', '')}_{item.get('latitude', 0):.4f}_{item.get('longitude', 0):.4f}"
        if item_id not in scheduled_ids:
            pool.append(item)
    return pool


def parse_suggestions(response_text):
    # (Keep your existing parse_suggestions function)
    suggestions = []
    lines = [line.strip() for line in response_text.strip().split('\n')]
    pattern1 = re.compile(r"^[*\-\d]*\.?\s*\*\*(.*?)\*\*\s*[:\-]?\s*(.*)")
    pattern2 = re.compile(r"^[*\-\d]*\.?\s*\*\*(.*?)\*\*$")
    pattern3 = re.compile(r"^[*\-\d]+\.?\s+(.*)")

    for line in lines:
        match1 = pattern1.match(line)
        match2 = pattern2.match(line)
        match3 = pattern3.match(line)
        display_text = line
        place_name = None

        if match1:
            place_name = match1.group(1).strip()
            description = match1.group(2).strip()
            display_text = f"**{place_name}**: {description}" if description else f"**{place_name}**"
        elif match2:
            place_name = match2.group(1).strip()
            display_text = f"**{place_name}**"
        elif match3:
            place_name = match3.group(1).strip()
            display_text = place_name
        elif len(line) > 5: # Fallback: treat the whole line as place name if it's reasonably long
             place_name = line.strip('*').strip('-').strip('.').strip() # Basic cleaning
             display_text = line # Keep original display text formatting

        if place_name:
            # Ensure place_name doesn't contain markdown meant for display_text only
            cleaned_place_name = re.sub(r'\*|:', '', place_name).strip()
            if cleaned_place_name: # Only add if we have a non-empty name after cleaning
                 suggestions.append({"display_text": display_text, "place_name": cleaned_place_name})
            else:
                # Handle cases where cleaning results in empty string, maybe log?
                print(f"DEBUG: Skipped suggestion due to empty place_name after cleaning: {line}")
    return suggestions

def update_map_data():
    # (Keep your existing update_map_data function)
    geocoded_for_map = []
    if 'curated_list' in st.session_state and 'geocoded_locations' in st.session_state:
        for item_dict in st.session_state.curated_list:
            display_text = item_dict['display_text']
            geo_info = st.session_state.geocoded_locations.get(display_text)
            if geo_info:
                 map_item = {
                     'lat': geo_info['latitude'],
                     'lon': geo_info['longitude'],
                     'name': geo_info.get('place_name', item_dict.get('place_name', display_text))
                 }
                 geocoded_for_map.append(map_item)

    if geocoded_for_map:
        st.session_state.map_data = pd.DataFrame(geocoded_for_map)
    else:
        st.session_state.map_data = pd.DataFrame()

# --- Initialize Session State ---
# (Keep existing initializations)
if 'messages' not in st.session_state: st.session_state.messages = []
if 'curated_list' not in st.session_state: st.session_state.curated_list = []
if 'latest_suggestions' not in st.session_state: st.session_state.latest_suggestions = []
if 'geocoded_locations' not in st.session_state: st.session_state.geocoded_locations = {}
if 'map_data' not in st.session_state: st.session_state.map_data = pd.DataFrame()
if 'confirm_remove_item' not in st.session_state: st.session_state.confirm_remove_item = None
if 'location' not in st.session_state: st.session_state.location = ""
if 'duration' not in st.session_state: st.session_state.duration = ""
if 'activity_prefs' not in st.session_state: st.session_state.activity_prefs = []
if 'budget_pref' not in st.session_state: st.session_state.budget_pref = "Any"
# Map view state initialization (for brainstorm map)
if 'map_zoom' not in st.session_state: st.session_state.map_zoom = 11
if 'map_pitch' not in st.session_state: st.session_state.map_pitch = 45
if 'map_bearing' not in st.session_state: st.session_state.map_bearing = 0
if 'map_terrain_exaggeration' not in st.session_state: st.session_state.map_terrain_exaggeration = 1.5
if 'map_style_selection' not in st.session_state: st.session_state.map_style_selection = "mapbox://styles/mapbox/satellite-streets-v12"

# Brainstorm map view state initialization (Simplified 2D)
if 'brainstorm_map_zoom' not in st.session_state:
    st.session_state.brainstorm_map_zoom = 11  # Default zoom for the 2D map
if 'brainstorm_map_style' not in st.session_state:
    st.session_state.brainstorm_map_style = "mapbox://styles/mapbox/streets-v12" # Default 2D style

# NEW: Initialize state for the detailed itinerary
if 'detailed_itinerary_data' not in st.session_state: st.session_state.detailed_itinerary_data = None
# REMOVE old state if it exists
if 'generated_itinerary' in st.session_state: del st.session_state['generated_itinerary']

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="AI Travel Planner") # Simplified title
st.title("AI Travel Planner (Gemini Powered)")

# --- Section 1: Define Your Trip ---
# (Keep Section 1 code as is)
st.header("1. Define Your Trip")
col1, col2 = st.columns(2)
with col1:
    st.text_input("Destination:", placeholder="e.g., Lisbon, Portugal", key='location')
    st.text_input("Trip Duration (optional):", placeholder="e.g., 5 days", key='duration')
with col2:
    st.multiselect(
        "Activity Preferences (optional):",
        ["Nature", "History", "Food", "Adventure", "Relaxation", "Culture", "Nightlife"],
        placeholder="Select one or more", key='activity_prefs'
    )
    st.selectbox(
        "Budget Style (optional):",
        ["Any", "Budget-friendly", "Mid-range", "Luxury"],
        index=0, key='budget_pref'
    )


# --- Section 2: Brainstorm Activities ---
st.header("2. Brainstorm Activities")

# --- System Prompt (Keep structure, adjust for Gemini nuances if needed later) ---
# Note: Gemini doesn't explicitly use a 'system' role like some models.
# We can pass system instructions using `system_instruction` in GenerativeModel
# or include them implicitly at the start of the conversation.
# Let's try the `system_instruction` approach with Gemini 1.5 models.

brainstorm_system_instruction = f"""You are a helpful travel brainstorming assistant. Your goal is to suggest individual activities, sights, or places based on the user's request and the trip context.

**Trip Context:**
*   **Destination:** {st.session_state.location or 'Not specified'}
*   **Duration:** {st.session_state.duration or 'Not specified'}
*   **Preferences:** {', '.join(st.session_state.activity_prefs) or 'None specified'}
*   **Budget:** {st.session_state.budget_pref or 'Not specified'}

**Your Task:**
1.  Analyze the user's latest request in the context of the ongoing conversation and the trip details above.
2.  Suggest a list of **specific, individual activities or places** relevant to their request.
3.  **IMPORTANT FORMATTING:** For each suggestion, put the main **Place Name in bold** using markdown (\*\*Place Name\*\*). You can optionally add a short description after the bolded name, perhaps separated by a colon or hyphen. Example: `**Oceanário de Lisboa**: Explore marine life.` or `**Belém Tower** - Iconic historical tower.`
4.  **DO NOT** create a daily schedule or itinerary at this stage. Just list potential options.
5.  **DO NOT** format in a table, just list out things in sentences.
6.  **DO NOT** include meta-commentary or think out loud.
7.  Provide around 3-7 suggestions per response unless the user asks for more/less.

Start suggesting based on the user's next message. Adhere strictly to the requested format."""


# Display Chat History (Optional - uncomment if needed)
# st.subheader("Chat History")
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# --- Chat Input ---
st.caption("Chat with Gemini to brainstorm activities:")
user_prompt = st.chat_input("Ask for suggestions (e.g., 'suggest some historical sites')")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # --- Prepare messages for Gemini API ---
    # Gemini expects roles 'user' and 'model'.
    # We'll format the history accordingly.
    gemini_history = []
    for msg in st.session_state.messages:
        role = "model" if msg["role"] == "assistant" else "user"
        # Gemini expects 'parts' which is a list of content pieces (here, just text)
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    # Ensure the last message is from the 'user' (which it should be here)
    # print("DEBUG: Gemini History:", gemini_history) # Debug print

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"): # Use 'assistant' for Streamlit display role
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("✨ Thinking with Gemini..."):
            try:
                # --- Initialize Gemini Model ---
                # Use system_instruction if model supports it (like Gemini 1.5 Pro/Flash)
                # For older models like gemini-1.0-pro, you might need to inject
                # the instructions into the first user message.
                model = genai.GenerativeModel(
                    GEMINI_MODEL,
                    system_instruction=brainstorm_system_instruction
                    # Add safety_settings if needed:
                    # safety_settings={'HARASSMENT':'BLOCK_NONE', ...}
                 )

                # --- Start a chat session (better for context management) ---
                chat = model.start_chat(history=gemini_history[:-1]) # History excluding the latest user prompt

                # --- Send the latest user prompt ---
                # The user_prompt is the last item in gemini_history
                response = chat.send_message(gemini_history[-1]['parts']) # Send only the content parts

                # --- Extract response ---
                # Handle potential blocks or errors
                if response.parts:
                     full_response = response.text
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                     full_response = f"⚠️ Request blocked by safety filter: {response.prompt_feedback.block_reason.name}"
                     st.warning(full_response)
                else:
                     # Handle unexpected empty response
                     full_response = "🤔 Gemini returned an empty response. Try rephrasing your request."
                     st.warning(full_response)


                message_placeholder.markdown(full_response) # Show response

                # --- Parse the response (Keep using existing parser) ---
                st.session_state.latest_suggestions = parse_suggestions(full_response)
                # print(f"DEBUG: Latest suggestions set: {st.session_state.latest_suggestions}")

            except Exception as e:
                st.error(f"🔴 An error occurred while contacting the Gemini API: {e}")
                full_response = "Sorry, I encountered an error connecting to the AI."
                message_placeholder.markdown(full_response)
                st.session_state.latest_suggestions = []

    # Append Gemini's response to session state history
    # Use role 'assistant' for consistency with Streamlit's display
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Section 3: Select Activities ---
# (Keep Section 3 code as is)
st.header("3. Select Activities")
if not st.session_state.latest_suggestions:
    st.info("Ask the AI for suggestions. Suggestions will appear here.")
else:
    st.markdown("Check the boxes next to activities you want to save:")
    curated_display_texts = {item['display_text'] for item in st.session_state.curated_list}
    newly_selected_dicts = []
    newly_deselected_dicts = []
    # Use enumerate to create unique keys even if display_text is duplicated in suggestions
    for idx, suggestion_dict in enumerate(st.session_state.latest_suggestions):
        display_text = suggestion_dict['display_text']
        place_name = suggestion_dict['place_name']
        # Create a more robust key using index and part of the text
        checkbox_key = f"cb_{idx}_{display_text[:30]}"
        is_in_curated = display_text in curated_display_texts
        # Checkbox state determines if it should be added or removed
        is_selected = st.checkbox(display_text, value=is_in_curated, key=checkbox_key)
        if is_selected and not is_in_curated:
            # Check if item with same display_text isn't already marked for addition
            if not any(n['display_text'] == display_text for n in newly_selected_dicts):
                 newly_selected_dicts.append(suggestion_dict)
        elif not is_selected and is_in_curated:
             # Check if item with same display_text isn't already marked for removal
             if not any(n['display_text'] == display_text for n in newly_deselected_dicts):
                 newly_deselected_dicts.append(suggestion_dict)
    needs_update = False
    if newly_selected_dicts:
        # Filter out potential duplicates before extending
        current_texts = {item['display_text'] for item in st.session_state.curated_list}
        unique_new = [d for d in newly_selected_dicts if d['display_text'] not in current_texts]
        st.session_state.curated_list.extend(unique_new)
        needs_update = True
    if newly_deselected_dicts:
        deselected_texts = {d['display_text'] for d in newly_deselected_dicts}
        # Filter the list, keeping items whose display_text is NOT in the deselected set
        st.session_state.curated_list = [item for item in st.session_state.curated_list if item['display_text'] not in deselected_texts]
        # Clean up geocoded locations for removed items
        for text in deselected_texts:
            if text in st.session_state.geocoded_locations:
                del st.session_state.geocoded_locations[text]
        needs_update = True
    if needs_update:
        update_map_data()
        st.rerun() # Rerun to update section 4 immediately after selection changes


# --- Section 4: Your Curated Activities & Geocoding ---
# (Keep Section 4 code as is)
st.header("4. Your Curated Activities & Geocoding")
if 'curated_list' not in st.session_state or not st.session_state.curated_list:
    st.write("No activities selected yet.")
else:
    cols_curated = st.columns([0.7, 0.3])
    # --- Column 1: Display List and Remove Buttons ---
    with cols_curated[0]:
        st.markdown("**Your Saved Items:**")
        items_to_remove_indices = []
        # Use enumerate to generate unique keys for remove buttons
        for i, item_dict in enumerate(st.session_state.curated_list):
            display_text = item_dict.get('display_text', f'Unknown Item {i}')
            place_name_for_key = item_dict.get('place_name', f'unknown_{i}')
            geo_status = ""
            if 'geocoded_locations' in st.session_state and display_text in st.session_state.geocoded_locations:
                geo_status = "📍 Geocoded" if st.session_state.geocoded_locations[display_text] else "❌ Not Found"
            item_row_cols = st.columns([0.8, 0.2])
            with item_row_cols[0]:
                 st.markdown(f"- {display_text} _{geo_status}_")
            with item_row_cols[1]:
                 # Use index `i` for uniqueness
                 unique_key_base = f"item_{i}_{place_name_for_key[:15]}"
                 if st.session_state.confirm_remove_item == i: # Use index for confirmation
                     if st.button("✔️", key=f"confirm_{unique_key_base}", help=f"Confirm removal", type="primary"):
                         items_to_remove_indices.append(i)
                         st.session_state.confirm_remove_item = None
                     if st.button("✖️", key=f"cancel_{unique_key_base}", help="Cancel removal"):
                         st.session_state.confirm_remove_item = None
                         st.rerun()
                 else:
                     if st.button("🗑️", key=f"request_{unique_key_base}", help=f"Remove {display_text}"):
                         st.session_state.confirm_remove_item = i # Store index to confirm
                         st.rerun()
        # Process removals based on indices
        if items_to_remove_indices:
            removed_items_texts = []
            # Iterate backwards to avoid index shifting issues
            for index in sorted(items_to_remove_indices, reverse=True):
                if 0 <= index < len(st.session_state.curated_list):
                    removed_item = st.session_state.curated_list.pop(index)
                    removed_items_texts.append(removed_item['display_text'])
                else:
                     st.warning(f"Tried to remove item at invalid index {index}")
            # Clean geocoded data for removed items by text
            for text in removed_items_texts:
                if text in st.session_state.geocoded_locations:
                    del st.session_state.geocoded_locations[text]
            update_map_data()
            st.toast(f"Removed {len(removed_items_texts)} item(s).")
            st.session_state.confirm_remove_item = None # Reset confirmation state
            st.rerun() # Rerun to update the list display

    # --- Column 2: Action Buttons (Geocode, Clear) ---
    with cols_curated[1]:
        st.markdown("**Actions:**")
        if st.button("🔄 Geocode Curated Activities"):
            # (Keep geocoding logic as is)
            st.session_state.geocoded_locations = {}
            total_items = len(st.session_state.curated_list)
            progress_bar = None
            if total_items > 0:
                 progress_bar = st.progress(0, text="Starting geocoding...")
                 for i, item_dict in enumerate(st.session_state.curated_list):
                     # Robustly get display text and place name
                     display_text = item_dict.get('display_text', f'Unknown Item {i}')
                     place_name_to_geocode = item_dict.get('place_name', None)
                     # If place_name is missing or empty, try display_text (after cleaning markdown)
                     if not place_name_to_geocode:
                         cleaned_display = re.sub(r'\*|:', '', display_text).strip()
                         place_name_to_geocode = cleaned_display if cleaned_display else None

                     if not place_name_to_geocode:
                         st.warning(f"Skipping item with unusable name: {display_text}")
                         st.session_state.geocoded_locations[display_text] = None # Mark as not found
                         if progress_bar: progress_bar.progress((i + 1) / total_items, text=f"Skipping invalid item")
                         continue # Skip to next item

                     status_text = f"Geocoding ({i+1}/{total_items}): {place_name_to_geocode[:30]}..."
                     if progress_bar: progress_bar.progress(i / total_items, text=status_text)

                     geo_result = cached_geocode_location(place_name_to_geocode) # Use cached tool
                     if geo_result:
                         result_to_store = {
                             "place_name": place_name_to_geocode, # Store the name actually used
                             "latitude": geo_result["latitude"],
                             "longitude": geo_result["longitude"],
                             "address": geo_result["address"]
                         }
                         st.session_state.geocoded_locations[display_text] = result_to_store
                     else:
                         st.session_state.geocoded_locations[display_text] = None # Mark as not found
                     # No artificial sleep needed if cached_geocode_tool handles Nominatim policy internally
                 if progress_bar: progress_bar.empty()
            update_map_data() # Rebuild map_data for brainstorm map
            found_count = sum(1 for v in st.session_state.geocoded_locations.values() if v is not None)
            if total_items > 0:
                 st.success(f"Geocoding attempt complete. Found coordinates for {found_count}/{total_items} items.")
                 if found_count < total_items:
                     st.warning("Some items could not be geocoded.")
            else:
                 st.info("No activities to geocode.")
            # Clear detailed itinerary if geocoding is re-run, as locations might change
            st.session_state.detailed_itinerary_data = None
            st.rerun()

        if st.button("Clear All Curated Activities"):
            st.session_state.curated_list = []
            st.session_state.geocoded_locations = {}
            st.session_state.latest_suggestions = [] # Also clear last suggestions
            st.session_state.map_data = pd.DataFrame() # Clear brainstorm map data
            st.session_state.confirm_remove_item = None
            st.session_state.detailed_itinerary_data = None # Clear detailed plan too
            st.toast("Curated list cleared!")
            st.rerun()


# --- Section 5: Simple 2D Brainstorm Map ---
st.header("5. Location Overview Map (2D)")
st.markdown("---")
# Define available 2D map styles
MAP_STYLES_2D = {
    "Streets": "mapbox://styles/mapbox/streets-v12",
    "Outdoors": "mapbox://styles/mapbox/outdoors-v12",
    "Light": "mapbox://styles/mapbox/light-v11",
    "Dark": "mapbox://styles/mapbox/dark-v11",
    "Satellite": "mapbox://styles/mapbox/satellite-v9", # Keep satellite as option
}
DEFAULT_STYLE_2D = "Streets"
DEFAULT_STYLE_URL_2D = MAP_STYLES_2D[DEFAULT_STYLE_2D]

if 'brainstorm_map_style' not in st.session_state:
    st.session_state.brainstorm_map_style = DEFAULT_STYLE_URL_2D
if st.session_state.brainstorm_map_style not in MAP_STYLES_2D.values():
     st.session_state.brainstorm_map_style = DEFAULT_STYLE_URL_2D

# Display map only if data exists
if not st.session_state.map_data.empty:
    map_df = st.session_state.map_data
    # --- Simplified Map Controls (Style & Zoom Only) ---
    map_control_cols = st.columns([0.6, 0.4])
    with map_control_cols[0]:
        current_style_name_2d = next((name for name, url in MAP_STYLES_2D.items() if url == st.session_state.brainstorm_map_style), DEFAULT_STYLE_2D)
        selected_style_name_2d = st.selectbox(
            "Map Style", options=list(MAP_STYLES_2D.keys()),
            index=list(MAP_STYLES_2D.keys()).index(current_style_name_2d),
            key="brainstorm_map_style_selector_2d", # Unique key
        )
        selected_style_url_2d = MAP_STYLES_2D[selected_style_name_2d]
        if st.session_state.brainstorm_map_style != selected_style_url_2d:
            st.session_state.brainstorm_map_style = selected_style_url_2d
            st.rerun()
    with map_control_cols[1]:
         # Use a separate zoom state if needed, or share? Let's share for now.
         st.session_state.brainstorm_map_zoom = st.slider("Zoom", 1, 18, st.session_state.brainstorm_map_zoom, 1, key="brainstorm_map_zoom_slider_2d")

    st.markdown("---")
    # --- Prepare Data & HTML for 2D Map ---
    locations_data = map_df.to_dict(orient='records')
    # Calculate center point only if data exists
    mid_lat = map_df["lat"].mean()
    mid_lon = map_df["lon"].mean()
    locations_json = json.dumps(locations_data)
    map_height = 450 # Smaller height for overview map

    # Simplified HTML/JS for 2D map
    brainstorm_map_html_2d = f"""
    <!DOCTYPE html><html><head>
        <meta charset="utf-8"><title>Overview Map</title>
        <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
        <link href="https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.css" rel="stylesheet">
        <script src="https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.js"></script>
        <style> body {{ margin: 0; padding: 0; }} #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }} </style>
    </head><body><div id="map"></div>
    <script>
        mapboxgl.accessToken = '{MAPBOX_ACCESS_TOKEN}';
        const locations = {locations_json};
        const map = new mapboxgl.Map({{ // Keep double braces
            container: 'map',
            style: '{st.session_state.brainstorm_map_style}', // Use 2D style state
            center: [{mid_lon}, {mid_lat}],
            zoom: {st.session_state.brainstorm_map_zoom}, // Use zoom state
            pitch: 0, // Force 2D view
            bearing: 0, // Force North-up
            antialias: true
        }});
        // Add markers after style loads
        map.on('load', () => {{ // Keep double braces
            if (locations && locations.length > 0) {{ // Keep double braces
                locations.forEach(loc => {{ // Keep double braces
                    // Simple uniform red marker for all brainstormed points
                    new mapboxgl.Marker({{color: "#FF4B4B"}}) // Keep double braces
                        .setLngLat([loc.lon, loc.lat])
                        .setPopup(new mapboxgl.Popup({{offset: 25}}).setText(loc.name || 'Location')) // Keep double braces
                        .addTo(map);
                }});
            }}
        }});
        // Add zoom and rotation controls
        map.addControl(new mapboxgl.NavigationControl({{visualizePitch: false}})); // Keep double braces, hide pitch control
    </script></body></html>
    """
    components.html(brainstorm_map_html_2d, height=map_height, scrolling=False)

    # Display failed items below this map
    if 'curated_list' in st.session_state and 'geocoded_locations' in st.session_state:
         failed_items = [ item['display_text'] for item in st.session_state.curated_list if item['display_text'] in st.session_state.geocoded_locations and st.session_state.geocoded_locations[item['display_text']] is None ]
         if failed_items:
             with st.expander("⚠️ Some locations could not be geocoded (not shown on map above):"):
                 for item_text in failed_items: st.markdown(f"- {item_text}")
else:
    st.info("Select and geocode activities in Section 4 to see the location overview map.")


# --- Section 6: Generate & View/Edit Interactive Itinerary ---
st.header("6. Generate & View/Edit Interactive Itinerary")
st.markdown("---")

# --- Input for Number of Days & Generate Button ---
# (Keep this part exactly as before)
num_curated_geocoded = len([
    item for item in st.session_state.get('curated_list', [])
    if st.session_state.geocoded_locations.get(item['display_text']) is not None
])
# ... (default_days_detailed calculation) ...
if num_curated_geocoded > 0:
    # Default to 3 days, but don't exceed the number of available geocoded activities.
    # This ensures the default 'value' is always <= 'max_value' in the number_input widget.
    default_days_detailed = min(3, max(1, num_curated_geocoded))
else:
    # If no activities, default to 1 (the input will be disabled anyway)
    default_days_detailed = 1

num_days_detailed = st.number_input(
    "Number of Days for Detailed Itinerary:", min_value=1,
    max_value=max(1, num_curated_geocoded), value=default_days_detailed,
    key='num_days_detailed_input', help="Requires successfully geocoded activities.",
    disabled=(num_curated_geocoded == 0)
)
if st.button("🚀 Generate Detailed Plan with Gemini", key="generate_detailed_button", disabled=(num_curated_geocoded == 0)):
    # (Keep the generation logic exactly as before - calls Gemini, stores result)
    st.session_state.detailed_itinerary_data = None
    # ... (collect geocoded_activities_list) ...
    geocoded_activities_list = []
    if 'geocoded_locations' in st.session_state and st.session_state.curated_list:
        for activity_dict in st.session_state.curated_list:
            display_text = activity_dict['display_text']
            geo_info = st.session_state.geocoded_locations.get(display_text)
            if geo_info:
                 activity_data_for_gemini = { "place_name": geo_info.get('place_name', activity_dict.get('place_name', 'Unknown')), "display_text": display_text, 'latitude': geo_info['latitude'], 'longitude': geo_info['longitude']}
                 geocoded_activities_list.append(activity_data_for_gemini)
    if not geocoded_activities_list: st.error("Cannot generate: No geocoded activities.")
    else:
        if len(geocoded_activities_list) < num_days_detailed: st.warning(f"Note: Fewer activities ({len(geocoded_activities_list)}) than days ({num_days_detailed}).")
        with st.spinner(f"Asking Gemini for a {num_days_detailed}-day detailed plan..."):
            detailed_plan = generate_detailed_itinerary_gemini( activities=geocoded_activities_list, num_days=num_days_detailed, destination=st.session_state.location, prefs=st.session_state.activity_prefs, budget=st.session_state.budget_pref)
            st.session_state.detailed_itinerary_data = detailed_plan
        if st.session_state.detailed_itinerary_data: st.success("✅ Detailed itinerary generated!")
        else: st.error("❌ Failed to generate detailed itinerary via Gemini.")
    st.rerun()

# --- Display Interactive Map Viewer (with integrated JS sidebar) ---
if st.session_state.get('detailed_itinerary_data'):
    st.subheader("Interactive Itinerary Map & Plan")
    try:
        itinerary_data = st.session_state.detailed_itinerary_data
        if not isinstance(itinerary_data, list) or not all('stops' in day for day in itinerary_data):
             st.error("Itinerary data invalid."); st.json(itinerary_data)
        else:
            itinerary_json = json.dumps(itinerary_data)
            map_center_lon, map_center_lat = -9.1393, 38.7223; initial_zoom = 11
            if itinerary_data and itinerary_data[0].get('stops') and itinerary_data[0]['stops'][0].get('coordinates'):
                first_coord = itinerary_data[0]['stops'][0]['coordinates']
                if len(first_coord) == 2: map_center_lon, map_center_lat = first_coord; initial_zoom = 12

            map_height_detailed = 750
            # *** Use the HTML that INCLUDES the JS Sidebar ***
            interactive_map_html_with_sidebar = f"""
            <!DOCTYPE html><html lang="en"><head>
                <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Interactive Itinerary Map</title>
                <link href='https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.css' rel='stylesheet' />
                <script src='https://api.mapbox.com/mapbox-gl-js/v3.4.0/mapbox-gl.js'></script>
                <style>
                    /* --- Paste the FULL CSS from the WORKING Map+Sidebar version --- */
                    body {{ margin: 0; padding: 0; font-family: 'Arial', sans-serif; overflow: hidden; }}
                    #app-container {{ display: flex; height: {map_height_detailed}px; width: 100%; position: relative; border: 1px solid #ddd; }}
                    #sidebar {{ width: 340px; background-color: #f8f9fa; padding: 15px; box-shadow: 2px 0 5px rgba(0,0,0,0.1); overflow-y: auto; z-index: 10; border-right: 1px solid #dee2e6; height: 100%; box-sizing: border-box; }}
                    #sidebar h2 {{ margin-top: 0; color: #343a40; border-bottom: 1px solid #ced4da; padding-bottom: 10px; font-size: 1.2em; }}
                    .day-header {{ background-color: #007bff; color: white; padding: 8px 12px; margin-top: 15px; margin-bottom: 5px; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background-color 0.2s ease; }} .day-header:hover {{ background-color: #0056b3; }} .day-header:first-of-type {{ margin-top: 5px; }}
                    .day-stops {{ list-style: none; padding: 0; margin: 0 0 15px 0; border-left: 3px solid transparent; padding-left: 10px; transition: border-left-color 0.3s ease; }} .day-stops.active {{ border-left-color: #007bff; }}
                    .destination-item {{ padding: 8px 5px; cursor: pointer; border-bottom: 1px solid #e9ecef; transition: background-color 0.2s ease; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; }} .destination-item:hover {{ background-color: #e9ecef; }} .destination-item:last-child {{ border-bottom: none; }}
                    .destination-item span.time {{ font-weight: bold; color: #6c757d; margin-right: 8px; flex-shrink: 0; width: 50px; text-align: right; }} .destination-item span.name {{ flex-grow: 1; text-align: left; margin-left: 5px; }} .destination-item span.type-prefix {{ font-style: italic; color: #555; margin-right: 5px; font-size: 0.85em; background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; border: 1px solid #ddd; white-space: nowrap; }}
                    /* Type Colors */ .destination-item--lunch, .destination-item--dinner {{ background-color: #fff3e0; }} .destination-item--lunch:hover, .destination-item--dinner:hover {{ background-color: #ffe0b2; }} .destination-item--break {{ background-color: #e3f2fd; }} .destination-item--break:hover {{ background-color: #bbdefb; }} .destination-item--museum {{ background-color: #ede7f6; }} .destination-item--museum:hover {{ background-color: #d1c4e9; }} .destination-item--park {{ background-color: #e8f5e9; }} .destination-item--park:hover {{ background-color: #c8e6c9; }} .destination-item--viewpoint {{ background-color: #fce4ec; }} .destination-item--viewpoint:hover {{ background-color: #f8bbd0; }}
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
                    // --- Paste the FULL JavaScript from the working Map+Sidebar version ---
                    // (Including map init, all helper functions, event listeners, map.on('style.load'))
                    // Ensure JS template literals use ${{...}}
                    mapboxgl.accessToken = '{MAPBOX_ACCESS_TOKEN}'; const itineraryData = {itinerary_json}; const mapCenter = [{map_center_lon}, {map_center_lat}]; const initialZoom = {initial_zoom}; const routeLayerId = 'route-line-layer'; const routeSourceId = 'route-line-source'; const markers = [];
                    const map = new mapboxgl.Map({{ container: 'map', style: 'mapbox://styles/mapbox/standard', center: mapCenter, zoom: initialZoom, pitch: 50, bearing: -10, antialias: true }});
                    const itineraryContentEl = document.getElementById('itinerary-content'); const infoPanelEl = document.getElementById('info-panel'); const infoPanelTitleEl = document.getElementById('info-panel-title'); const infoPanelDescriptionEl = document.getElementById('info-panel-description'); const infoPanelCloseBtn = document.getElementById('info-panel-close');
                    // ... (ALL JS functions: populateSidebar, handleStopClick, addMarker, getTypeColor, highlightDay, drawRouteForDay, flyToDayBounds, showInfoPanel, hideInfoPanel) ...
                    // Example: populateSidebar start
                    function populateSidebar() {{ itineraryContentEl.innerHTML = ''; if (!itineraryData || itineraryData.length === 0) {{ itineraryContentEl.innerHTML = '<p>Error: No itinerary data.</p>'; return; }} itineraryData.forEach(dayData => {{ const dayHeader = document.createElement('div'); dayHeader.className = 'day-header'; dayHeader.textContent = dayData.title || `Day ${{dayData.day}}`; dayHeader.setAttribute('data-day', dayData.day); itineraryContentEl.appendChild(dayHeader); const stopsList = document.createElement('ul'); stopsList.className = 'day-stops'; stopsList.setAttribute('data-day', dayData.day); if (!dayData.stops || dayData.stops.length === 0) {{ stopsList.innerHTML = '<li>No activities scheduled.</li>'; }} else {{ dayData.stops.forEach(stop => {{ if (!stop || !stop.coordinates || stop.coordinates.length !== 2) {{ console.warn("Skipping stop:", stop); return; }} const listItem = document.createElement('li'); listItem.className = 'destination-item'; const stopType = stop.type ? stop.type.toLowerCase().replace(/\s+/g, '-') : 'sightseeing'; listItem.classList.add(`destination-item--${{stopType}}`); listItem.setAttribute('data-lng', stop.coordinates[0]); listItem.setAttribute('data-lat', stop.coordinates[1]); listItem.setAttribute('data-zoom', stop.zoom || 16); listItem.setAttribute('data-pitch', stop.pitch || 50); listItem.setAttribute('data-bearing', stop.bearing || 0); listItem.setAttribute('data-name', stop.name || 'Unnamed'); listItem.setAttribute('data-description', stop.description || ''); listItem.setAttribute('data-type', stopType); let typePrefixHTML = ''; const defaultTypes = ['sightseeing', 'activity']; if (stop.type && !defaultTypes.includes(stopType)) {{ typePrefixHTML = `<span class="type-prefix">${{stop.type}}</span>`; }} listItem.innerHTML = `<span class="time">${{stop.time || ''}}</span>${{typePrefixHTML}}<span class="name">${{stop.name || 'Unnamed'}}</span>`; listItem.addEventListener('click', handleStopClick); stopsList.appendChild(listItem); addMarker(stop); }}); }} itineraryContentEl.appendChild(stopsList); dayHeader.addEventListener('click', (e) => {{ const day = parseInt(e.currentTarget.getAttribute('data-day')); highlightDay(day); drawRouteForDay(day); flyToDayBounds(day); hideInfoPanel(); }}); }}); }}
                    // ... (Rest of JS functions) ...
                    function handleStopClick(e) {{ e.stopPropagation(); const target = e.currentTarget; const lng = parseFloat(target.getAttribute('data-lng')); const lat = parseFloat(target.getAttribute('data-lat')); const zoom = parseFloat(target.getAttribute('data-zoom')); const pitch = parseFloat(target.getAttribute('data-pitch')); const bearing = parseFloat(target.getAttribute('data-bearing')); const name = target.getAttribute('data-name'); const description = target.getAttribute('data-description'); if (isNaN(lng+lat+zoom+pitch+bearing)) {{ console.error("Parse Error in handleStopClick"); return; }} document.querySelectorAll('.destination-item').forEach(i => i.style.fontWeight = 'normal'); target.style.fontWeight = 'bold'; map.flyTo({{center: [lng, lat], zoom: zoom, pitch: pitch, bearing: bearing, essential: true, speed: 1.2, curve: 1.4}}); showInfoPanel(name, description); }} function addMarker(stop) {{ const el = document.createElement('div'); el.className = 'mapboxgl-marker'; el.style.backgroundColor = getTypeColor(stop.type); const popup = new mapboxgl.Popup({{offset: 25, closeButton: false}}).setHTML(`<b>${{stop.name}}</b><br>${{stop.time || ''}}`); const marker = new mapboxgl.Marker(el).setLngLat(stop.coordinates).setPopup(popup).addTo(map); el.addEventListener('mouseenter', () => marker.togglePopup()); el.addEventListener('mouseleave', () => marker.togglePopup()); markers.push(marker); }} function getTypeColor(type) {{ const typeLower = type ? type.toLowerCase().replace(/\s+/g, '-') : 'sightseeing'; switch (typeLower) {{ case 'lunch': case 'dinner': return '#FFA726'; case 'break': return '#42A5F5'; case 'museum': return '#AB47BC'; case 'park': return '#66BB6A'; case 'viewpoint': return '#EC407A'; case 'shopping': return '#FFCA28'; default: return '#FF5252'; }} }} function highlightDay(dayNum) {{ document.querySelectorAll('.day-stops').forEach(ul => ul.classList.remove('active')); const activeList = document.querySelector(`.day-stops[data-day="${{dayNum}}"]`); if (activeList) activeList.classList.add('active'); document.querySelectorAll('.day-header').forEach(hdr => hdr.style.backgroundColor = '#007bff'); const activeHdr = document.querySelector(`.day-header[data-day="${{dayNum}}"]`); if (activeHdr) activeHdr.style.backgroundColor = '#0056b3'; }} function drawRouteForDay(dayNum) {{ const dayData = itineraryData.find(d => d.day === dayNum); if (!dayData || !dayData.stops || dayData.stops.length < 1) {{ if (map.getLayer(routeLayerId)) map.removeLayer(routeLayerId); if (map.getSource(routeSourceId)) map.removeSource(routeSourceId); return; }} const coords = dayData.stops.map(s => s.coordinates).filter(c => c && c.length === 2); if (coords.length < 1) {{ if (map.getLayer(routeLayerId)) map.removeLayer(routeLayerId); if (map.getSource(routeSourceId)) map.removeSource(routeSourceId); return; }} const geojson = {{'type': 'Feature', 'properties':{{}}, 'geometry': {{'type': 'LineString', 'coordinates': coords}}}}; if (map.getSource(routeSourceId)) {{ map.getSource(routeSourceId).setData(geojson); }} else {{ map.addSource(routeSourceId, {{'type': 'geojson', 'data': geojson}}); map.addLayer({{ 'id': routeLayerId, 'type': 'line', 'source': routeSourceId, 'layout': {{'line-join': 'round', 'line-cap': 'round'}}, 'paint': {{'line-color': '#ff5722', 'line-width': 4, 'line-opacity': 0.8}} }}, 'road-label'); }} }} function flyToDayBounds(dayNum) {{ const dayData = itineraryData.find(d => d.day === dayNum); if (!dayData || !dayData.stops || dayData.stops.length === 0) return; const coords = dayData.stops.map(s => s.coordinates).filter(c => c && c.length === 2); if (coords.length === 0) return; if (coords.length === 1) {{ map.flyTo({{center: coords[0], zoom: 15, pitch: 50}}); return; }} const bounds = new mapboxgl.LngLatBounds(); coords.forEach(c => bounds.extend(c)); map.fitBounds(bounds, {{padding: {{top: 50, bottom: 50, left: 380, right: 50}}, maxZoom: 16, pitch: 45, duration: 1500}}); }} function showInfoPanel(title, description) {{ infoPanelTitleEl.textContent = title; infoPanelDescriptionEl.textContent = description || "No details."; infoPanelEl.style.display = 'block'; }} function hideInfoPanel() {{ infoPanelEl.style.display = 'none'; document.querySelectorAll('.destination-item').forEach(item => item.style.fontWeight = 'normal'); }}
                    // Event listeners and init
                    infoPanelCloseBtn.addEventListener('click', hideInfoPanel); map.on('click', hideInfoPanel);
                    map.on('style.load', () => {{ console.log("Interactive map style loaded."); if (!map.getSource('mapbox-dem')) {{ map.addSource('mapbox-dem', {{'type': 'raster-dem', 'url': 'mapbox://mapbox.mapbox-terrain-dem-v1', 'tileSize': 512, 'maxzoom': 14}}); }} map.setTerrain({{ 'source': 'mapbox-dem', 'exaggeration': 1.5 }}); if (!map.getLayer('sky')) {{ map.addLayer({{ 'id': 'sky', 'type': 'sky', 'paint': {{ 'sky-type': 'atmosphere', 'sky-atmosphere-sun': [0.0, 0.0], 'sky-atmosphere-sun-intensity': 5 }} }}); }} populateSidebar(); /* Call AFTER style loaded */ if (itineraryData && itineraryData.length > 0) {{ highlightDay(itineraryData[0].day); drawRouteForDay(itineraryData[0].day); }} console.log("Sidebar populated, initial route drawn."); }});
                    map.addControl(new mapboxgl.NavigationControl(), 'top-right'); map.addControl(new mapboxgl.FullscreenControl(), 'top-right'); map.addControl(new mapboxgl.ScaleControl()); window.addEventListener('resize', () => {{ map.resize(); }}); map.on('error', (e) => console.error("Mapbox error:", e.error?.message || e)); console.log("Interactive map setup initiated.");
                </script>
            </body></html>
            """
            # Render the map component
            components.html(interactive_map_html_with_sidebar, height=map_height_detailed + 20, scrolling=False)

            # --- Itinerary Editor Expander (Below the Map) ---
            st.markdown("---") # Separator
            with st.expander("✏️ Edit Itinerary Plan", expanded=False):
                # Get the list of ALL potential activities (curated and geocoded)
                all_curated_geocoded = []
                if 'geocoded_locations' in st.session_state and st.session_state.curated_list:
                    # ... (logic to populate all_curated_geocoded - same as before) ...
                     for activity_dict in st.session_state.curated_list:
                         display_text = activity_dict['display_text']
                         geo_info = st.session_state.geocoded_locations.get(display_text)
                         if geo_info: all_curated_geocoded.append({ "place_name": geo_info.get('place_name', activity_dict.get('place_name', 'Unknown')), "display_text": display_text, 'latitude': geo_info['latitude'], 'longitude': geo_info['longitude'] })

                # Get items currently available (not in the itinerary)
                pool_items = get_pool_items(all_curated_geocoded, itinerary_data)

                # Use columns for better layout within the expander
                edit_col_pool, edit_col_plan = st.columns(2)

                with edit_col_pool:
                    st.markdown("**Available Activities (Pool):**")
                    if not pool_items:
                        st.caption("_All curated items are scheduled._")
                    else:
                        # Use a container for scrolling if list is long
                        with st.container(height=400): # Adjust height as needed
                            for i, item in enumerate(pool_items):
                                item_id = f"{item.get('place_name', '')}_{item.get('latitude', 0):.4f}_{item.get('longitude', 0):.4f}"
                                cols_pool_item = st.columns([0.7, 0.3])
                                with cols_pool_item[0]:
                                    st.markdown(f"- {item.get('place_name', 'Unknown')}")
                                with cols_pool_item[1]:
                                    add_day_key = f"add_{item_id}_{i}"
                                    day_options = [f"Day {d['day']}" for d in itinerary_data]
                                    selected_day_to_add = st.selectbox(
                                        "Add:", ["-"] + day_options, key=add_day_key, label_visibility="collapsed", index=0
                                    )
                                    if selected_day_to_add != "-":
                                        day_num_to_add = int(selected_day_to_add.split(" ")[1])
                                        for day_plan in st.session_state.detailed_itinerary_data:
                                            if day_plan['day'] == day_num_to_add:
                                                new_stop = { "time": "New", "type": "activity", "name": item.get('place_name'), "coordinates": [item.get('longitude'), item.get('latitude')], "description": "Added from pool.", "zoom": 16, "pitch": 50 }
                                                if 'stops' not in day_plan: day_plan['stops'] = []
                                                day_plan['stops'].append(new_stop)
                                                st.toast(f"Added '{item.get('place_name')}' to Day {day_num_to_add}.")
                                                # Reset selectbox - This requires a bit more care in Streamlit
                                                # Often, just rerunning is enough, but sometimes you need explicit reset
                                                # st.session_state[add_day_key] = "-" # Try this, may not always work perfectly
                                                st.rerun()
                                st.divider() # Separator within pool list

                with edit_col_plan:
                    st.markdown("**Scheduled Plan Editor:**")
                    if not itinerary_data:
                        st.write("No plan to edit.")
                    else:
                        # Use a container for scrolling
                         with st.container(height=400): # Adjust height
                            for day_index, day_plan in enumerate(itinerary_data):
                                day_num = day_plan.get("day", day_index + 1)
                                st.markdown(f"**Day {day_num}**")
                                stops = day_plan.get('stops', [])
                                if not stops:
                                    st.caption("_No activities scheduled._")
                                else:
                                    for stop_index, stop in enumerate(stops):
                                        stop_id = f"{stop.get('name', '')}_{stop.get('coordinates', [0,0])[0]:.4f}_{stop.get('coordinates', [0,0])[1]:.4f}"
                                        stop_key_base = f"edit_stop_{day_num}_{stop_index}_{stop_id}"
                                        # Display stop with edit buttons
                                        cols_edit_item = st.columns([0.1, 0.6, 0.1, 0.1, 0.1]) # Time, Name, Up, Down, Remove
                                        with cols_edit_item[0]: st.caption(f"{stop.get('time', '')}")
                                        with cols_edit_item[1]: st.markdown(f"{stop.get('name', 'Unknown')}")
                                        with cols_edit_item[2]: # Up
                                            if stop_index > 0:
                                                if st.button("⬆️", key=f"up_{stop_key_base}", help="Move Up"):
                                                    # Swap logic...
                                                    st.session_state.detailed_itinerary_data[day_index]['stops'].insert(stop_index - 1, st.session_state.detailed_itinerary_data[day_index]['stops'].pop(stop_index))
                                                    st.rerun()
                                        with cols_edit_item[3]: # Down
                                            if stop_index < len(stops) - 1:
                                                if st.button("⬇️", key=f"down_{stop_key_base}", help="Move Down"):
                                                    # Swap logic...
                                                    st.session_state.detailed_itinerary_data[day_index]['stops'].insert(stop_index + 1, st.session_state.detailed_itinerary_data[day_index]['stops'].pop(stop_index))
                                                    st.rerun()
                                        with cols_edit_item[4]: # Remove
                                            if st.button("🗑️", key=f"rm_{stop_key_base}", help="Remove"):
                                                removed = st.session_state.detailed_itinerary_data[day_index]['stops'].pop(stop_index)
                                                st.toast(f"Removed '{removed.get('name')}'")
                                                st.rerun()
                                        st.divider() # Separator between stops in editor
                                st.markdown("<br>", unsafe_allow_html=True) # Add space between days in editor


    except Exception as e:
        st.error(f"Error displaying interactive map/editor: {e}")
        st.json(st.session_state.get('detailed_itinerary_data', {}))

# ... (Rest of the file, messages for generating itinerary if needed) ...
elif st.session_state.get('detailed_itinerary_data') is None and num_curated_geocoded > 0:
     st.info("Click '🚀 Generate Detailed Plan' above to view and edit your itinerary.")
elif num_curated_geocoded == 0:
    st.info("Add activities (Section 3) and geocode them (Section 4) first.")

# --- End of file ---