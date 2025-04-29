# ai_travel_planner/src/app.py

import streamlit as st
import ollama
import re
import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pydeck as pdk
from tools import geocode_location  # Assuming geocode_location is in a separate file named tools.py
from itinerary_agent import create_basic_itinerary

# --- Configuration ---
OLLAMA_MODEL = 'deepseek-r1:7B'
GEOCODER_USER_AGENT = "ai_travel_planner_app_v0.2" # Increment version slightly

# --- Helper Functions ---
@st.cache_data # Cache based on the place_name argument
def cached_geocode_tool(place_name):
    """Calls the geocoding tool and caches the result."""
    print(f"DEBUG: Calling geocode_location tool for: {place_name}") # Optional debug print
    # This now calls the function imported from tools.py
    return geocode_location(place_name)


# --- REVISED: Parsing Function to extract Place Name ---
def parse_suggestions(response_text):
    """
    Parses the LLM response to extract suggestions, trying to isolate Place Names.
    Assumes LLM might format as '**Place Name:** Description' or similar.
    Returns a list of dictionaries: [{'display_text': 'Full suggestion', 'place_name': 'Extracted Name'}, ...]
    """
    suggestions = []
    lines = [line.strip() for line in response_text.strip().split('\n')]

    # Regex to find patterns like: optional list marker, optional spaces, **Place Name**, optional separator (like :), rest of description
    # It captures the text between ** and ** as group 1 (place_name)
    # Pattern 1: **Place Name:** Description
    pattern1 = re.compile(r"^[*\-\d]*\.?\s*\*\*(.*?)\*\*\s*[:\-]?\s*(.*)")
    # Pattern 2: Just **Place Name** on a line (maybe description is implicit)
    pattern2 = re.compile(r"^[*\-\d]*\.?\s*\*\*(.*?)\*\*$")
    # Pattern 3: Fallback for simple list items without bolding (less ideal)
    pattern3 = re.compile(r"^[*\-\d]+\.?\s+(.*)")

    for line in lines:
        match1 = pattern1.match(line)
        match2 = pattern2.match(line)
        match3 = pattern3.match(line)

        display_text = line # Default display text is the whole line
        place_name = None   # Default place name is None

        if match1:
            place_name = match1.group(1).strip()
            description = match1.group(2).strip()
            # Reconstruct display text for clarity if needed, or keep original line
            display_text = f"**{place_name}**: {description}" if description else f"**{place_name}**"
        elif match2:
            place_name = match2.group(1).strip()
            display_text = f"**{place_name}**"
        elif match3:
            # Fallback: Use the whole suggestion text after the marker as the place name
            place_name = match3.group(1).strip()
            display_text = place_name # Display text is the same
            # We could add a flag here indicating lower confidence in the place name if needed
        elif len(line) > 5: # Very basic fallback if no pattern matches
             place_name = line # Assume the whole line might be the name
             display_text = line

        # Only add if we reasonably think we got something
        if place_name:
            suggestions.append({"display_text": display_text, "place_name": place_name})

    # If parsing failed completely, return empty list
    # print(f"DEBUG: Parsed suggestions: {suggestions}") # DEBUG Line
    return suggestions


# --- Initialize Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'curated_list' not in st.session_state:
    # --- REVISED: Curated list now stores dictionaries ---
    st.session_state.curated_list = [] # List of {'display_text': ..., 'place_name': ...}
if 'latest_suggestions' not in st.session_state:
    st.session_state.latest_suggestions = [] # List of {'display_text': ..., 'place_name': ...}
if 'geocoded_locations' not in st.session_state:
    # --- REVISED: Keys are now the display_text ---
    # Stores: { 'display_text': {'place_name':..., 'latitude':..., 'longitude':..., 'address':...} or None }
    st.session_state.geocoded_locations = {}
if 'map_data' not in st.session_state:
    st.session_state.map_data = pd.DataFrame()

# Keys for widgets need to be defined if used before assignment in conditional blocks
if 'location' not in st.session_state: st.session_state.location = ""
if 'duration' not in st.session_state: st.session_state.duration = ""
if 'activity_prefs' not in st.session_state: st.session_state.activity_prefs = []
if 'budget_pref' not in st.session_state: st.session_state.budget_pref = "Any"

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("AI Travel Planner - Step 3: Brainstorm, Curate & Map (Refined)")

# --- Section 1: Define Your Trip ---
st.header("1. Define Your Trip")
col1, col2 = st.columns(2)
with col1:
    st.text_input("Destination:", placeholder="e.g., Lisbon, Portugal", key='location')
    st.text_input("Trip Duration (optional):", placeholder="e.g., 5 days", key='duration')
with col2:
    st.multiselect(
        "Activity Preferences:",
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

# --- REVISED: System prompt focusing on individual suggestions & formatting ---
BRAINSTORM_SYSTEM_PROMPT = f"""You are a helpful travel brainstorming assistant. Your goal is to suggest individual activities, sights, or places based on the user's request and the trip context.

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
5.  **DO NOT** include meta-commentary (like /think).
6.  Provide around 3-7 suggestions per response unless the user asks for more/less.

Start suggesting based on the user's next message."""

# Display Chat History (Optional)
# ...

# --- ADDED: Label for chat input ---
st.caption("Chat to brainstorm activities here:")
user_prompt = st.chat_input("Ask for suggestions (e.g., 'suggest some historical sites')")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Prepare messages for Ollama API (use the NEW system prompt)
    messages_for_api = [{"role": "system", "content": BRAINSTORM_SYSTEM_PROMPT}] + st.session_state.messages

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Brainstorming..."):
            try:
                response = ollama.chat(model=OLLAMA_MODEL, messages=messages_for_api)
                full_response = response['message']['content']
                message_placeholder.markdown(full_response) # Show raw response first
                # Parse the response to get structured suggestions
                st.session_state.latest_suggestions = parse_suggestions(full_response)
                # print(f"DEBUG: Latest suggestions set: {st.session_state.latest_suggestions}") # DEBUG
            except Exception as e:
                st.error(f"An error occurred while contacting Ollama: {e}")
                full_response = "Sorry, I encountered an error."
                message_placeholder.markdown(full_response)
                st.session_state.latest_suggestions = []

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Section 3: Select Activities ---
st.header("3. Select Activities")

if not st.session_state.latest_suggestions:
    st.info("Ask the AI for suggestions using the chat box above. Suggestions will appear here.")
else:
    st.markdown("Check the boxes next to activities you want to save:")

    # --- REVISED: Checkbox logic uses the suggestion dictionary ---
    curated_display_texts = {item['display_text'] for item in st.session_state.curated_list}
    newly_selected_dicts = []
    newly_deselected_dicts = []

    for suggestion_dict in st.session_state.latest_suggestions:
        display_text = suggestion_dict['display_text']
        place_name = suggestion_dict['place_name'] # Although not directly used here, it's part of the dict
        checkbox_key = f"cb_{display_text[:50]}" # Key based on display text
        is_in_curated = display_text in curated_display_texts

        is_selected = st.checkbox(display_text, value=is_in_curated, key=checkbox_key)

        if is_selected and not is_in_curated:
            newly_selected_dicts.append(suggestion_dict)
        elif not is_selected and is_in_curated:
            newly_deselected_dicts.append(suggestion_dict)

    needs_update = False
    if newly_selected_dicts:
        st.session_state.curated_list.extend(newly_selected_dicts)
        needs_update = True
    if newly_deselected_dicts:
        deselected_texts = {d['display_text'] for d in newly_deselected_dicts}
        st.session_state.curated_list = [item for item in st.session_state.curated_list if item['display_text'] not in deselected_texts]
        # Remove from geocoded_locations if deselected
        for item_dict in newly_deselected_dicts:
            if item_dict['display_text'] in st.session_state.geocoded_locations:
                del st.session_state.geocoded_locations[item_dict['display_text']]
        needs_update = True

    # If list changed, update map data if it exists
    if needs_update and not st.session_state.map_data.empty:
        current_geocoded = [st.session_state.geocoded_locations.get(item['display_text']) for item in st.session_state.curated_list if st.session_state.geocoded_locations.get(item['display_text'])]
        if current_geocoded:
             # Prepare DF: Need 'lat', 'lon'. Add 'name' for tooltips perhaps
             map_df_data = [{'lat': g['latitude'], 'lon': g['longitude'], 'name': g['place_name']} for g in current_geocoded if g]
             if map_df_data:
                 st.session_state.map_data = pd.DataFrame(map_df_data)
             else:
                  st.session_state.map_data = pd.DataFrame()
        else:
             st.session_state.map_data = pd.DataFrame()
        # st.rerun() # Optional rerun

# --- Section 4: Your Curated Activities ---
st.header("4. Your Curated Activities")

if not st.session_state.curated_list:
    st.write("No activities selected yet.")
else:
    cols_curated = st.columns([3, 1]) # Layout with 3/4 width for list, 1/4 for buttons
    with cols_curated[0]:
        st.markdown("**Your Saved Items:**")
        # Display curated items and their geocoding status
        for i, item_dict in enumerate(st.session_state.curated_list):
            display_text = item_dict['display_text']
            geo_status = "" # Default: Not attempted or status unknown

            # Check if geocoding has been attempted for this item (key exists in the dict)
            if display_text in st.session_state.geocoded_locations:
                # Check if the attempt was successful (result is not None)
                if st.session_state.geocoded_locations[display_text] is not None:
                    geo_status = "📍 Geocoded"
                else:
                    # Geocoding attempted but failed (result stored as None)
                    geo_status = "❌ Not Found"

            # Display the item with its status icon
            st.markdown(f"- {display_text} _{geo_status}_")

    with cols_curated[1]:
        st.markdown("**Actions:**")
        # Button to trigger geocoding
        if st.button("🔄 Geocode & Update Map"):
            # Reset only the geocoded locations dictionary for a fresh run, keep the curated list
            st.session_state.geocoded_locations = {}
            geocoded_data_list = [] # To collect successful results for the map DataFrame
            total_items = len(st.session_state.curated_list)

            # Show progress bar only if there are items to process
            progress_bar = None
            if total_items > 0:
                progress_bar = st.progress(0, text="Starting geocoding...")

            # Loop through the user's curated list
            for i, item_dict in enumerate(st.session_state.curated_list):
                display_text = item_dict['display_text']
                # Use the extracted 'place_name' for geocoding; fallback to display_text if 'place_name' isn't found
                place_name_to_geocode = item_dict.get('place_name', display_text)

                # Skip if the place name is somehow empty or invalid
                if not place_name_to_geocode:
                    st.warning(f"Skipping item with invalid place name: {display_text}")
                    st.session_state.geocoded_locations[display_text] = None # Mark as failed
                    if progress_bar:
                        progress_bar.progress((i + 1) / total_items, text=f"Skipping invalid item")
                    continue

                # Update progress bar text before making the call
                if progress_bar:
                    progress_bar.progress(i / total_items, text=f"Geocoding: {place_name_to_geocode}...")

                # --- CORE CHANGE: Use the cached wrapper function ---
                # This function calls tools.geocode_location internally
                geo_result = cached_geocode_tool(place_name_to_geocode)
                # --- END CORE CHANGE ---

                # Store the result (either the dict or None) in session state, keyed by the unique display_text
                if geo_result:
                    # Success! Store the dictionary containing lat, lon, address
                    # Add the place_name used for the query back into the stored dict for reference
                    result_to_store = {
                        "place_name": place_name_to_geocode,
                        "latitude": geo_result["latitude"],
                        "longitude": geo_result["longitude"],
                        "address": geo_result["address"]
                    }
                    st.session_state.geocoded_locations[display_text] = result_to_store
                    # Add the successful result to our list for DataFrame creation
                    geocoded_data_list.append(result_to_store)
                else:
                    # Failure or not found, store None
                    st.session_state.geocoded_locations[display_text] = None

                # Update progress bar after the call completes
                if progress_bar:
                    progress_bar.progress((i + 1) / total_items, text=f"Geocoded: {place_name_to_geocode}")

                # IMPORTANT: Respect Nominatim usage policy - pause briefly after each API call
                time.sleep(0.2) # 200ms pause, adjust if needed

            # Clean up the progress bar after the loop finishes
            if progress_bar:
                progress_bar.empty()

            # --- Process results after the loop ---
            if geocoded_data_list:
                 # Create DataFrame for the map ONLY from successfully geocoded items
                 # Ensure required columns ('lat', 'lon') exist and add 'name' for tooltips
                 map_df_data = []
                 for g in geocoded_data_list:
                     if g and 'latitude' in g and 'longitude' in g:
                         map_df_data.append({
                             'lat': g['latitude'],
                             'lon': g['longitude'],
                             'name': g.get('place_name', 'Unknown') # Use place_name for tooltip
                         })

                 if map_df_data:
                     st.session_state.map_data = pd.DataFrame(map_df_data)
                 else:
                     st.session_state.map_data = pd.DataFrame() # Ensure it's an empty DF if no valid data

                 st.success(f"Geocoding complete. Found coordinates for {len(st.session_state.map_data)}/{total_items} items.")
            else:
                 # No items were successfully geocoded
                 st.session_state.map_data = pd.DataFrame() # Ensure map data is empty
                 st.warning("Could not geocode any of the curated activities.")

            # Rerun the app to update the UI immediately with the new geocoding statuses and map data
            st.rerun()

        # Button to clear the curated list and related state
        if st.button("Clear Curated List"):
            st.session_state.curated_list = []
            # Optional: Clear latest suggestions if desired
            # st.session_state.latest_suggestions = []
            st.session_state.geocoded_locations = {}
            st.session_state.map_data = pd.DataFrame()
            # Clear any previously generated itinerary if it exists
            if 'generated_itinerary' in st.session_state:
                del st.session_state.generated_itinerary
            st.toast("Curated list cleared!") # Optional user feedback
            st.rerun()

# --- Section 5: Map Visualization ---
# --- Section 5: Activity Map (Pydeck) ---
st.header("5. Activity Map (Pydeck)")

# Check if we have valid map data (DataFrame with lat, lon, name)
if 'map_data' in st.session_state and not st.session_state.map_data.empty:
    map_df = st.session_state.map_data

    # --- Pydeck Configuration ---

    # Define the layer for points on the map
    layer = pdk.Layer(
        "ScatterplotLayer", # Use ScatterplotLayer for points
        map_df,
        get_position=["lon", "lat"], # Specify columns for coordinates
        get_color="[200, 30, 0, 160]", # RGBA color for points
        get_radius=150, # Radius of points in meters
        radius_min_pixels=3, # Ensure points are visible when zoomed out
        radius_max_pixels=100,
        pickable=True, # Enable hovering/clicking
        auto_highlight=True, # Highlight point on hover
    )

    # Define the initial view of the map
    # Center the view on the average coordinates of the points
    mid_lat = map_df["lat"].mean()
    mid_lon = map_df["lon"].mean()

    initial_view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=11, # Initial zoom level (adjust as needed)
        pitch=0, # Initial angle (0 for 2D view)
        bearing=0, # Initial map rotation
    )

    # Define tooltip information (appears on hover)
    tooltip = {
        "html": "<b>{name}</b>", # Display the 'name' column value in bold
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontFamily": '"Helvetica Neue", Arial, sans-serif',
            "fontSize": "12px",
            "padding": "4px 8px",
        }
    }

    # Render the map using st.pydeck_chart
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=initial_view_state,
        map_provider="mapbox", # Explicitly state map provider
        map_style=pdk.map_styles.MAPBOX_LIGHT, # Choose a map style (e.g., MAPBOX_DARK, MAPBOX_SATELLITE)
        tooltip=tooltip,
        # Ensure Mapbox key is available via env var or Streamlit secrets
    ))
    st.caption("Map showing successfully geocoded locations. Hover over points for names.")

    # --- List Failed Items (Keep this part) ---
    if 'curated_list' in st.session_state and 'geocoded_locations' in st.session_state:
        failed_items = [
            item['display_text'] for item in st.session_state.curated_list
            if item['display_text'] in st.session_state.geocoded_locations and st.session_state.geocoded_locations[item['display_text']] is None
        ]
        if failed_items:
            with st.expander("⚠️ Some locations could not be geocoded:"):
                for item_text in failed_items:
                    st.markdown(f"- {item_text}")
else:
    # Message when no map data is available
    st.info("Select activities and click '🔄 Geocode & Update Map' in Section 4 to see the map.")
    

# --- Section 6: Generate & View Itinerary ---
st.header("6. Generate & View Itinerary")

# Input for number of days (can default based on user input or curated list size)
# Provide a sensible default, e.g., number of curated items or 1
default_days = 1
if st.session_state.curated_list:
    # Attempt to parse duration, fallback to list size or 1
    try:
        # Basic parsing of strings like "5 days", "3", etc.
        duration_str = str(st.session_state.get('duration', '')).lower().replace("days","").strip()
        default_days = int(duration_str) if duration_str.isdigit() else len(st.session_state.curated_list)
        default_days = max(1, default_days) # Ensure at least 1 day
    except ValueError:
        default_days = max(1, len(st.session_state.curated_list))
else:
     default_days = 1 # Fallback if no list exists yet


num_days_input = st.number_input(
    "Number of Days for Itinerary:",
    min_value=1,
    value=default_days,
    key='num_days_input',
    help="How many days should the itinerary cover? Clustering will group activities."
)

# Button to trigger itinerary generation
if st.button("✨ Generate Basic Itinerary"):
    # 1. Get successfully geocoded activities
    geocoded_activities_list = []
    if 'geocoded_locations' in st.session_state and st.session_state.curated_list:
        for activity_dict in st.session_state.curated_list:
            display_text = activity_dict['display_text']
            geo_info = st.session_state.geocoded_locations.get(display_text)
            # Only include if geocoding was successful (geo_info is not None)
            if geo_info:
                # Ensure the dictionary passed to the agent has lat/lon
                # It might already be the case if geo_info comes directly from geocoded_locations storage
                activity_data_for_agent = {
                    **activity_dict, # Include original info like display_text, place_name
                    'latitude': geo_info['latitude'],
                    'longitude': geo_info['longitude'],
                    'address': geo_info.get('address', '') # Include address if available
                }
                geocoded_activities_list.append(activity_data_for_agent)

    # 2. Check if we have enough data
    if not geocoded_activities_list:
        st.warning("Please select and geocode some activities first before generating an itinerary.")
    elif len(geocoded_activities_list) < num_days_input:
         st.warning(f"Warning: You only have {len(geocoded_activities_list)} geocoded activities, which is less than the requested {num_days_input} days. The itinerary might group items unexpectedly or have empty days.")
         # Proceed anyway, agent handles this internally now.
         with st.spinner("Generating itinerary..."):
             generated_itinerary = create_basic_itinerary(geocoded_activities_list, num_days_input)
             st.session_state.generated_itinerary = generated_itinerary # Store in session
    else:
        # 3. Call the agent function
        with st.spinner("Generating itinerary..."):
            generated_itinerary = create_basic_itinerary(geocoded_activities_list, num_days_input)
            st.session_state.generated_itinerary = generated_itinerary # Store in session

    # Force UI update after generation attempt (even if None)
    st.rerun()


# --- Display the generated itinerary (if it exists) ---
if 'generated_itinerary' in st.session_state and st.session_state.generated_itinerary:
    st.subheader("Your Generated Daily Plan:")
    itinerary_data = st.session_state.generated_itinerary
    sorted_days = sorted(itinerary_data.keys()) # Sort days numerically

    # Use columns for better layout if many days, or expanders
    num_columns = min(len(sorted_days), 3) # Max 3 columns layout
    if num_columns > 0:
         cols = st.columns(num_columns)
         col_index = 0
         for day in sorted_days:
             with cols[col_index % num_columns]:
                 st.markdown(f"**Day {day}**")
                 activities_today = itinerary_data[day]
                 if activities_today:
                     for activity in activities_today:
                         # Display the place name or display text
                         display_name = activity.get('place_name', activity.get('display_text', 'Unknown Activity'))
                         st.markdown(f"- {display_name}")
                 else:
                     st.markdown("_No activities assigned._")
                 st.markdown("---") # Separator between days in the same column
             col_index += 1
    else:
         st.write("Itinerary generated, but contains no days or activities.")


elif 'generated_itinerary' in st.session_state and st.session_state.generated_itinerary is None:
    # Explicitly handle the case where generation failed and returned None
    st.error("Could not generate itinerary. Check if activities are geocoded or if there are enough activities for the requested days.")