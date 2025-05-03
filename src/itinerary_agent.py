# src/itinerary_agent.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai # Import Gemini
import os
import json
import re

# --- Existing create_basic_itinerary function (keep as is) ---
def create_basic_itinerary(activities_with_coords: list[dict], num_days: int) -> dict | None:
    """
    Groups geocoded activities into days based on geographic proximity using K-Means.
    (Keep the existing code for this function)
    """
    # --- Input Validation ---
    if not activities_with_coords:
        print("Itinerary Agent (Basic): No geocoded activities provided.")
        return None
    if num_days <= 0:
        print(f"Itinerary Agent (Basic): Invalid number of days ({num_days}).")
        return None

    actual_activities = [a for a in activities_with_coords if isinstance(a.get('latitude'), (int, float)) and isinstance(a.get('longitude'), (int, float))]

    if not actual_activities:
         print("Itinerary Agent (Basic): No activities with valid coordinates provided.")
         return None

    if len(actual_activities) < num_days:
        print(f"Itinerary Agent (Basic): Warning - Fewer activities ({len(actual_activities)}) than days ({num_days}). Adjusting days for clustering.")
        num_days = len(actual_activities)
        if num_days == 0: return None # Should not happen if actual_activities is not empty

    # --- Data Preparation ---
    try:
        df = pd.DataFrame(actual_activities)
        # Ensure coordinates are numeric (already checked, but belt-and-suspenders)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df.dropna(subset=['latitude', 'longitude'], inplace=True)

        if df.empty:
            print("Itinerary Agent (Basic): No valid coordinates found after DataFrame conversion.")
            return None
        # This check might be redundant now due to adjustment above, but safe
        if len(df) < num_days:
            print(f"Itinerary Agent (Basic): Warning - Only {len(df)} activities with valid coordinates remain after cleaning, less than {num_days} days. Adjusting.")
            num_days = len(df)
            if num_days == 0 : return None

    except Exception as e:
        print(f"Itinerary Agent (Basic): Error preparing DataFrame - {e}")
        return None

    # --- Clustering ---
    coords = df[['latitude', 'longitude']].values
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)

    kmeans = KMeans(n_clusters=num_days, random_state=42, n_init=10)
    try:
        df['day_cluster'] = kmeans.fit_predict(scaled_coords)
    except Exception as e:
        print(f"Itinerary Agent (Basic): Error during K-Means clustering - {e}")
        return None

    # --- Itinerary Creation ---
    itinerary = {day + 1: [] for day in range(num_days)}

    # Create a mapping from coords back to original dicts to handle duplicates
    coord_map = {}
    for item in actual_activities:
        coord_tuple = (item.get('latitude'), item.get('longitude'))
        if coord_tuple not in coord_map:
            coord_map[coord_tuple] = []
        coord_map[coord_tuple].append(item)

    processed_indices = set() # Keep track of indices we've assigned from df

    for day_num_zero_based in range(num_days):
        day_num = day_num_zero_based + 1
        # Get indices of activities in this cluster
        cluster_indices = df[df['day_cluster'] == day_num_zero_based].index

        for idx in cluster_indices:
            if idx in processed_indices:
                continue # Skip if this specific row index already processed

            lat = df.loc[idx, 'latitude']
            lon = df.loc[idx, 'longitude']
            coord_tuple = (lat, lon)

            if coord_tuple in coord_map and coord_map[coord_tuple]:
                # Take one matching original dict and remove it from the map list
                original_dict = coord_map[coord_tuple].pop(0)
                itinerary[day_num].append(original_dict)
                processed_indices.add(idx) # Mark this df row as processed
            else:
                print(f"Itinerary Agent (Basic): Warning - Could not find matching original dict for row index {idx}: {lat}, {lon}")

    print(f"Itinerary Agent (Basic): Successfully created basic itinerary for {num_days} days.")
    return itinerary


# --- NEW: Detailed Itinerary Generation with Gemini ---
def generate_detailed_itinerary_gemini(
    activities: list[dict],
    num_days: int,
    destination: str,
    prefs: list[str],
    budget: str
) -> list[dict] | None:
    """
    Uses Gemini to generate a detailed, timed itinerary JSON based on curated activities.

    Args:
        activities: List of curated activity dicts (must include 'place_name', 'latitude', 'longitude').
        num_days: The number of days for the itinerary.
        destination: The trip destination.
        prefs: List of user activity preferences.
        budget: User budget preference.

    Returns:
        A list of dictionaries representing the itinerary structure needed for the JS,
        or None if generation fails.
        Example structure:
        [
            {
                "day": 1, "title": "Day 1: Exploration",
                "stops": [
                    {"time": "09:30", "type": "sightseeing", "name": "Place A", "coordinates": [lon, lat], "description": "...", "zoom": 16, "pitch": 50},
                    {"time": "12:00", "type": "lunch", "name": "Restaurant B", "coordinates": [lon, lat], "description": "...", "zoom": 15, "pitch": 45},
                    ...
                ]
            },
            ...
        ]
    """
    if not activities:
        print("Itinerary Agent (Detailed): No activities provided for detailed generation.")
        return None

    print(f"Itinerary Agent (Detailed): Starting generation for {num_days} days in {destination}.")

    # --- Prepare Input for Gemini ---
    activity_list_str = ""
    for i, act in enumerate(activities):
        name = act.get('place_name', act.get('display_text', f'Activity {i+1}'))
        lat = act.get('latitude')
        lon = act.get('longitude')
        activity_list_str += f"- {name} (Coords: {lat:.4f}, {lon:.4f})\n"

    prompt = f"""
    You are an expert travel planner creating a detailed, interactive itinerary.

    **Trip Context:**
    *   **Destination:** {destination or 'Not specified'}
    *   **Duration:** {num_days} days
    *   **Core Activities Provided by User:**
    {activity_list_str}
    *   **User Preferences:** {', '.join(prefs) or 'None specified'}
    *   **Budget Style:** {budget or 'Not specified'}

    **Your Task:**
    1.  Create a logical and enjoyable itinerary spanning exactly **{num_days} days**.
    2.  **Prioritize including ALL the core activities** provided by the user. Distribute them sensibly across the days based on location and type.
    3.  **Add realistic timings** for each activity (e.g., "09:00", "11:30", "14:00", "17:30"). Assume reasonable travel time between nearby locations, but don't explicitly state travel time.
    4.  **Suggest appropriate activity types** for each stop. Use simple categories like: "sightseeing", "museum", "park", "lunch", "dinner", "break", "shopping", "activity", "viewpoint". If it's one of the user's core activities, try to match its likely type. For added meals/breaks, use "lunch", "dinner", or "break".
    5.  **Write a brief, engaging, single-sentence description** for each stop, highlighting what to see or do there.
    6.  **Include map parameters:** For each stop, suggest a reasonable `zoom` (usually 15-17) and `pitch` (usually 40-60) for viewing it on a 3D map.
    7.  **Structure the output ONLY as a JSON list of day objects.** Adhere strictly to the following format:

    ```json
    [
      {{
        "day": 1,
        "title": "Day 1: [Your Creative Day Title]",
        "stops": [
          {{
            "time": "HH:MM",
            "type": "activity_type", // e.g., "sightseeing", "lunch"
            "name": "Exact Place Name from Input OR Your Suggestion",
            "coordinates": [longitude, latitude], // Use coordinates from the input list
            "description": "One-sentence engaging description.",
            "zoom": 16, // Number between 14-18
            "pitch": 50 // Number between 30-70
          }},
          // ... more stops for Day 1
        ]
      }},
      // ... more day objects for Day 2, Day 3, etc. up to num_days
    ]
    ```

    **Important Rules:**
    *   The final output MUST be **only the JSON data** structure specified above. No introductory text, explanations, apologies, or concluding remarks.
    *   Ensure all coordinates provided in the input activities list are used correctly in the output JSON (`[longitude, latitude]` format).
    *   Be creative but realistic with timings and flow.
    *   Generate exactly {num_days} day objects in the list.
    """

    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            print("🔴 Error: GOOGLE_API_KEY environment variable not found.")
            return None
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest', # Or another capable Gemini model
            generation_config={"response_mime_type": "application/json"} # Request JSON output
        )
        print("Itinerary Agent (Detailed): Sending request to Gemini...")
        response = model.generate_content(prompt)

        # --- Process Response ---
        if response.parts:
            raw_json = response.text
            # print("DEBUG: Raw Gemini Response:\n", raw_json) # Optional debug

            # Validate and parse the JSON
            try:
                # Sometimes the model might wrap the JSON in ```json ... ```
                cleaned_json = re.sub(r'^```json\s*|\s*```$', '', raw_json.strip(), flags=re.DOTALL)
                itinerary_data = json.loads(cleaned_json)

                # Basic validation of the structure
                if isinstance(itinerary_data, list) and \
                   all(isinstance(day, dict) and 'day' in day and 'title' in day and 'stops' in day for day in itinerary_data) and \
                   len(itinerary_data) == num_days:
                     print(f"Itinerary Agent (Detailed): Successfully generated and parsed itinerary for {len(itinerary_data)} days.")
                     # Add further validation per stop if needed
                     return itinerary_data
                else:
                    print("Itinerary Agent (Detailed): Error - Gemini output did not match the expected JSON structure or number of days.")
                    print("--- Faulty JSON Received ---")
                    print(cleaned_json)
                    print("--- End Faulty JSON ---")
                    return None

            except json.JSONDecodeError as json_err:
                print(f"Itinerary Agent (Detailed): Error - Failed to decode JSON response from Gemini: {json_err}")
                print("--- Raw Response Received ---")
                print(raw_json)
                print("--- End Raw Response ---")
                return None
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             block_reason = response.prompt_feedback.block_reason
             print(f"Itinerary Agent (Detailed): ⚠️ Request blocked by safety filter: {block_reason}")
             return None
        else:
            print("Itinerary Agent (Detailed): Error - Gemini returned an empty response.")
            return None

    except Exception as e:
        print(f"Itinerary Agent (Detailed): 🔴 An error occurred while contacting the Gemini API: {e}")
        return None
    
        return None

def brainstorm_places_for_quick_mode(location: str, duration: str, user_prompt: str) -> list[str] | None:
    """
    Uses Gemini to suggest a list of relevant place names based on user input for Quick Mode.
    Args:
        location: The destination city/area.
        duration: The trip duration (e.g., "3 days").
        user_prompt: The user's free-text description of preferences.
    Returns:
        A list of suggested place names, or None if generation fails.
    """
    print(f"Itinerary Agent (Quick Brainstorm): For {location}, {duration}, prompt: '{user_prompt[:50]}...'")
    # Estimate number of places needed (e.g., 5-7 per day, adjust as needed)
    days = 1
    try:
        match = re.search(r'\d+', duration)
        if match: days = int(match.group())
        days = max(1, min(days, 10)) # Clamp days (e.g., 1-10)
    except:
        days = 3 # Default if duration parsing fails
    num_places_to_suggest = days * 6 # Aim for ~6 places per day

    prompt = f"""
    You are a travel assistant helping generate ideas for a trip.
    Based on the user's request, suggest a list of specific, well-known place names (landmarks, museums, neighborhoods, parks, significant restaurants/markets if mentioned) relevant to their interests in the specified location.

    **Trip Details:**
    *   **Location:** {location}
    *   **Duration:** {duration}
    *   **User Interests/Request:** {user_prompt}

    **Your Task:**
    1.  Identify key themes and preferences from the user's request.
    2.  Suggest around **{num_places_to_suggest} distinct place names** in {location} that match these interests. Prioritize popular and relevant locations.
    3.  **Output ONLY a simple numbered list of the place names.** Do not include descriptions, markdown formatting (like bolding), categories, or any introductory/concluding text. Just the names.

    **Example Output:**
    1. Eiffel Tower
    2. Louvre Museum
    3. Montmartre
    4. Sacré-Cœur Basilica
    5. Seine River Cruise
    6. Musée d'Orsay
    """
    try:
        # Assumes genai is configured in the calling script (Quick Mode page)
        # If not, configure it here using GOOGLE_API_KEY from os.getenv
        # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        # if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not found")
        # genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel('gemini-1.5-flash-latest') # Consider making model name configurable
        print("Itinerary Agent (Quick Brainstorm): Sending request to Gemini...")
        response = model.generate_content(prompt)

        if response.parts:
            raw_text = response.text
            # print("DEBUG: Raw Quick Brainstorm Response:\n", raw_text) # Optional
            # Parse the numbered list
            place_names = []
            lines = raw_text.strip().split('\n')
            for line in lines:
                # Try to match lines starting with number, dot, optional space
                match = re.match(r"^\d+\.?\s*(.*)", line.strip())
                if match:
                    place = match.group(1).strip()
                    if place: # Avoid empty strings
                        place_names.append(place)

            if place_names:
                print(f"Itinerary Agent (Quick Brainstorm): Extracted {len(place_names)} place names.")
                return place_names
            else:
                print("Itinerary Agent (Quick Brainstorm): Failed to parse place names from response.")
                print("--- Raw Response ---")
                print(raw_text)
                print("--- End Raw Response ---")
                return None
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             print(f"Itinerary Agent (Quick Brainstorm): ⚠️ Request blocked: {response.prompt_feedback.block_reason.name}")
             return None
        else:
            print("Itinerary Agent (Quick Brainstorm): Error - Gemini returned an empty response.")
            return None

    except Exception as e:
        print(f"Itinerary Agent (Quick Brainstorm): 🔴 Error contacting Gemini: {e}")
        return None
    

# --- NEW: Function to Modify an Existing Itinerary via Chat ---
def modify_detailed_itinerary_gemini(
    current_itinerary_json: str, # Pass the current itinerary as a JSON string
    user_request: str,
    destination: str, # Keep original context
    prefs: list[str], # Keep original context
    budget: str      # Keep original context
) -> tuple[str | None, str | None]:
    """
    Uses Gemini to modify an existing detailed itinerary based on user chat request.

    Args:
        current_itinerary_json: The current itinerary data as a JSON string.
        user_request: The user's latest chat message requesting a change.
        destination: Original trip destination (for context).
        prefs: Original user preferences (for context).
        budget: Original budget style (for context).

    Returns:
        A tuple: (new_itinerary_json_str, error_message).
        - If successful, new_itinerary_json_str contains the updated JSON, error_message is None.
        - If Gemini explains why it can't modify or fails, new_itinerary_json_str is None,
          and error_message contains the explanation or error details.
    """
    print(f"Itinerary Agent (Modify): Requesting change: '{user_request[:50]}...'")

    prompt = f"""
You are an expert travel planner refining an existing itinerary based on user feedback. You are given an existing travel itinerary in JSON format and a user request to modify it.

**Original Trip Context:**
*   **Destination:** {destination or 'Not specified'}
*   **User Preferences:** {', '.join(prefs) or 'None specified'}
*   **Budget Style:** {budget or 'Not specified'}

**Current Itinerary (JSON Format):**
```json
{current_itinerary_json}

**User's Modification Request:**
"{user_request}"

**Your Task:**
Your task is to modify the existing itinerary based ONLY on the user's request and return the complete, updated itinerary as a single, valid JSON object.
1. Analyze the user's request in the context of the current itinerary.
2. If the request is feasible and clear, modify the entire itinerary JSON provided above to incorporate the change.
3. Maintain the exact same JSON structure and format for the output, including all required fields for each stop (day, title, stops list with time, type, name, coordinates, description, zoom, pitch). Ensure coordinates remain valid [longitude, latitude] lists.
4. Adjust Timings: After modifying the stops within a day (reordering, adding, removing), review and adjust the time fields for all stops in that day to ensure a logical, sequential flow throughout the day. Estimate reasonable durations and implicit travel times. Ensure times are in "HH:MM" format.
5. Update Day Titles: After modifying the stops for a day, review the day's title. If the main theme or focus of the day has significantly changed due to the modifications (e.g., swapping a beach day for a museum day), update the title field (e.g., "Day X: [New Theme]") to accurately reflect, the second"day": 2`, etc.
6. Maintain Structure & Fields: Preserve the exact JSON structure (list of day objects, each with day, title, stops list). Ensure all required fields (time, type, name, coordinates, description, zoom, pitch) are present and valid for every stop in the updated plan. Ensure coordinates remain valid [longitude, latitude] lists.
7. Handle New Places: If adding new places, try to make reasonable assumptions for coordinates or use placeholders like [0, 0] if coordinates cannot be determined, but clearly state this limitation in an INFO message if necessary (and don't output JSON in that case, as instructed below). Prioritize modifying existing stops.
8. Output JSON Only (on success): If the request is fulfilled, output ONLY the complete, updated, and re-sequenced JSON data structure representing the full modified itinerary list. Do not include any introductory text, explanations, apologies, or concluding remarks outside the JSON structure itself.
9. Output Explanation Only (on failure/impossibility): If the request is unclear, impossible (e.g., "add a day trip to the moon"), requires coordinates you cannot determine reliably, or fundamentally breaks the itinerary logic, DO NOT output JSON. Instead, provide a short, polite explanation of why you cannot fulfill the request. Start your explanation with "INFO:".

**IMPORTANT OUTPUT REQUIREMENTS**:
1. Return ONLY the JSON: Your entire response MUST be the updated itinerary in JSON format. Do not include any introductory text, explanations, apologies, or markdown formatting like json wrappers outside the JSON object itself.
2. Maintain Structure: Adhere strictly to the original JSON structure (list of day objects, each with 'day', 'title', 'stops'; each stop with 'name', 'time', 'type', 'description', 'coordinates', etc.).
3. VALID COORDINATES ARE ESSENTIAL:
    - Every stop in the 'stops' list MUST include a 'coordinates' field.
    - The 'coordinates' field MUST be a list containing exactly two numerical values: [longitude, latitude].
    - Correct Example: "coordinates": [-9.1393, 38.7223]
    - Incorrect Examples: "coordinates": null, "coordinates": "missing", "coordinates": {{ "lon": -9.1, "lat": 38.7 }}, "coordinates": [-9.1393]
4. Handle New Locations: If the user request requires adding a new location not present in the original itinerary, you MUST determine its correct geographical coordinates and include them in the valid [longitude, latitude] format. If you cannot determine coordinates, explain this difficulty INSTEAD of returning invalid JSON (though preferably, try your best to find them).
5. Complete Itinerary: Ensure the returned JSON represents the entire modified trip plan, not just the changed parts

**Example Scenario 1:**
User Request: "Can we switch the museum visit on Day 1 to the afternoon and have lunch earlier?"
Your Output: (Should be the full JSON itinerary list with Day 1 stops reordered and times adjusted)

**Example Scenario 2:**
User Request: "Remove Day 2 entirely."
Your Output: (Should be the full JSON itinerary list, containing only Day 1, Day 3, etc., with day numbers potentially re-sequenced if needed, or keep original day numbers if simpler).

**Example Scenario 3 (Time/Title Change):**
Current Day 1: {{"day": 1, "title": "Day 1: Coastal Views", "stops": [{{"time": "10:00", "name": "Beach Visit", ...}}, {{"time": "13:00", "name": "Lunch", ...}}, {{"time": "15:00", "name": "Cliff Walk", ...}}]}}
User Request: "Replace the beach visit on day 1 with the Art Museum visit."
Your Output: (Should be full JSON, with Day 1 like: {{"day": 1, "title": "Day 1: Art & Coast", "stops": [{{"time": "10:30", "name": "Art Museum", ...}}, {{"time": "13:30", "name": "Lunch", ...}}, {{"time": "15:30", "name": "Cliff Walk", ...}}]}} - Note adjusted times and potentially title).

**Example Scenario 4 (Day Swap):**
User Request: "Swap Day 1 and Day 2"
Your Output: (Should be full JSON, where the object with `"day": 1` now contains the stops originally from Day 2, and the object with `"day": 2` contains the stops originally from Day 1. Titles should also be reviewed/updated for the new content of Day 1 and Day 2).

Produce the output now based on the user's request.
"""
    try:
        # Ensure Google API Key is configured (it should be by the calling page)
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            print("🔴 Error (Modify Agent): GOOGLE_API_KEY not found.")
            return None, "Error: Google API Key not configured."
        # It's usually configured already, but double-checking doesn't hurt if running standalone
        # genai.configure(api_key=GOOGLE_API_KEY) # Usually already done

        model = genai.GenerativeModel(
            'gemini-1.5-flash-latest', # Or your preferred model
            generation_config={"response_mime_type": "application/json"} # CRITICAL: Request JSON output
            # Note: If Gemini gives an explanation (starts INFO:), it won't be JSON.
        )
        print("Itinerary Agent (Modify): Sending request to Gemini...")
        # Safety settings might be needed depending on the user requests
        # safety_settings={'HARASSMENT':'BLOCK_NONE', ...}
        response = model.generate_content(prompt) # Add safety_settings=safety_settings if needed

        # --- Process Response ---
        if response.parts:
            raw_text = response.text.strip()
            # print("DEBUG: Raw Gemini Modify Response:\n", raw_text) # Optional debug

            # Check if Gemini provided an explanation instead of JSON
            if raw_text.startswith("INFO:"):
                print("Itinerary Agent (Modify): Gemini provided info/explanation.")
                # Return the explanation as the error message
                return None, raw_text

            # Attempt to parse the response as JSON
            try:
                # Clean potential markdown fences just in case
                cleaned_json_text = re.sub(r'^```json\s*|\s*```$', '', raw_text, flags=re.DOTALL)

                # Validate JSON structure (basic check)
                parsed_itinerary = json.loads(cleaned_json_text)

                # More thorough validation
                if isinstance(parsed_itinerary, list) and \
                all(isinstance(day, dict) and 'day' in day and 'title' in day and 'stops' in day and isinstance(day['stops'], list) for day in parsed_itinerary):
                    # Even more detail: check stops format
                    valid_stops = True
                    for day in parsed_itinerary:
                        for stop in day['stops']:
                            if not (isinstance(stop, dict) and
                                    'time' in stop and
                                    'type' in stop and
                                    'name' in stop and
                                    'coordinates' in stop and isinstance(stop['coordinates'], list) and len(stop['coordinates']) == 2 and
                                    'description' in stop and
                                    'zoom' in stop and
                                    'pitch' in stop):
                                valid_stops = False
                                print(f"Itinerary Agent (Modify): Invalid stop structure found: {stop}")
                                break
                        if not valid_stops: break

                    if valid_stops:
                        print("Itinerary Agent (Modify): Successfully received and parsed valid modified itinerary JSON.")
                        return cleaned_json_text, None # Return the valid JSON string
                    else:
                        print("Itinerary Agent (Modify): Error - Gemini output JSON structure is invalid (stop detail issue).")
                        return None, f"Error: AI response was JSON but had an invalid stop structure.\n```json\n{cleaned_json_text}\n```"
                else:
                    print("Itinerary Agent (Modify): Error - Gemini output JSON structure is invalid (day/list issue).")
                    return None, f"Error: AI response was JSON but had an invalid overall structure.\n```json\n{cleaned_json_text}\n```"

            except json.JSONDecodeError as json_err:
                print(f"Itinerary Agent (Modify): Error - Failed to decode JSON response: {json_err}")
                # Return the raw text as an error/explanation if JSON parsing fails
                # It might contain useful info from the AI even if not perfect JSON
                error_detail = f"Error: AI response was not valid JSON.\nDetails: {json_err}\nResponse:\n{raw_text}"
                return None, error_detail

        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason.name # Use .name for the string representation
            print(f"Itinerary Agent (Modify): ⚠️ Request blocked by safety filter: {block_reason}")
            return None, f"Error: Your request was blocked by the safety filter ({block_reason}). Please rephrase your request."
        else:
            # Handle cases like stop reasons other than block, or unexpected empty response
            print(f"Itinerary Agent (Modify): Error - Gemini response issue. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
            return None, "Error: AI returned an unexpected or empty response."

    except Exception as e:
        print(f"Itinerary Agent (Modify): 🔴 An unexpected error occurred: {e}")
        # You might want to log the full traceback here for debugging
        # import traceback
        # traceback.print_exc()
        return None, f"Error: An unexpected error occurred while contacting the AI: {e}"
    
# --- Keep other functions (create_basic_itinerary, generate_detailed_itinerary_gemini) ---