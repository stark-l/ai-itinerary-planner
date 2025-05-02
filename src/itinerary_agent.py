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
# --- Keep other functions (create_basic_itinerary, generate_detailed_itinerary_gemini) ---