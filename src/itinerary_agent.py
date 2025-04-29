# src/itinerary_agent.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Added for better clustering

def create_basic_itinerary(activities_with_coords: list[dict], num_days: int) -> dict | None:
    """
    Groups geocoded activities into days based on geographic proximity using K-Means.

    Args:
        activities_with_coords: A list of dictionaries. Each dict MUST have at least
                                'latitude' and 'longitude' keys with numeric values.
                                It should ideally contain 'place_name' or 'display_text' too.
                                Example: [{'place_name': 'Eiffel Tower', 'latitude': 48.85, 'longitude': 2.29, ...}, ...]
        num_days: The desired number of days for the itinerary (determines the number of clusters).

    Returns:
        A dictionary where keys are day numbers (1 to num_days) and values are lists
        of the input activity dictionaries assigned to that day. Example:
        {
            1: [{'place_name': 'Louvre', ...}, {'place_name': 'Tuileries Garden', ...}],
            2: [{'place_name': 'Eiffel Tower', ...}, ...]
        }
        Returns None if input is insufficient (e.g., fewer geocoded activities than days,
        or num_days is zero or negative).
    """
    # --- Input Validation ---
    if not activities_with_coords:
        print("Itinerary Agent: No geocoded activities provided.")
        return None
    if num_days <= 0:
        print(f"Itinerary Agent: Invalid number of days ({num_days}).")
        return None
    if len(activities_with_coords) < num_days:
        print(f"Itinerary Agent: Warning - Fewer activities ({len(activities_with_coords)}) than days ({num_days}). Clustering might be suboptimal.")
        # Decide handling: proceed anyway, or return None? Let's proceed but it might merge clusters.
        # Alternatively, could reduce num_days: num_days = len(activities_with_coords)
        pass # Allow clustering even if fewer points than clusters

    # --- Data Preparation ---
    try:
        df = pd.DataFrame(activities_with_coords)
        # Ensure coordinates are numeric and handle potential errors/missing values
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        # Drop rows where coordinates couldn't be converted
        df.dropna(subset=['latitude', 'longitude'], inplace=True)

        if df.empty:
             print("Itinerary Agent: No valid coordinates found after cleaning.")
             return None
        if len(df) < num_days:
             print(f"Itinerary Agent: Warning - Only {len(df)} activities with valid coordinates remain, less than {num_days} days.")
             # Adjust num_days to the actual number of points if fewer than requested days
             num_days = len(df)
             if num_days == 0 : return None # Should not happen due to df.empty check, but safe


    except Exception as e:
        print(f"Itinerary Agent: Error preparing DataFrame - {e}")
        return None

    # --- Clustering ---
    # Select coordinate columns for clustering
    coords = df[['latitude', 'longitude']].values

    # --- ADDED: Scaling coordinates ---
    # K-Means is sensitive to feature scales. Scaling helps.
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    # --- END ADDED ---


    # Perform K-Means clustering (using scaled coordinates)
    # n_init='auto' or 10 to suppress future warnings. random_state for reproducibility.
    kmeans = KMeans(n_clusters=num_days, random_state=42, n_init=10)
    try:
        df['day_cluster'] = kmeans.fit_predict(scaled_coords) # Use scaled_coords
    except Exception as e:
        print(f"Itinerary Agent: Error during K-Means clustering - {e}")
        return None

    # --- Itinerary Creation ---
    # Initialize the itinerary dictionary with keys from 1 to num_days
    itinerary = {day + 1: [] for day in range(num_days)}

    # Assign activities to days based on cluster assignment
    for _, activity_row in df.iterrows():
        # Convert the pandas row back to a dictionary to store in the itinerary
        # Make sure to use the original dictionary data if possible, or construct from row
        # activity_dict = activity_row.to_dict() # This includes the 'day_cluster' column

        # Find the original dictionary corresponding to this row (safer if columns were dropped/added)
        # This assumes 'latitude' and 'longitude' are unique enough identifiers in this context
        original_dict = next((item for item in activities_with_coords
                            if item.get('latitude') == activity_row['latitude'] and
                               item.get('longitude') == activity_row['longitude']), None)

        if original_dict:
             assigned_day = activity_row['day_cluster'] + 1 # Cluster index (0 to k-1) to Day number (1 to k)
             itinerary[assigned_day].append(original_dict)
        else:
             print(f"Itinerary Agent: Warning - Could not find original dict for row: {activity_row['latitude']}, {activity_row['longitude']}")


    print(f"Itinerary Agent: Successfully created itinerary for {num_days} days.")
    return itinerary