# ai_travel_planner/src/tools.py

import requests
import json
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# --- Configuration ---
GEOCODER_USER_AGENT = "ai_travel_planner_app_v0.3_tools" # Unique user agent
OSRM_ROUTE_URL = "http://router.project-osrm.org/route/v1/driving/" # Public demo server
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Tool 1: Geocoding (Refined version of the function from app.py)
# Note: We might not need @st.cache_data here if the agent manages caching,
# but keeping it for potential direct use or testing doesn't hurt for now.
# from streamlit import cache_data # If you want Streamlit caching here
# @cache_data
def geocode_location(place_name: str, attempt=1, max_attempts=3) -> dict | None:
    """
    Geocodes a place name using Nominatim.

    Args:
        place_name: The string name of the place to geocode.
        attempt: Current retry attempt number.
        max_attempts: Maximum number of retry attempts.

    Returns:
        A dictionary {'latitude': float, 'longitude': float, 'address': str} if successful,
        None otherwise.
    """
    # print(f"[Tool Log] Geocoding: '{place_name}' (Attempt {attempt})") # Optional logging
    try:
        geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT)
        location = geolocator.geocode(place_name, timeout=10)
        if location:
            return {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "address": location.address
            }
        else:
            # print(f"[Tool Log] Geocoding: Place '{place_name}' not found.")
            return None
    except GeocoderTimedOut:
        # print(f"[Tool Log] Geocoding timed out for: '{place_name}'. Retrying...")
        if attempt < max_attempts:
            time.sleep(1) # Wait before retry
            return geocode_location(place_name, attempt + 1, max_attempts)
        else:
            # print(f"[Tool Log] Geocoding failed for '{place_name}' after {max_attempts} attempts (Timeout).")
            return None
    except (GeocoderServiceError, Exception) as e:
        # print(f"[Tool Log] Geocoding error for '{place_name}': {e}")
        return None

# Tool 2: Routing
def get_route(start_coords: tuple[float, float], end_coords: tuple[float, float]) -> dict | None:
    """
    Gets route information between two points using OSRM.

    Args:
        start_coords: Tuple of (latitude, longitude) for the start point.
        end_coords: Tuple of (latitude, longitude) for the end point.

    Returns:
        A dictionary {'distance_meters': float, 'duration_seconds': float, 'geometry': list[list[float]]}
        containing route distance, duration, and geometry (list of [lon, lat] pairs),
        or None if the route could not be found or an error occurred.
        Returns simplified geometry (polyline). For full resolution, adjust overview=full.
    """
    # OSRM expects coordinates as {longitude},{latitude} string pairs
    start_lon, start_lat = start_coords[1], start_coords[0]
    end_lon, end_lat = end_coords[1], end_coords[0]
    coords_param = f"{start_lon},{start_lat};{end_lon},{end_lat}"

    # Construct the OSRM API request URL
    # 'overview=simplified' gives a less detailed polyline (good enough for visualization)
    # 'geometries=geojson' returns geometry in standard GeoJSON format
    url = f"{OSRM_ROUTE_URL}{coords_param}?overview=simplified&geometries=geojson"
    # print(f"[Tool Log] Requesting route: {url}") # Optional logging

    try:
        response = requests.get(url, timeout=15) # Increased timeout for routing
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get('code') == 'Ok' and data.get('routes'):
            route = data['routes'][0] # Get the first route
            geometry_coords = route['geometry']['coordinates'] # List of [lon, lat] pairs
            return {
                "distance_meters": route.get('distance'),
                "duration_seconds": route.get('duration'),
                "geometry": geometry_coords # GeoJSON linestring coordinates
            }
        else:
            # print(f"[Tool Log] OSRM could not find a route. Response: {data.get('code')}")
            return None
    except requests.exceptions.RequestException as e:
        # print(f"[Tool Log] OSRM API request failed: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        # print(f"[Tool Log] Failed to parse OSRM response or missing key: {e}")
        return None

# Tool 3: Point of Interest (POI) Search
def find_nearby_pois(coords: tuple[float, float], category: str, radius_meters: int = 1000) -> list[dict] | None:
    """
    Finds points of interest (POIs) near given coordinates using Overpass API.

    Args:
        coords: Tuple of (latitude, longitude) for the center point.
        category: The type of POI to search for (e.g., "restaurant", "museum", "cafe", "atm").
                  Should correspond to common OpenStreetMap amenity tags or names.
        radius_meters: The search radius around the coordinates.

    Returns:
        A list of dictionaries, each representing a POI:
        [{'name': str, 'latitude': float, 'longitude': float, 'tags': dict}]
        or None if an error occurred.
    """
    lat, lon = coords[0], coords[1]

    # Construct Overpass QL query
    # This query looks for nodes/ways/relations tagged with amenity=category OR name~category (case-insensitive regex)
    # within the specified radius around the coordinates.
    # Timeout set for the query execution on the server. Data size limit.
    # Adjust query based on common OSM tags for different categories (e.g., tourism=museum, shop=*)
    # Using a simple amenity tag search first:
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="{category}"](around:{radius_meters},{lat},{lon});
      way["amenity"="{category}"](around:{radius_meters},{lat},{lon});
      relation["amenity"="{category}"](around:{radius_meters},{lat},{lon});
    );
    out center;
    """
    # Alternative query trying name regex (more complex, might be slower)
    # query = f"""
    # [out:json][timeout:25];
    # (
    #   node[~"^(amenity|tourism|shop)$"~"{category}",i](around:{radius_meters},{lat},{lon});
    #   way[~"^(amenity|tourism|shop)$"~"{category}",i](around:{radius_meters},{lat},{lon});
    #   relation[~"^(amenity|tourism|shop)$"~"{category}",i](around:{radius_meters},{lat},{lon});
    # );
    # out center;
    # """

    # print(f"[Tool Log] Requesting POIs with query: {query}") # Optional logging

    try:
        response = requests.post(OVERPASS_API_URL, data=query, timeout=30) # Increased timeout
        response.raise_for_status()
        data = response.json()

        pois = []
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            name = tags.get('name', f"Unnamed {category}") # Default name if none tagged

            # Get coordinates (different for nodes vs ways/relations)
            if element['type'] == 'node':
                poi_lat, poi_lon = element.get('lat'), element.get('lon')
            elif 'center' in element: # Use center for ways/relations
                poi_lat, poi_lon = element['center'].get('lat'), element['center'].get('lon')
            else: # Skip if no coords
                continue

            if poi_lat is not None and poi_lon is not None:
                 # Include essential tags if needed later
                poi_info = {
                    "name": name,
                    "latitude": poi_lat,
                    "longitude": poi_lon,
                    "tags": tags # Store all tags for potential future use
                }
                pois.append(poi_info)

        # print(f"[Tool Log] Found {len(pois)} POIs for category '{category}'.")
        return pois

    except requests.exceptions.RequestException as e:
        # print(f"[Tool Log] Overpass API request failed: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        # print(f"[Tool Log] Failed to parse Overpass response or missing key: {e}")
        return None

# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    print("--- Testing Geocoding ---")
    eiffel_tower_coords = geocode_location("Eiffel Tower, Paris")
    if eiffel_tower_coords:
        print(f"Eiffel Tower: {eiffel_tower_coords}")
    else:
        print("Eiffel Tower geocoding failed.")

    louvre_coords = geocode_location("Louvre Museum") # Relies on Nominatim context or user location if ambiguous
    if louvre_coords:
        print(f"Louvre Museum: {louvre_coords}")
    else:
        print("Louvre Museum geocoding failed.")

    # Ensure coordinates are valid before testing routing/POI
    if eiffel_tower_coords and louvre_coords:
        print("\n--- Testing Routing ---")
        start = (eiffel_tower_coords['latitude'], eiffel_tower_coords['longitude'])
        end = (louvre_coords['latitude'], louvre_coords['longitude'])
        route_info = get_route(start, end)
        if route_info:
            print(f"Route Eiffel Tower to Louvre:")
            print(f"  Distance: {route_info['distance_meters']:.0f} meters")
            print(f"  Duration: {route_info['duration_seconds'] / 60:.1f} minutes")
            print(f"  Geometry points: {len(route_info['geometry'])}")
        else:
            print("Routing failed.")

        print("\n--- Testing POI Search ---")
        nearby_restaurants = find_nearby_pois(start, category="restaurant", radius_meters=500)
        if nearby_restaurants is not None: # Check for None, as empty list is valid
            print(f"Found {len(nearby_restaurants)} restaurants near Eiffel Tower:")
            for poi in nearby_restaurants[:3]: # Print first few
                print(f"  - {poi['name']} ({poi['latitude']:.4f}, {poi['longitude']:.4f})")
        else:
            print("POI search failed.")

    else:
        print("\nSkipping Routing/POI tests due to failed geocoding.")