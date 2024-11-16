import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

# Function to get coordinates from Nominatim (OpenStreetMap)
def get_coordinates(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200 and response.json():
        data = response.json()[0]  # Get the first result
        return float(data['lat']), float(data['lon'])
    else:
        return None

# Function to get directions from Mapbox Directions API
def get_directions(api_key, start, end):
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
    params = {
        "access_token": api_key,
        "geometries": "geojson",
        "overview": "full"
    }
    response = requests.get(url, params=params)
    return response.json()

# Function to plot the route on the map using folium
def plot_route(directions_response):
    route = directions_response['routes'][0]['geometry']['coordinates']
    # Create a map centered at the starting point
    start_lat, start_lon = route[0][1], route[0][0]
    route_map = folium.Map(location=[start_lat, start_lon], zoom_start=13)

    # Convert route coordinates into (latitude, longitude) pairs and plot the route
    coordinates = [(coord[1], coord[0]) for coord in route]
    folium.PolyLine(coordinates, color="blue", weight=5).add_to(route_map)

    # Add start and end markers
    # Add start and end markers
    folium.Marker(location=coordinates[0], popup="Start", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(location=coordinates[-1], popup="End", icon=folium.Icon(color="red")).add_to(route_map)

    return route_map

# Initialize session state for coordinates and the map
if "start_coords" not in st.session_state:
    st.session_state.start_coords = None
if "end_coords" not in st.session_state:
    st.session_state.end_coords = None
if "route_map" not in st.session_state:
    st.session_state.route_map = None

# Streamlit app layout
st.title("Route Finder using Mapbox & Nominatim")
st.markdown("Enter the addresses of two places to get the route displayed on a map.")

# Input fields for addresses
start_address = st.text_input("Start Address", "1600 Amphitheatre Parkway, Mountain View, CA")
end_address = st.text_input("End Address", "1 Infinite Loop, Cupertino, CA")
api_key = st.text_input("Enter your Mapbox API Key", type="password")

# Button to trigger the route finding
if st.button("Find Route"):
    # Get coordinates for both addresses
    st.session_state.start_coords = get_coordinates(start_address)
    st.session_state.end_coords = get_coordinates(end_address)

    if st.session_state.start_coords and st.session_state.end_coords:
        st.success(f"Coordinates found!\nStart: {st.session_state.start_coords}\nEnd: {st.session_state.end_coords}")

        # Fetch directions from Mapbox API
        directions_response = get_directions(api_key, st.session_state.start_coords, st.session_state.end_coords)

        # Check if the route data is valid
        if "routes" in directions_response:
            st.success("Route found!")

            # Plot the route on the map and save it in session state
            st.session_state.route_map = plot_route(directions_response)
        else:
            st.error("Error fetching route. Please check the coordinates or API key.")
    else:
        st.error("Could not find coordinates for one or both addresses. Please check the inputs.")

# Display the route map if it exists in session state
if st.session_state.route_map:
    st_folium(st.session_state.route_map, width=700, height=500)