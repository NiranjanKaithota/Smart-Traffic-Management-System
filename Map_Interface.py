import streamlit as st
import folium
from streamlit_folium import folium_static
import requests

# Function to get latitude and longitude from a location using Nominatim
def get_lat_lon(location):
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&addressdetails=1"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
    return None, None

# Title of the app
st.title("Pinpoint Location on Map")

# Input for user location
location = st.text_input("Enter a location (city, address, etc.):")

if location:
    lat, lon = get_lat_lon(location)

    if lat is not None and lon is not None:
        # Create a map centered at the location
        m = folium.Map(location=[lat, lon], zoom_start=12)

        # Add a marker for the location
        folium.Marker([lat, lon], popup=f"Location: {location}").add_to(m)

        # Display the map in Streamlit
        folium_static(m)
    else:
        st.error("Could not find the location. Please check your input.")