import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
import polyline
from geopy.distance import geodesic
from datetime import datetime, timedelta, time as dt_time
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import time
import yfinance as yf

# Function to get coordinates from an address using Nominatim API
def get_coordinates_from_address(address):
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'addressdetails': 1,
        'limit': 1
    }
    response = requests.get(nominatim_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            st.error(f"Address '{address}' not found.")
            return None
    else:
        st.error(f"Nominatim API request failed: {response.status_code}")
        return None

# Function to calculate travel time based on distance and conditions
def estimate_travel_time(distance_km, departure_time):
    average_speed_kmh = 60  # Assuming average speed of 60 km/h
    estimated_time_hours = distance_km / average_speed_kmh
    estimated_time = timedelta(hours=estimated_time_hours)

    # Adjust estimated travel time based on time of day (rush hour adjustments)
    current_time = departure_time.time()
    if dt_time(8, 0) <= current_time <= dt_time(10, 0) or dt_time(17, 0) <= current_time <= dt_time(20, 0):
        estimated_time += timedelta(minutes=30)

    # Further adjust travel time based on distance ranges
    if 5 < distance_km < 10:
        estimated_time += timedelta(minutes=20)
    elif 10 < distance_km < 20:
        estimated_time += timedelta(minutes=40)
    elif distance_km > 20:
        estimated_time += timedelta(minutes=50)

    return estimated_time

# Function to fit ARIMA model
def fit_arima(df):
    model = ARIMA(df['Value'], order=(5, 1, 2))  # Adjusted order for better performance
    arima_model = model.fit()
    return arima_model

# Function to generate ARIMA forecast
def generate_arima_forecast(arima_model, steps=10):
    forecast = arima_model.forecast(steps=steps)
    return forecast

# Main Streamlit App
st.title("Integrated Route & Travel Time Estimation with ARIMA Forecasting")

# Taking user input for source and destination addresses
st.write("Enter the addresses for the source and destination.")
source_address = st.text_input("Enter Source Address", value="Banshankari, Bangalore")
dest_address = st.text_input("Enter Destination Address", value="Vaishnavi Tech Park, Bangalore")

# Time of travel input
departure_time = st.time_input("Enter the departure time:", value=None)

# Button to generate route
if st.button("Generate Route"):
    # Get coordinates from addresses
    source_coords = get_coordinates_from_address(source_address)
    dest_coords = get_coordinates_from_address(dest_address)

    if source_coords and dest_coords:
        source = (source_coords[1], source_coords[0])  # Order: (lon, lat)
        dest = (dest_coords[1], dest_coords[0])

        start = "{},{}".format(source[0], source[1])
        end = "{},{}".format(dest[0], dest[1])

        # OSRM API for driving directions
        url = f'http://router.project-osrm.org/route/v1/driving/{start};{end}?alternatives=false&overview=full'
        headers = {'Content-type': 'application/json'}
        r = requests.get(url, headers=headers)

        if r.status_code == 200:
            routejson = r.json()
            geometry = routejson['routes'][0]['geometry']
            coordinates = polyline.decode(geometry)

            # Creating a dataframe for the coordinates
            df_out = pd.DataFrame(coordinates, columns=['lat', 'long'])

            # Plotting the route on a map
            fig = go.Figure()
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lat=df_out['lat'],
                lon=df_out['long'],
                line=dict(width=4, color='blue'),
                name="Route"
            ))

            fig.add_trace(go.Scattermapbox(
                mode="markers+text",
                lat=[source_coords[0], dest_coords[0]],
                lon=[source_coords[1], dest_coords[1]],
                marker=dict(size=10, color='red'),
                text=["Start", "End"],
                textposition="bottom center"
            ))

            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    zoom=8,
                    center=dict(lat=(source_coords[0] + dest_coords[0]) / 2, lon=(source_coords[1] + dest_coords[1]) / 2)
                ),
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=600,
                width=900
            )

            st.plotly_chart(fig)

            # Calculate straight-line distance
            distance_km = geodesic((source_coords[0], source_coords[1]), (dest_coords[0], dest_coords[1])).kilometers
            st.write(f"**Real-line Distance:** {distance_km:.2f} km")

            # # Estimate travel time
            # departure_datetime = datetime.combine(datetime.today(), departure_time)
            # estimated_travel_time = estimate_travel_time(distance_km, departure_datetime)
            # arrival_time = departure_datetime + estimated_travel_time
            # st.write(f"**Estimated Travel Time (approx.):** {estimated_travel_time}")
            # st.write(f"**Estimated Arrival Time:** {arrival_time.strftime('%H:%M:%S')}")
            
            # Estimate travel time (Reverse Logic)
            arrival_datetime = datetime.combine(datetime.today(), departure_time)
            estimated_travel_time = estimate_travel_time(distance_km, arrival_datetime)
            departure_time = arrival_datetime - estimated_travel_time - timedelta(minutes=10)
            
            actual_time_range1 = departure_time - timedelta(minutes=5)
            actual_time_range2 = departure_time + timedelta(minutes=5)
            st.write(f"**Estimated Travel Time (approx.):** {estimated_travel_time}")
            st.write(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%H:%M')} to {actual_time_range2.strftime('%H:%M')}")
        else:
            distance_km = geodesic((source_coords[0], source_coords[1]), (dest_coords[0], dest_coords[1])).kilometers
            st.write(f"**Real-line Distance:** {distance_km:.2f} km")
            arrival_datetime = datetime.combine(datetime.today(), departure_time)
            estimated_travel_time = estimate_travel_time(distance_km, arrival_datetime)
            departure_time = arrival_datetime - estimated_travel_time - timedelta(minutes=10)
            
            actual_time_range1 = departure_time - timedelta(minutes=5)
            actual_time_range2 = departure_time + timedelta(minutes=5)
            st.write(f"**Estimated Travel Time (approx.):** {estimated_travel_time}")
            st.toast(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%H:%M')} to {actual_time_range2.strftime('%H:%M')}")
            st.write(f"**Optimal Departure Time slot:** {actual_time_range1.strftime('%H:%M')} to {actual_time_range2.strftime('%H:%M')}")
            
            #st.error("OSRM API request failed.")
