import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import time
import streamlit as st
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime, timedelta
from geopy.distance import geodesic
import yfinance as yf
from threading import Thread

# Function to geocode an address using Nominatim
def geocode_address(address):
    geolocator = Nominatim(user_agent="streamlit_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        return None

# Function to create a Folium map with markers and route
def create_map(start_coords, dest_coords, start_address, destination_address):
    # Center the map between start and destination
    avg_lat = (start_coords[0] + dest_coords[0]) / 2
    avg_lon = (start_coords[1] + dest_coords[1]) / 2
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

    # Add start marker
    folium.Marker(
        location=start_coords,
        popup="Start: " + start_address,
        icon=folium.Icon(color='green')
    ).add_to(m)

    # Add destination marker
    folium.Marker(
        location=dest_coords,
        popup="Destination: " + destination_address,
        icon=folium.Icon(color='red')
    ).add_to(m)

    # Draw a line between start and destination
    folium.PolyLine(
        locations=[start_coords, dest_coords],
        weight=5,
        color='blue',
        opacity=0.8
    ).add_to(m)

    return m

# Function to fit ARIMA model
def fit_arima(df):
    model = ARIMA(df['Value'], order=(5, 1, 2))  # Adjusted order for better performance
    arima_model = model.fit()
    return arima_model

# Function to generate ARIMA forecast
def generate_arima_forecast(arima_model, steps=10):
    forecast = arima_model.forecast(steps=steps)
    return forecast

# Function to fetch live data from Yahoo Finance
def fetch_live_data(ticker, period='1d', interval='1m'):
    data = yf.download(tickers=ticker, period=period, interval=interval)
    data = data.reset_index()
    data = data[['Datetime', 'Close']]
    data.rename(columns={'Datetime': 'Date', 'Close': 'Value'}, inplace=True)
    data.set_index('Date', inplace=True)
    return data

# Function to continuously fetch live data and update the dataframe
def update_data(ticker, df, interval=60):
    while True:
        new_data = fetch_live_data(ticker, period='1d', interval='1m')
        if not new_data.empty:
            df_new = new_data[~new_data.index.isin(df.index)]
            if not df_new.empty:
                df = pd.concat([df, df_new])
                df = df[~df.index.duplicated(keep='last')]
                st.session_state['df'] = df
        time.sleep(interval)

# Streamlit App
def main():
    st.set_page_config(page_title="Live Data & ARIMA Forecasting App", layout="wide")
    st.title("Location Pinpoint, Route Visualization & Live Data Forecasting App")

    # Initialize session state for data
    if 'df' not in st.session_state:
        st.session_state['df'] = pd.DataFrame()

    # Section 1: Pinpoint a Single Location
    st.header("Pinpoint a Single Location")
    single_location = st.text_input("Enter a location (city, address, etc.):")

    if single_location:
        with st.spinner("Geocoding the location..."):
            single_coords = geocode_address(single_location)
            time.sleep(1)  # To ensure RateLimiter's delay is respected

        if single_coords:
            st.success(f"Location found: {single_location} (Lat: {single_coords[0]:.4f}, Lon: {single_coords[1]:.4f})")
            single_map = folium.Map(location=single_coords, zoom_start=12)
            folium.Marker(
                location=single_coords,
                popup=f"Location: {single_location}",
                icon=folium.Icon(color='blue')
            ).add_to(single_map)
            folium_static(single_map)
        else:
            st.error("Could not find the location. Please check your input.")

    st.markdown("---")

    # Section 2: Calculate and Visualize Route Between Two Locations
    st.header("Calculate and Visualize Route Between Two Locations")

    with st.form("route_form"):
        start_address = st.text_input("Enter start address:")
        destination_address = st.text_input("Enter destination address:")
        departure_time = st.time_input("Enter the departure time:", value=datetime.now().time())
        submitted = st.form_submit_button("Visualize Route")

    if submitted:
        if not start_address or not destination_address:
            st.error("Please enter both start and destination addresses.")
        else:
            with st.spinner("Geocoding the addresses..."):
                start_coords = geocode_address(start_address)
                dest_coords = geocode_address(destination_address)
                time.sleep(1)  # To ensure RateLimiter's delay is respected

            if not start_coords:
                st.error(f"Start address '{start_address}' not found.")
            elif not dest_coords:
                st.error(f"Destination address '{destination_address}' not found.")
            else:
                st.success("Both locations found successfully!")

                # Create and display the map with route
                route_map = create_map(start_coords, dest_coords, start_address, destination_address)
                folium_static(route_map)

                # Calculate straight-line distance
                distance_km = geodesic(start_coords, dest_coords).kilometers
                st.write(f"**Straight-line Distance:** {distance_km:.2f} km")

                # Estimate travel time assuming an average speed (e.g., 60 km/h)
                average_speed_kmh = 60  
                estimated_time_hours = distance_km / average_speed_kmh
                estimated_time = timedelta(hours=estimated_time_hours)
                departure_datetime = datetime.combine(datetime.today(), departure_time)
                arrival_time = departure_datetime + estimated_time
                st.write(f"**Estimated Travel Time (approx.):** {estimated_time}")
                st.write(f"**Estimated Arrival Time:** {arrival_time.strftime('%H:%M:%S')}")

                # Optional: Display departure and arrival times
                st.write(f"**Departure Time:** {departure_datetime.strftime('%H:%M:%S')}")
                st.write(f"**Arrival Time:** {arrival_time.strftime('%H:%M:%S')}")

                # Additional Information
                st.info("**Note:** Travel time estimation is based on straight-line distance and an assumed average speed. For precise travel times, integrating a routing service API is recommended.")

    st.markdown("---")

    # Section 3: ARIMA Forecasting with Live Data
    st.header("Live Data & ARIMA Forecasting")

    # Sidebar for selecting stock ticker
    st.sidebar.header("Live Data Configuration")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, GOOGL):", value="AAPL")
    update_interval = st.sidebar.slider("Data Update Interval (seconds):", min_value=30, max_value=300, value=60, step=30)
    forecast_steps = st.sidebar.slider("Number of Forecast Steps:", min_value=1, max_value=60, value=10, step=1)

    # Button to start live data fetching
    if st.sidebar.button("Start Live Data"):
        if 'thread' not in st.session_state:
            # Initialize data
            with st.spinner("Fetching initial data..."):
                df_initial = fetch_live_data(ticker, period='1d', interval='1m')
                if df_initial.empty:
                    st.sidebar.error("Failed to fetch data. Please check the ticker symbol.")
                else:
                    st.session_state['df'] = df_initial
                    st.success(f"Initial data for {ticker} fetched successfully!")

            # Start background thread for data updating
            thread = Thread(target=update_data, args=(ticker, st.session_state['df'], update_interval), daemon=True)
            thread.start()
            st.session_state['thread'] = thread
            st.sidebar.success("Live data fetching started!")
        else:
            st.sidebar.info("Live data is already being fetched.")

    # Display live data if available
    if not st.session_state['df'].empty:
        df = st.session_state['df']

        # Display the latest data
        st.subheader(f"Live Data for {ticker}")
        st.write(df.tail())

        # Display the time series chart
        st.line_chart(df['Value'])

        # Fit ARIMA model
        with st.spinner("Fitting ARIMA model..."):
            try:
                arima_model = fit_arima(df)
                st.success("ARIMA model fitted successfully!")
            except Exception as e:
                st.error(f"Error fitting ARIMA model: {e}")
                arima_model = None

        # Generate forecast
        if arima_model:
            with st.spinner("Generating forecast..."):
                forecast = generate_arima_forecast(arima_model, steps=forecast_steps)

            # Create forecast DataFrame
            last_date = df.index[-1]
            forecast_dates = pd.date_range(last_date + timedelta(minutes=1), periods=forecast_steps, freq='T')
            forecast_df = pd.DataFrame({'Value': forecast}, index=forecast_dates)

            # Combine actual and forecast data for plotting
            combined_df = pd.concat([df, forecast_df])

            # Plot actual data and forecast
            st.subheader("Actual Data and Forecast")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df['Value'], label='Actual Data', color='blue')
            ax.plot(forecast_df.index, forecast_df['Value'], label='Forecast', color='green', marker='o')
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(f'ARIMA Forecast for {ticker}')
            st.pyplot(fig)

            # Display forecast values
            st.subheader("Forecasted Values")
            st.write(forecast_df)

            st.info("**Note:** The ARIMA model parameters are currently set to order=(5,1,2). Depending on your data, you might need to adjust these parameters for optimal forecasting performance.")

    st.markdown("---")
    st.markdown("© 2024 Your Name. All rights reserved.")

if __name__ == "__main__":
    main()
