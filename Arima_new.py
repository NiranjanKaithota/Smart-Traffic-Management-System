import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import time
from PIL import Image
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from ProphetPlot import ProphetModel
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime, timedelta, time as dt_time
from geopy.distance import geodesic
import yfinance as yf
from threading import Thread
import polyline
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie 

st.set_page_config(
    page_title="Real Time Prediction",
    page_icon="🎉",
    layout="wide"
)

def arima():
    # Data creation function
    def data_create():
        start_time = '06:00:00'
        end_time = '22:00:00'
        frequency = '5min'  # 5-minute intervals
        
        time_index = pd.date_range(start=start_time, end=end_time, freq=frequency)
        
        # Create a sinusoidal base pattern for PCU values
        time_in_hours = (time_index.hour + time_index.minute / 60)
        base_pcu = 200 + 100 * np.sin((time_in_hours - 6) / 16 * 2 * np.pi)  # Peaks around midday
        
        # Define peak traffic hours: 9 AM to 11 AM and 5 PM to 7 PM
        morning_peak = (time_index.hour >= 9) & (time_index.hour < 11)
        evening_peak = (time_index.hour >= 17) & (time_index.hour < 19)
        
        # Adjust PCU for peak hours
        pcu = np.where(morning_peak, base_pcu + 150, base_pcu)
        pcu = np.where(evening_peak, pcu + 150, pcu)
        
        # Add some random noise
        noise = np.random.normal(0, 20, len(time_index))
        pcu += noise
        
        # Create a DataFrame with time and PCU values
        df = pd.DataFrame({'time': time_index, 'pcu': pcu})
        df.set_index('time', inplace=True)
        
        return df

    # Generate data and predictions
    df = data_create()

    # Build ARIMA model
    model = ARIMA(df['pcu'], order=(29, 0, 16), trend='c')
    model_fit = model.fit()

    # Predict the full range of values
    predictions = model_fit.predict(start=0, end=len(df) - 1, typ='levels')

    # Streamlit app
    

    # Custom CSS to set the background color
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title('Dynamic ARIMA Prediction for PCU')

    # Create a plot with dark style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set plot limits and labels
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylim(df['pcu'].min(), df['pcu'].max())
    ax.set_title('Dynamic ARIMA Prediction for PCU', color='white', fontsize=16)
    ax.set_xlabel('Time', color='white', fontsize=12)
    ax.set_ylabel('PCU', color='white', fontsize=12)

    # Customize tick labels
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Format x-axis to show only time
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # Create empty line plots for predicted and actual data
    line_pred, = ax.plot([], [], lw=2, label='Predicted Data', color='#FFA500')  # Bright orange
    line_actual, = ax.plot([], [], lw=2, label='Actual Data', color='#00BFFF')  # Deep sky blue

    # Customize legend
    ax.legend(facecolor='#1C1C1C', edgecolor='#1C1C1C', labelcolor='white')

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)

    # Streamlit plot
    plot_placeholder = st.empty()

    # Initialize empty lists to store time points and predictions
    x_data_pred, y_data_pred = [], []
    x_data_actual, y_data_actual = [], []

    # Update function
    def update(frame):
        # Plot the predicted data up to the current frame
        x_data_pred.append(df.index[frame])
        y_data_pred.append(predictions.iloc[frame])
        
        # Plot the actual data only if it is 10 steps behind the predicted data
        if frame >= 10:
            x_data_actual.append(df.index[frame - 10])
            y_data_actual.append(df['pcu'].iloc[frame - 10])
        
        # Update the lines
        line_pred.set_data(x_data_pred, y_data_pred)
        line_actual.set_data(x_data_actual, y_data_actual)
        
        # Redraw the plot
        fig.canvas.draw()
        
        # Update the Streamlit plot
        plot_placeholder.pyplot(fig)

    # Animate the plot
    for frame in range(len(df)):
        update(frame)
        time.sleep(0.01)  # Control the speed of the animation

    # Display final data
    st.subheader('Final Data')
    st.dataframe(df.style.background_gradient(cmap='YlOrRd'))

arima()