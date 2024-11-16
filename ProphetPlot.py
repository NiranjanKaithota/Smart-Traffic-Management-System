import pickle
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

##st.set_page_config(
##    page_title="Yukthi",
##    page_icon="🎉",
##    layout="wide"
##)

class ProphetModel:
    def predict(self, time2):
        # Load the pre-trained Prophet model
        with open(r'D:\Mob\myenv\prophet_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Define the current time for prediction
        date_string = "2024-10-01 08:00:00"
        global time1
        time1 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

        # Calculate the difference between time1 and time2 in minutes
        time_difference = time2 - time1
        minutes_difference = time_difference.total_seconds() / 60

        # Calculate the number of periods based on 5-minute intervals
        period = int(minutes_difference / 5)

        # Generate future timestamps for forecasting
        future = loaded_model.make_future_dataframe(periods=period + 1, freq='5min')
        # Predict future values
        forecast = loaded_model.predict(future)
        filtered_forecast = forecast[(forecast['ds'] <= time2) & (forecast['ds'] >= time1)]
        return filtered_forecast

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv(r"D:\Mob\myenv\traffic_synthetic.csv")
    df_prophet = df.reset_index().rename(columns={'time': 'ds', 'pcu': 'y'})

    # Specify the time2 for which you want to predict
    date_string = "2024-10-01 15:30:00"
    time2 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    # Initialize the model and predict
    model = ProphetModel()
    forecast = model.predict(time2)
    
    # Convert 'ds' columns to datetime in both actual and forecast datasets
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Create the figure using Plotly
    fig = go.Figure()

    # Add the actual data trace
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Data'))

    # Add the forecasted data trace
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast Data', line=dict(color='green', width=2, dash='dot')))

    # Customize layout
    fig.update_layout(title="Actual vs Forecast Data", xaxis_title="Time", yaxis_title="PCU", legend_title="Data Type")

    # Create a button to zoom in between time1 and time2
    if st.button("View"):
        # If button is pressed, zoom between time1 and time2
        fig.update_xaxes(range=[time1, time2])

    # Render the Plotly chart with Streamlit
    st.plotly_chart(fig)
    
    #if st.button("Back"):
    #    welcome_page()