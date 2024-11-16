import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st
import time
from statsmodels.tsa.arima.model import ARIMA

# Function to create initial time series data (df)
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
    noise = np.random.normal(0, 15, len(time_index))
    pcu += noise
    
    # Create a DataFrame with time and PCU values
    df = pd.DataFrame({'time': time_index, 'pcu': pcu})
    df.set_index('time', inplace=True)
    
    return df

# Function to predict the next PCU value using ARIMA
def predict_next(df1):
    try:
        # Fit an ARIMA model to df1's PCU values
        model = ARIMA(df1['pcu'], order=(3, 1, 2), trend='t')
        arima_model = model.fit()

        # Predict the next PCU value
        forecast = arima_model.forecast(steps=1)

        if len(forecast) > 0:
            return forecast.iloc[0]  # Use iloc to access the first forecast value safely
        else:
            st.warning("Forecast is empty.")
            return df1['pcu'].iloc[-1]  # Return the last known value if forecast fails
    except ValueError as e:
        # Handle ARIMA errors related to trend terms and differencing silently
        if "trend terms of lower order than d + D" in str(e):
            return df1['pcu'].iloc[-1]  # Return the last known value
        else:
            raise e  # For other errors, raise them normally
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return df1['pcu'].iloc[-1]  # Return the last known value if any error occurs

# Function to update and plot the graph in Plotly
def update_plot(df1, df2):
    # Create the actual PCU plot
    actual_trace = go.Scatter(
        x=df1.index, 
        y=df1['pcu'], 
        mode='lines', 
        name='Actual PCU',
        line=dict(color='blue')
    )

    # Create the predicted PCU plot
    predicted_trace = go.Scatter(
        x=df2.index, 
        y=df2['pcu'], 
        mode='lines', 
        name='Predicted PCU',
        line=dict(color='red', dash='dash')
    )

    # Create the layout
    layout = go.Layout(
        title='PCU Time Series (Actual vs Predicted)',
        xaxis_title='Time',
        yaxis_title='PCU',
        xaxis=dict(
            tickformat='%H:%M',
            nticks=20
        ),
        yaxis=dict(range=[0, max(df1['pcu'].max(), df2['pcu'].max()) + 50]),
        legend=dict(x=0, y=1)
    )

    fig = go.Figure(data=[actual_trace, predicted_trace], layout=layout)
    return fig

# Streamlit app setup
st.title("PCU Traffic Prediction")

# Create the initial data (df)
df = data_create()

# Initialize empty DataFrames for df1 (actual) and df2 (predicted)
df1 = pd.DataFrame(columns=['pcu'])
df2 = pd.DataFrame(columns=['pcu'])

# Create a placeholder for the Plotly chart
chart_placeholder = st.empty()

# Progressively update df1 and df2, and plot the results in real time
for frame in range(len(df)):
    # Add the next row from df to df1 using proper indexing
    df1 = pd.concat([df1, df.iloc[[frame]]])
    
    # Predict the next PCU value based on df1 and store it in df2
    if len(df1) > 5:  # Ensure there is enough data for the ARIMA model
        predicted_pcu = predict_next(df1)
        df2 = pd.concat([df2, pd.DataFrame({'pcu': [predicted_pcu]}, index=[df1.index[-1] + pd.Timedelta(minutes=5)])])
    
    # Update the plot with actual and predicted values
    fig = update_plot(df1, df2)
    
    # Display the updated chart in the placeholder
    chart_placeholder.plotly_chart(fig)
    
    # Add a delay to simulate real-time updates
    time.sleep(0.2)
