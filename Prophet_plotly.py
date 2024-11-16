import pickle
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go

class ProphetModel:
    def predict(self, time2):
        # Load the pre-trained Prophet model
        with open(r'myenv\prophet_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Define the current time for prediction (can be dynamically adjusted)
        date_string = "2024-10-01 17:00:00"
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

        return forecast

if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv(r"myenv\traffic_synthetic.csv")
    df_prophet = df.reset_index().rename(columns={'time': 'ds', 'pcu': 'y'})

    # Specify the time2 for which you want to predict
    date_string = "2024-10-01 23:00:00"
    time2 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

    # Initialize the model and predict
    model = ProphetModel()
    forecast = model.predict(time2)

    # Convert 'ds' columns to datetime in both actual and forecast datasets
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    # Plot actual vs forecasted values
    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual'))

    # Plot forecast data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Update plot layout
    fig.update_layout(
        title='Time Series Plot: Actual vs Forecast',
        xaxis_title='Time',
        yaxis_title='Values'
    )

    # Display the plot
    fig.show()
