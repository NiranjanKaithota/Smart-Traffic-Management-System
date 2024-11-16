import pickle
from datetime import datetime, timedelta

class ProphetModel:
    def predict(self, time2):
        # Load the pre-trained Prophet model
        with open('prophet_final.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Time format for parsing
        time_format = "%H:%M:%S"
        # Get current time and format it
        # time1 = datetime.now()
        date_string = "2023-04-13 17:45:00"

        # Parse the date string into a datetime object
        time1 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        start_time = time1.strftime(time_format)
        end_time = time2.strftime(time_format)
        # Convert times to datetime objects
        start_time = datetime.strptime(start_time, time_format)
        end_time = datetime.strptime(end_time, time_format)

        # Calculate the difference between the two times in minutes
        time_difference = end_time - start_time
        minutes_difference = time_difference.total_seconds() / 60
        # Calculate the number of periods based on 5-minute intervals
        period = int(minutes_difference / 5)
        # Generate future timestamps based on the calculated period
        future = loaded_model.make_future_dataframe(periods=period + 1, freq='5T')
        # Use the loaded model to predict the future PSU values (or any target values)
        forecast = loaded_model.predict(future)
        # Filter predictions that are within the range of now and time2
        filtered_forecast = forecast[(forecast['ds'] <= time2) & (forecast['ds'] >= time1)]
        return filtered_forecast

if __name__ == "__main__":
    # Example: Predict PSU values for the next hour
    # time2 = datetime.now() + timedelta(hours=1)
    date_string = "2023-04-13 11:30:00"

# Parse the date string into a datetime object
    time2 = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    model = ProphetModel()
    # Call the predict function
    predictions = model.predict(time2)
    # Print the predictions
    print(predictions["ds"].dtype)
    print(predictions)