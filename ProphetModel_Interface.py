import streamlit as st
import pickle
from datetime import datetime,time
import plotly.express as px

# ProphetModel class remains the same
class ProphetModel:
    def predict(self, time2):
        # Load the pre-trained Prophet model
        with open('prophet_final.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Time format for parsing
        time_format = "%H:%M:%S"
        
        # Hardcoded start time (can change this to current datetime if needed)
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

# Streamlit App Code with Customizations
def main():
    # Custom CSS for background and button styling
    st.markdown("""
    <style>
    /* Set background image and other page styles */
    body {
        background-image: url('https://via.placeholder.com/1500');
        background-size: cover;
        background-position: center;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    
    /* Style for the buttons */
    div.stButton > button {
        background-color: #ff6347;
        color: white;
        border: 2px solid #ff6347;
        border-radius: 8px;
        padding: 10px;
        font-size: 18px;
        transition: background-color 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #ff4500;
        color: white;
    }
    
    /* Change font and text color */
    .stTextInput label, .stTimeInput label, .stDateInput label {
        font-size: 20px;
        color: white;
    }

    /* Style the title */
    h1 {
        font-size: 50px;
        color: #ffcc00;
        text-shadow: 2px 2px #ff4500;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and image for the page
    st.title("📊 Prophet Model Prediction")
    
    # Display a resized image (you can replace this with any relevant image)
    ##st.image("D:/Mob/myenv/Image_Ben", width=750, caption="Future Predictions Visualization")
    
    default_date = datetime(2023, 4, 13).date()

    # Use the default date or custom date if necessary
    date_input = st.date_input("Select Date", value=default_date)

    # Allow user to input time, with no default to current time
    custom_time = time(12, 0)  # Example custom default time (12:00 PM)
    time_input = st.time_input("Select Time", value=custom_time)

    # Combine date and time inputs into a datetime object
    selected_datetime = datetime.combine(date_input, time_input)

    # Create an instance of the model
    model = ProphetModel()

    # Button to trigger prediction
    if st.button('Predict'):
        # Call the predict function
        predictions = model.predict(selected_datetime)

        
        
        # Display the predictions on the Streamlit app
        if not predictions.empty:
            st.write("### Predicted Values:")
            st.dataframe(predictions[["ds", "yhat"]])  # Displaying predicted datetime and values
            
            # Plot the data using Plotly
            fig = px.line(predictions, x="ds", y="yhat", title="Predicted Values Over Time",
                          labels={"ds": "Date/Time", "yhat": "Predicted Value"})
            
            # Display the plot
            st.plotly_chart(fig)
        else:
            st.write("No predictions available for the selected time range.")

if __name__ == "__main__":
    main()
 