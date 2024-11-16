import streamlit as st
import pandas as pd
import os

# Function to handle login and redirect to welcome page
def handle_login(email, password):
    # For simplicity, assume successful login
    if email and password:
        st.session_state['logged_in'] = True
        st.session_state['email'] = email
        st.success("Login successful!")

# Function to create a new account
def create_new_account():
    st.header("Create New Account")

    # Collect essential data for new account creation
    name = st.text_input("Name", key="create_name")
    email = st.text_input("Email", key="create_email")
    password = st.text_input("Password", type="password", key="create_password")
    workspace = st.text_input("Workspace", key="create_workspace")
    home_address = st.text_area("Home Address", key="create_address")

    if st.button("Create Account", key="create_account_btn"):
        with open("users.txt", "a") as file:
            file.write(f"{name},{email},{password},{workspace},{home_address}\n")
        st.success("Account created successfully!")
        st.info("You can now log in using your credentials.")

# Main login page
def login_page():
    st.header("Login Page")

    # Prompt user for email and password
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    # Log In button
    if st.button("Log In", key="login_btn"):
        handle_login(email, password)

    # Sign in with Google and Facebook (placeholders)
    if st.button("Sign in through Google", key="google_signin_btn"):
        st.info("Google login is currently under development.")
    if st.button("Sign in through Facebook", key="facebook_signin_btn"):
        st.info("Facebook login is currently under development.")

# Function to collect commute route details
def commute_form():
    st.header("Welcome!")
    st.subheader("Please fill in your commute details")

    # Input fields
    source = st.text_input("Source", key="source")
    destination = st.text_input("Destination", key="destination")

    junctions = ["Silk Board", "Sarjapur Road", "Bellandur", "Kadubeesanahalli",
                 "Marathahalli", "Mahadevpura", "KR Puram"]
    junction = st.selectbox("Usual Junction of Commute", junctions, key="junction")

    mode_of_commute = st.selectbox("Mode of Commute", ["Car", "Bike", "Bus", "Walk"], key="mode_of_commute")

    if st.button("Submit", key="submit_commute"):
        # Save the details into a CSV file
        data = {
            "Email": [st.session_state['email']],
            "Source": [source],
            "Destination": [destination],
            "Junction": [junction],
            "Mode of Commute": [mode_of_commute]
        }
        df = pd.DataFrame(data)
        if not os.path.isfile("commute_route.csv"):
            df.to_csv("commute_route.csv", index=False)
        else:
            df.to_csv("commute_route.csv", mode='a', header=False, index=False)

        st.success("Commute details saved successfully!")
        
        second_app_url = "http://localhost:8501"  # Replace with the URL of the second Streamlit app
        
        # Redirect to the second Streamlit app
        st.markdown(f'<meta http-equiv="refresh" content="0; url={second_app_url}">', unsafe_allow_html=True)
        
        #if st.button("Run Map"):
        #    st.write("Running Map Interface...")
        #    # Execute the second streamlit app using the os module
        #    os.system("streamlit run Map_Interface.py")

    
# Main function to handle page navigation
def main():
    st.title("Commute Tracker App")
    i=0
    # Initialize session state for  login tracking
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        

    # Check if user is logged in
    if st.session_state['logged_in']:
        commute_form()  # Show the commute form if logged in
    else:
        page = st.selectbox("Select an option", ["Login", "Create Account"])

        if page == "Login":
            login_page()
        elif page == "Create Account":
            create_new_account()

if __name__ == "__main__":
    main()
