import os
import streamlit as st
import google.generativeai as genai

# Retrieve the API key and model name from st.secrets
API_KEY = st.secrets["general"]["GEMINI_API_KEY"]
MODEL_NAME = st.secrets["general"]["MODEL_NAME"]

# Configure the Gemini API with the provided API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="FinBodu-10K",
    page_icon=":robot_face:",
    layout="wide",
)

# Initialize the chat history in the session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set the application title and subheader
st.title("FinBody-10K 💰🤖: Your Financial Chat Assistant")
st.subheader("Hello! I'm FinBody-10K. Ready to assist you with all your finance-related questions. Let's dive into the world of finance together! 📈📊")

# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input from the chat input field
user_input = st.chat_input("Type your question and press Enter:")
if user_input:
    # Display the user's message in the chat interface
    st.chat_message("user").markdown(user_input)
    # Append the user's message to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate a response from the Gemini API based on the user input
    response = model.generate_content(user_input)
    assistant_reply = response.text

    # Display the assistant's response in the chat interface
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    # Append the assistant's response to the chat history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
