import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from PIL import Image

# Load environment variables
load_dotenv()

# Configure Google Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model with generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize chat session with system message
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "You are a chatbot which works on behalf of IRTC. Address all complaints from passengers of IRTC, "
                "categorizing them based on cleanliness, ticketing issues, delays, food quality, amenities, or behavior. "
                "Use image recognition when possible."
            ],
        },
        {
            "role": "model",
            "parts": [
                "I will categorize complaints based on cleanliness, ticketing issues, delays, food quality, amenities, or behavior. "
                "If an image is provided, I will analyze it for automatic categorization. I'll confirm the category with the user."
            ],
        },
    ]
)

# Function to get response from Gemini model
def get_gemini_response(user_input, image=None):
    try:
        response_texts = []
        if image:
            # Simulate handling the image as part of the complaint process
            # You can integrate image analysis with a relevant service
            response_texts.append("Image received and being analyzed.")
        
        # Get response from Gemini model
        response = chat_session.send_message(user_input)
        
        if response and hasattr(response, 'text') and response.text:
            response_texts.append(response.text)
        else:
            response_texts.append("No valid response received.")
        
        return "\n".join(response_texts)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, there was an error processing your request."

# Initialize Streamlit app
st.set_page_config(page_title="IRTC Complaint Bot")

st.title("IRTC Complaint Bot")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display chat history
st.subheader("Chat History")
for role, text in st.session_state['chat_history']:
    st.markdown(f"**{role}:** {text}")

# Input field for user's message
user_prompt = st.text_input("Type your complaint or question here...", key="user_input")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

# Handle send button
if st.button("Send"):
    if user_prompt:
        # Add user's message to chat history
        st.session_state['chat_history'].append(("You", user_prompt))
        
        # Process the image (if uploaded)
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            image = None

        # Get response from Gemini model
        response_text = get_gemini_response(user_prompt, image)
        
        if response_text:
            # Add model's response to chat history
            st.session_state['chat_history'].append(("Bot", response_text))
        
        # Clear the input field by resetting the text input widget
        st.rerun()
