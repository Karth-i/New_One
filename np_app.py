import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import requests
import re
from collections import defaultdict
import requests
from tensorflow.keras.initializers import Orthogonal
# Load the saved model
url = 'https://github.com/Karth-i/New_One/raw/main/new_np1.keras'
model = load_model(url)
# model = load_model('https://github.com/Karth-i/New_One/blob/main/new_np1.keras')
response = requests.get(model_url)

with open("new_np1.keras", "wb") as f:
    f.write(response.content)
model = tf.keras.models.load_model("new_np1.keras")
# Create the Streamlit interface
st.title('User Sentiment Analysis')

# Allow users to upload a file
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # Prepare the file for prediction
    text = uploaded_file.read().decode('utf-8')
    predictions = model.predict([text])
    user_sentiment = np.argmax(predictions)
    
    # Display the prediction
    st.write(f"The predicted sentiment for the uploaded file is: {'Positive' if user_sentiment == 1 else 'Negative'}")
    
    # Create a drop-down displaying the present users
    users = ['User1', 'User2', 'User3']
    selected_user = st.selectbox('Select a user', users)
    
    # Display the prediction for the selected user
    if selected_user is not None:
        st.write(f"The predicted sentiment for the selected user is: {'Positive' if np.argmax(model.predict([selected_user])) == 1 else 'Negative'}")
