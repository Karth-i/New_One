import streamlit as st
import re
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
import zipfile
import os
import requests
from keras.initializers import glorot_uniform, Orthogonal

# Set custom objects for initializers
custom_objects = {'Orthogonal': Orthogonal, 'glorot_uniform': glorot_uniform,'tf': tf, 'GRU': GRU}

# Download and extract the model zip file
model_h5_url = "https://github.com/Karth-i/New_One/raw/main/model1.h5"
model_dir = "model1"

response = requests.get(model_h5_url)
with open("model1.h5", "wb") as f:
    f.write(response.content)

# Load model
try:
    loaded_model = keras.models.load_model("model1.h5", custom_objects=custom_objects)
except Exception as e:
    st.write(f"Error loading the model: {e}")
    st.stop()

# Download and extract GloVe embeddings zip file
glove_embeddings_zip_url = "https://github.com/Karth-i/New_One/raw/main/glove.6B.50d.txt.zip"
glove_dir = "glove"

response = requests.get(glove_embeddings_zip_url)
with open("glove.zip", "wb") as f:
    f.write(response.content)

with zipfile.ZipFile("glove.zip", 'r') as zip_ref:
    zip_ref.extractall(glove_dir)

# Load GloVe embeddings
path_to_glove_file = os.path.join(glove_dir, "glove.6B.50d.txt")
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

# Define text preprocessing function
def extract_english_words(text):
    english_words = []
    # Regular expression pattern to match English words
    english_pattern = re.compile(r'\b[a-zA-Z]+\b')
    # Find all English words in the text
    english_words = english_pattern.findall(text)
    return english_words

# Define sentiment analysis function
def predict_sentiment(message):
    max_words = 9000
    maxlen = 200

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_words,
        output_mode='int',
        output_sequence_length=maxlen
    )

    # Adapt the layer to the new message
    vectorize_layer.adapt([message])

    # Convert the message into numerical vectors
    message_vectors = vectorize_layer([message])

    # Add a batch dimension to the message vectors
    message_vectors = tf.expand_dims(message_vectors, axis=0)

    try:
        # Use the model to predict the sentiment of the message
        predictions = loaded_model.predict(message_vectors)

        # Get the predicted sentiment label
        predicted_label = np.argmax(predictions, axis=1)[0]

        if predicted_label == 0:
            return "Negative Sentiment Detected !!!"
        elif predicted_label == 1:
            return "Neutral Sentiment Detected"
        else:
            return "Positive Sentiment Detected"

    except Exception as e:
        return f"Error occurred: {e}"

# Streamlit interface
st.title("WhatsApp Chat Sentiment Analysis")

# Upload WhatsApp chat history file
uploaded_file = st.file_uploader("Upload WhatsApp Chat History")

if uploaded_file is not None:
    # Save uploaded file
    with open("whatsapp_chat.txt", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Process the WhatsApp chat history file
    user_messages = defaultdict(list)
    with open("whatsapp_chat.txt", 'r', encoding='utf-8') as file:
        # Read the lines of the WhatsApp chat history file
        lines = file.readlines()

    current_user = None
    for line in lines:
        # Split each line by the timestamp and sender's name
        parts = line.split(' - ')
        if len(parts) > 1:
            user_message = parts[1].strip()  # Extract the user name and message content
            user_parts = user_message.split(': ')
            if len(user_parts) > 1:
                user = user_parts[0].strip()  # Extract the user name
                message = ': '.join(user_parts[1:]).strip()  # Reconstruct the message part

                # Check if the user has changed
                if user != current_user:
                    # If the user has changed, reset the current_user variable
                    current_user = user
                    # Initialize an empty list for the user if it's the first message from that user
                    if not user_messages[current_user]:
                        user_messages[current_user] = []

                # Extract English words from the message and add them to the corresponding user's list
                english_words = extract_english_words(message)
                user_messages[current_user].extend(english_words)

    # Select user
    selected_user = st.selectbox("Select a user name:", list(user_messages.keys()))

    # Get messages of the selected user
    messages = user_messages[selected_user]

    # Perform sentiment analysis on the selected user's messages
    for message in messages:
        st.write(f"Message: {message} - Sentiment: {predict_sentiment(message)}")
