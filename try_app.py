import streamlit as st
import re
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, GRU, GlobalMaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import requests

# Download the model and GloVe embeddings
model_h5_url = "https://github.com/Karth-i/New_One/raw/main/model1.h5"
glove_embeddings_url = "https://github.com/Karth-i/New_One/raw/main/glove.6B.50d.txt"

model_dir = "model1"
glove_dir = "glove"

response = requests.get(model_h5_url)
with open("model1.h5", "wb") as f:
    f.write(response.content)

response = requests.get(glove_embeddings_url)
with open("glove.6B.50d.txt", "wb") as f:
    f.write(response.content)

# Load GloVe embeddings
embeddings_index = {}
with open("glove.6B.50d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        if len(values) == 0:
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Load the model
loaded_model = tf.keras.models.load_model("model1.h5")

# Define text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'http\S+', '', text)  # Remove links
    return text

# Define sentiment analysis function
def predict_sentiment(message):
    max_words = 9000
    maxlen = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts([message])
    sequences = tokenizer.texts_to_sequences([message])
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)

    try:
        predictions = loaded_model.predict(padded_sequences)
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

                # Preprocess the message
                message = preprocess_text(message)

                # Add preprocessed message to the corresponding user's list
                user_messages[current_user].append(message)

    # Select user
    selected_user = st.selectbox("Select a user name:", list(user_messages.keys()))

    # Get messages of the selected user
    messages = user_messages[selected_user]

    # Perform sentiment analysis on the selected user's messages
    for message in messages:
        st.write(f"Message: {message} - Sentiment: {predict_sentiment(message)}")
