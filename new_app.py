import requests
import tensorflow as tf
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import streamlit as st

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text: str) -> str:
    """Preprocess text for sentiment analysis"""
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a string
    text = " ".join(words)
    return text

def load_model() -> tf.keras.Model:
    """Load the sentiment analysis model"""
    model_url = "https://github.com/Karth-i/New_One/raw/9ba3e1c71a83bf70df186c342b837a9745721849/model1.h5"
    response = requests.get(model_url)
    with open("model.h5", "wb") as f:
        f.write(response.content)
    
    model = tf.keras.models.load_model("model.h5", compile=False)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GRU):
            layer.recurrent_initializer = tf.keras.initializers.glorot_uniform()
    return model

def predict_sentiment(model: tf.keras.Model, text: str) -> str:
    """Predict the sentiment of a given text"""
    # Preprocess text
    text = preprocess_text(text)
    # Tokenize text
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(["start", text, "end"])
    sequence = tokenizer.texts_to_sequences(["start", text, "end"])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding="post")
    # Predict sentiment
    sentiment = model.predict(sequence.reshape(1, -1))
    # Map sentiment to label
    if sentiment > 0.5:
        label = "Positive"
    else:
        label = "Negative"
    return label

def main():
    """Sentiment Analysis for Karthi's messages"""
    # Load model
    model = load_model()
    # Set up Streamlit app
    st.set_page_config(page_title="Sentiment Analysis for Karthi's messages")
    st.title("Sentiment Analysis for Karthi's messages")
    # Get user input
    user_input = st.text_input("Enter a message:")
    if user_input:
        # Predict sentiment
        sentiment = predict_sentiment(model, user_input)
        # Display sentiment
        st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
