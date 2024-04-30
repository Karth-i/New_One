# -*- coding: utf-8 -*-
"""Copy of app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1s0mUtL60iJknE3AR7_yXpeuv8Hfu889P
"""

pip install streamlit

import streamlit as st
import re
from collections import defaultdict
from transformers import RobertaTokenizer
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Embedding
from collections import defaultdict

from google.colab import drive
drive.mount('/content/drive')

model_path = '/content/drive/MyDrive/your_folder/model1'

path_to_glove_file = "/content/glove.6B.50d.txt"
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))

max_words = 9000
maxlen = 200
# Define the custom layers
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_words,
    output_mode='int',
    output_sequence_length=maxlen,
    input_shape = (None,)
)

voc = vectorize_layer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

num_tokens = len(voc) + 2
embedding_dim = 50

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    trainable=False,
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])

loaded_model = keras.models.load_model(model_path)

predictions = loaded_model.predict(["i am loving it but i like being loved in a doom " ])

predicted_labels = np.argmax(predictions, axis=1)

if predicted_labels == 0:
  print("Negative Sentiment Detetded !!!")
elif predicted_labels == 1:
  print("Neutral Sentiment Detected")
else:
  print("Positive Sentiment Detected")



"""**Chat History**"""

fc = "/content/WhatsApp Chat.txt"

def extract_english_words(text):
    english_words = []
    # Regular expression pattern to match English words
    english_pattern = re.compile(r'\b[a-zA-Z]+\b')
    # Find all English words in the text
    english_words = english_pattern.findall(text)
    return english_words

def process_whatsapp_file(file_path):
    user_messages = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as file:
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

    return user_messages



# Path to your WhatsApp chat history file
whatsapp_file_path = fc

# Process the WhatsApp chat history file
user_messages = process_whatsapp_file(whatsapp_file_path)

selected_user = input("Select a user name: " + str(list(user_messages.keys())))
messages = user_messages[selected_user]

text_dataset = tf.data.Dataset.from_tensor_slices(messages)
print(text_dataset)
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_words,
    output_mode='int',
    output_sequence_length=maxlen,
)

print(user_messages)
# Adapt the layer to the new messages
vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices([" ".join(messages)]).batch(1))

# Convert the messages into numerical vectors
message_vectors = vectorize_layer([" ".join(messages)])

# Add a batch dimension to the message vectors
message_vectors = tf.expand_dims(message_vectors, axis=0)
message_vectors = tf.reshape(message_vectors, (maxlen,))

print(message_vectors,"\n")

try:
    # Use the model to predict the sentiment of the messages
    predictions = loaded_model.predict(messages)

    # Get the predicted sentiment labels
    predicted_labels = np.argmax(predictions, axis=1)
    print(f"Predicted sentiment label: {predicted_labels[0]}")

    if predicted_labels[0] == 0:
      print("Predicted sentiment label: Negative Sentiment Detected !!!")
    elif predicted_labels[0] == 1:
      print("Predicted sentiment label: Neutral Sentiment Detected")
    else:
      print("Predicted sentiment label: Positive Sentiment Detected")

except Exception as e:
    print(f"Error occurred: {e}")
    # Use the model to predict the sentiment of the messages
    predictions = loaded_model.predict(text_dataset)

    # Get the predicted sentiment labels
    predicted_labels = np.argmax(predictions, axis=1)
    print(f"Predicted sentiment label: {predicted_labels}")

    if predicted_labels[0] == 0:
      print("Predicted sentiment label: Negative Sentiment Detected !!!")
    elif predicted_labels[0] == 1:
      print("Predicted sentiment label: Neutral Sentiment Detected")
    else:
      print("Predicted sentiment label: Positive Sentiment Detected")