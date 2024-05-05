import streamlit as st
import re
from collections import defaultdict
from transformers import RobertaTokenizer
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Embedding


model_path = 'https://github.com/Karth-i/New_One/blob/main/model1.h5'

path_to_glove_file = "https://github.com/Karth-i/New_One/raw/main/glove.6B.50d.txt.zip"
glove_dir = "glove"
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))

# Define the custom layers
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_words,
    output_mode='int',
    output_sequence_length=maxlen
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

predictions = loaded_model.predict(["i fell sad"])

predicted_labels = np.argmax(predictions, axis=1)

if predicted_labels == 0:
  print("Negative Sentiment Dedected !!!")
elif predicted_labels == 1:
  print("Neutral Sentiment Dedected")
else:
  print("Positive Sentiment Dedected")
