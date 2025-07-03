import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Step 1: Load the Data and load the model
# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')
model.summary()

# Step 2: Helper Function

# Function to decode review
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review

# Step 3: Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Step 4: User input and prediction
example_review = "This movie was fantastic! The acting was great and the plot was thrilling."
sentiment, predict_score = predict_sentiment(example_review)

print(f"Example Review: {example_review}")
print(f"Sentiment: {sentiment}")
print(f"Prediction Score: {predict_score}")