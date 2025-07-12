# This project aims to develop a deep learning model for predicting the next word in a given sequence of words.
# The model is built using Long Short-Term Memory (LSTM) networks, which are well-suited for sequence prediction tasks. 
# The project includes the following steps:

# 1- Data Collection: We use the text of Shakespeare's "Hamlet" as our dataset. This rich, complex text provides a good challenge for our model.

# 2- Data Preprocessing: The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. 
# The sequences are then split into training and testing sets.

# 3- Model Building: An LSTM model is constructed with an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the probability of the next word.

# 4- Model Training: The model is trained using the prepared sequences, with early stopping implemented to prevent overfitting. 
# Early stopping monitors the validation loss and stops training when the loss stops improving.

# 5- Model Evaluation: The model is evaluated using a set of example sentences to test its ability to predict the next word accurately.

# 6- Deployment: A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.


# 1. Data Collection
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences # Use to make sure the sentance length will be same for training model.
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



# 2. Load the dataset
data = gutenberg.raw('shakespeare-hamlet.txt')
with open('hamlet_data.txt', 'w') as file:
    file.write(data)


# Load the Dataset
with open('hamlet_data.txt', 'r') as file:
    text = file.read().lower()

# Tokenize the text : creating the indexes for words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index)+1
# print(f"-------- Total Word: {total_words}")
# print(f"-------- Word Index: {tokenizer.word_index}")

# Create Input Sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequences = token_list[:i+1]
        input_sequences.append(n_gram_sequences)


# Pad Sequences : Make every sentence with equal length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


#Create predictors and lable
import tensorflow as tf
x = input_sequences[:, :-1] # All the words except last word
y = input_sequences[:, -1] # Only last word from all the word
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test,  = train_test_split(x, y, test_size=0.2)

# Define early stopping
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights= True)


# Train LSTM RNN
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150, return_sequences=False))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(total_words, activation='softmax'))
model.build(input_shape=(None, max_sequence_len))
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.summary()


# model = Sequential()
# model.add(Embedding(total_words, 100, input_length = max_sequence_len))
# model.add(LSTM(150, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(Dense(total_words, activation='softmax'))

# # Train the model
#history = model.fit(x_train, y_train,epochs=50, validation_data=(x_test, y_test), verbose=1)

# Save the model
model.save('next_word_lstm.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

input_text="Kin. Hee mad confession of"
print(f"Input text:{input_text}")
max_sequence_len=model.input_shape[1]+1
next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
print(f"Next Word Prediction:{next_word}")


