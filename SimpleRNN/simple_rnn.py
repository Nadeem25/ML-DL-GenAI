import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


# Load the imdb dataset
max_features = 10000 # Vocabulary size
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
# print(f'Testing data shape:{X_test.shape}, Training lables shape: {y_test.shape}')

# Inspect sample review and its label
sample_review = X_train[0]
sample_label = y_train[0]

# print(f"Sample review (as integer): {sample_review}")
# print(f"Sample label: {sample_label}")

# Mapping of words index back to words
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
decode_review = ' '.join([reverse_word_index.get(i-3, '?')  for i in sample_review])
print(f"Decode Sample Review number to word: {decode_review}")
print(f"-----------------------------------------------------------------------")

# Use sequence for pre padding
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
#print(f"X Train : {X_train[0]}")

# Train Simple RNN
model = Sequential()
dim = 128
model.add(Embedding(max_features,dim,input_length=max_length)) # Embedding Layer
model.add(SimpleRNN(dim,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(f"Model Summary: {model.summary()}")


#Compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


## Create an instance of EarlyStopping callback
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)


# Train the model with early stopping
history = model.fit(
    X_train, y_train, epochs=10, batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)

# Save model file
model.save('simple_rnn_imdb.h5')