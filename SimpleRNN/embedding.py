import numpy as np
from tensorflow.keras.preprocessing.text import one_hot

sent=['the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good']

# Define the vocabulary size
voc_size = 10000

# One Hot Representation
one_hot_repr = [one_hot(words, voc_size) for words in sent]
print(one_hot_repr)

# One embedding representation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences

# Set the sentance length because most of the senetances length are diffrent
sent_length = 8
embedded_docs = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
print("Sentance with equal length:", embedded_docs)

# Feature Representatin
dim=10
model = Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile('adam', 'mse')
model.summary()
print(f"Vector representation: {model.predict(embedded_docs)}")

