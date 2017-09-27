# Generating text from Alice in Wonderland

# Import libraries
import numpy as np
from keras.utils import np_utils
from sklearn.externals import joblib

# Get raw data and make lowercase
raw_text = open('wonderland.txt').read()
raw_text = raw_text.lower()

# Assemble a dict of all unique characters to a corresponding number
chars = sorted(list(set(raw_text)))
char_mapping = dict((c,i) for i, c in enumerate(chars))
joblib.dump(char_mapping, 'char_mapping.sav')

# Get number of unique characters, and number of total characters
n_chars = len(raw_text)
n_vocab = len(chars)

# Create sequences of 100 and their corresponding output
x_train = []
y_train = []
seq_length = 100
for i in range(n_chars-seq_length):
    x_train.append([char_mapping[char] for char in raw_text[i:i+seq_length]])
    y_train.append(char_mapping[raw_text[i+seq_length]])
joblib.dump(x_train, 'x_train.sav')

# Reshape data and normalize
x = np.reshape(x_train, (len(x_train), seq_length, 1))
x = x/float(n_vocab)

# One-Hot-Encode y data
y = np_utils.to_categorical(y_train)

# Assemble RNN
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit and save RNN
model.fit(x, y, epochs=50, batch_size=64)
model.save('model.h5')

# Load char mapping and model from disk
from keras.models import load_model
from sklearn.externals import joblib
model = load_model('model.h5')
char_mapping = joblib.load('char_mapping.sav')
x_train = joblib.load('x_train.sav')

# Create an inverse map of char mapping
reverse_mapping = {}
for x in char_mapping:
    reverse_mapping[char_mapping[x]] = x

# Select a random seed
import numpy as np
start = np.random.randint(0, len(x_train)-1)
seed = x_train[start]
print('Seed: ')
print(''.join([reverse_mapping[x] for x in seed]))
print('-----------------------------')
print(''.join([reverse_mapping[x] for x in seed]), end='')

last = seed
for i in range(1000):
    x = np.reshape(last, (1, len(last), 1))
    x = x/float(len(char_mapping))
    y = np.argmax(model.predict(x).flatten())
    del last[0]
    last.append(y)
    print(reverse_mapping[y], end='')