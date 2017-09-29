# Generating text from Alice in Wonderland

# Import libraries
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from keras import backend as K

K.set_learning_phase(1)

# Get raw data and make lowercase
raw_text = open('shakespeare.txt').read()
raw_text = raw_text.lower()

# Assemble a dict of all unique characters to a corresponding number
chars = sorted(list(set(raw_text)))
char_mapping = dict((c,i) for i, c in enumerate(chars))
joblib.dump(char_mapping, 'shakespeare/char_mapping.sav')

# Get number of unique characters, and number of total characters
n_chars = len(raw_text)
n_vocab = len(chars)

# Create One Hot Encoder, and fit on char mapping
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit(np.reshape([char_mapping[char] for char in chars], (len(chars), 1)))
joblib.dump(ohe, 'shakespeare/ohe.sav')

# Encode raw_text into One-Hot-Encoding
encoded_text = [char_mapping[char] for char in raw_text]
encoded_text = np.reshape(encoded_text, (len(raw_text), 1))
encoded_text = ohe.transform(encoded_text).toarray()
del raw_text

# Create sequences of 100 and their corresponding output
seq_length = 100
x_train = []
y_train = []
for i in range(n_chars - seq_length):
    sequence = encoded_text[i:i+seq_length,:-1]
    x_train.append(sequence)
    y_train.append(encoded_text[i+seq_length])

# Create np arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Assemble RNN
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
model = Sequential()
model.add(LSTM(units=256, return_sequences=True, input_shape=(seq_length, n_vocab-1)))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Fit and save RNN
model.fit(x_train, y_train, epochs=50, batch_size=64)
model.save('shakespeare/model.h5')

# Import libraries
from keras.models import load_model
from sklearn.externals import joblib
import numpy as np
from keras import backend as K

# Set learning phase (disable dropout)
K.set_learning_phase(0)

# Load model, char mapping, encoder, and raw text from disk
model = load_model('shakespeare/model_1-2.h5')
char_mapping = joblib.load('shakespeare/char_mapping.sav')
ohe = joblib.load('shakespeare/ohe.sav')
raw_text = open('shakespeare.txt').read().lower()
n_vocab = len(char_mapping)

# Create an inverse map of char mapping
reverse_mapping = {}
for x in char_mapping:
    reverse_mapping[char_mapping[x]] = x

def select(output):
    # Probabilistically determine output based on softmax output
    from random import uniform
    letter = uniform(0., 1.)
    added = 0
    for i in range(len(output)):
        if added + output[i] >= letter:
            return i
        else:
            added += output[i]
            
def generate_text(seed, steps):
    seed = seed.lower()
    print(seed, end='')
    last = list(seed)
    for i in range(steps):
        # Get input sequence and encode it
        input_seq = [char_mapping[char] for char in last]
        input_seq = np.reshape(input_seq, (len(input_seq), 1))
        input_seq = ohe.transform(input_seq).toarray()[:,:-1]
        input_seq = np.expand_dims(input_seq, axis=0)
        
        # Predict output character and add to input sequence
        y = model.predict(input_seq).flatten()
        y = select(y)
        del last[0]
        last.append(reverse_mapping[y])
        print(reverse_mapping[y], end='')

# Select a random 100-character seed
import numpy as np
start = np.random.randint(0, len(raw_text)-100)
seed = raw_text[start:start+100]
print('Seed: ')
print(seed)
print('-----------------------------')
            
# Predict 1000 characters
generate_text(seed, 1000)

def pad(text, length):
    if len(text) > length:
        return text
    n_padding = length-len(text)-1
    return '{}\n{}'.format(''.join([' ']*n_padding), text)