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
start = np.random.randint(0, len(raw_text)-100)
seed = raw_text[start:start+100]
print('Seed: ')
print(seed)
print('-----------------------------')
            
# Predict 1000 characters
generate_text(seed, 1000)