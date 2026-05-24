import numpy as np
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM 
from keras.optimizers import RMSprop 
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import random
import sys
words = []
with open('input.txt', 'r', encoding='utf-8') as file:
    for line in file:
        next_line = line.split()
        for i, word in enumerate(next_line):
            if i == len(next_line) - 1:
                words.append(word + '\n')
            else:
                words.append(word)
vocabulary = sorted(list(set(words)))
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))
max_length = 10
steps = 1
sentences = []
next_chars = []
for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_chars.append(words[i + max_length])
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1
model = Sequential()
model.add(LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)
def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
model.fit(X, y, batch_size = 128, epochs = 50)
def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)
    generated = ''
    sentence = words[start_index: start_index + max_length]
    for i in range (0, len(sentence), 1):
        if i == 0:
            generated += sentence[i][0].upper() + sentence[i][1:] + ' '
        else:
            generated += sentence[i] + ' '
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]
            generated += next_char + ' '
            sentence = sentence[1:]
            sentence.append(next_char)
    return generated
def print_and_log(text, file_path="result.txt"):
    print(text)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
print_and_log(generate_text(1500, 0.2))
