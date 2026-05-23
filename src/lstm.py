import os
import numpy as np
import tensorflow as tf
import re
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
 
from keras.optimizers import RMSprop
 
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import random
import sys

with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()
words = re.findall(r'\b[а-яёa-z0-9]+\b|[.,!?;:]', text)
vocabulary = sorted(list(set(words)))
 
word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))

max_length = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

model = Sequential()
model.add(tf.keras.Input(shape=(max_length, len(vocabulary))))
model.add(LSTM(128))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.clip(preds, 1e-8, 1.0)

    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(X, y, batch_size = 128, epochs = 50)

def words_to_text(words_list):
    result = ''

    for word in words_list:
        if word in '.,!?;:':
            result += word
        else:
            if result:
                result += ' '
            result += word

    return result

def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)

    sentence = words[start_index: start_index + max_length]
    generated = sentence.copy()

    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))

        for t, word in enumerate(sentence):
            x_pred[0, t, word_to_indices[word]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = indices_to_word[next_index]

        generated.append(next_word)
        sentence = sentence[1:] + [next_word]

    return words_to_text(generated)

os.makedirs('result', exist_ok=True)

generated_text = generate_text(1500, 0.5)

with open('result/gen.txt', 'w', encoding='utf-8') as file:
    file.write(generated_text)

print(generated_text)
