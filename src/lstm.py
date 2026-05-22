import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import random
import sys
import re

with open('src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

words = re.findall(r'\w+|[.,!?;\-]', text)

vocabulary = sorted(list(set(words)))
vocab_size = len(vocabulary)

word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))

max_length = 7 
steps = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

X = np.zeros((len(sentences), max_length, vocab_size), dtype=bool)
y = np.zeros((len(sentences), vocab_size), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(max_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(X, y, batch_size=128, epochs=30)

def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)
    generated_words = []
    sentence = words[start_index: start_index + max_length]
    generated_words.extend(sentence)
    
    for i in range(length):
        x_pred = np.zeros((1, max_length, vocab_size))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_to_indices[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = indices_to_word[next_index]

        generated_words.append(next_word)
        sentence = sentence[1:] + [next_word]
        
    result_text = ""
    for word in generated_words:
        if word in ".,!?;-" or not result_text:
            result_text += word
        else:
            result_text += " " + word
            
    return result_text

generated_output = generate_text(1300, diversity=0.5)

with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_output)
print("\nТекст сохранен в result/gen.txt")