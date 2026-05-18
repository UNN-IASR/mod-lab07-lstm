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

with open('input.txt', 'r') as file:
    text = file.read()

# Получаем алфавит
vocabulary = sorted(list(set(text)))
 
# Создаем словари, содержащие индекс символа и связываем их с символами
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

# Разбиваем текст на цепочки длины max_length
# Каждый временной шаг будет загружать очередную цепочку в сеть
max_length = 10
steps = 1
sentences = []
next_chars = []

# Создаем список цепочек и список символов, которые следуют за цепочками
for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])

# Создаем тренировочный набор
# Создаем битовые вектора для входных значений
# (Номер_цепочки-Номер_символа_в цепочке-Код_символа)
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
# Выходные данные
# (Номер_цепочки-Код_символа)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

# Строим LSTM-сеть
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

# Обучение LSTM модели
model.fit(X, y, batch_size = 128, epochs = 50)

def generate_text(length, diversity):
    # Случайное начало
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]
 
            generated += next_char
            sentence = sentence[1:] + next_char
    return generated

gen = generate_text(1500, 0.2)
print(gen)

import os
os.makedirs('result', exist_ok=True)
with open('result/gen.txt', 'w', encoding='utf-8') as file:
    file.write(gen)
