# Copyright 2026 UNN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
import random
import os
import re

input_file = 'src/input.txt'
result_file = 'result/gen.txt'

print("Загрузка текста...")
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read().lower()

words = re.findall(r'\b\w+\b|[.,!?;]', text)

vocabulary = sorted(list(set(words)))
print(f"Всего слов в тексте: {len(words)}")
print(f"Уникальных слов (размер словаря): {len(vocabulary)}")

if len(words) < 10000:
    print("ВНИМАНИЕ: По ТЗ во входном файле должно быть не менее 10 000 слов!")

word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))

max_length = 5 
steps = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

print(f"Сформировано тренировочных цепочек: {len(sentences)}")

print("Векторизация данных (это может занять немного времени)...")
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

print("Создание модели...")
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
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

print("Начало обучения...")
model.fit(X, y, batch_size=128, epochs=50)

def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)
    sentence = words[start_index: start_index + max_length]
    
    generated = list(sentence)
    
    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_to_indices[word]] = 1.
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = indices_to_word[next_index]
        
        generated.append(next_word)
        sentence = sentence[1:] + [next_word]
        
    result_text = " ".join(generated)
    result_text = re.sub(r'\s+([.,!?;])', r'\1', result_text)
    return result_text

print("\nГенерация результирующего текста...")
final_text = generate_text(1000, 0.5)

os.makedirs(os.path.dirname(result_file), exist_ok=True)
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(final_text)

print(f"Успех! Сгенерированный текст сохранен в {result_file}")
