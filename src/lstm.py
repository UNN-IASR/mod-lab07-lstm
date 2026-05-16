import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import random
import sys
import os

inputFile = 'src/input.txt'

with open(inputFile, 'r', encoding='utf-8') as f:
    text = f.read()

words = text.split()
print(f"Всего слов в тексте: {len(words)}")

vocabulary = sorted(set(words))
vocab_size = len(vocabulary)
print(f"Размер словаря (уникальных слов в тексте): {vocab_size}")

word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for i, word in enumerate(vocabulary)}

max_length = 10
step = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, step):
    sentences.append(words[i:i+max_length])
    next_words.append(words[i+max_length])

X = np.zeros((len(sentences), max_length), dtype=np.int32)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t] = word_to_idx[word]
    y[i, word_to_idx[next_words[i]]] = 1

embedding_dim = 128
lstm_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(lstm_units, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature   #1е-7 - для защиты от log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)
    context = words[start_index:start_index+max_length]
    generated = context.copy()
    for _ in range(length):
        x_pred = np.zeros((1, max_length), dtype=np.int32)
        for t, word in enumerate(context):
            x_pred[0, t] = word_to_idx[word]

        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_with_temperature(preds, diversity)
        next_word = idx_to_word[next_idx]

        generated.append(next_word)
        context = context[1:] + [next_word]

    return ' '.join(generated)
  
BATCH_SIZE = 128
EPOCHS = 20

model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

generated_text = generate_text(length=1500, diversity=0.5)
print("\nСгенерированный текст:\n")
print(generated_text)

os.makedirs('result', exist_ok=True)
with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
