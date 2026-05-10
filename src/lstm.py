import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

import random
import os

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


with open('input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

words = [word for line in lines for word in line.split()]


vocabulary_of_words = sorted(set(words))
word_to_index = {w: i for i, w in enumerate(vocabulary_of_words)}
index_to_word = {i: w for i, w in enumerate(vocabulary_of_words)}
vocab_size = len(vocabulary_of_words)

print(f"Размер словаря: {vocab_size}")
print(f"Слов в тексте: {len(words)}")


max_len = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_len, steps):
    sentences.append(words[i: i + max_len])
    next_words.append(words[i + max_len])


X = np.array([[word_to_index[w] for w in s] for s in sentences], dtype=np.int32)
y = np.array([word_to_index[w] for w in next_words], dtype=np.int32)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Память X: {X.nbytes / 1024 / 1024:.1f} МБ")


EMBED_DIM = 64

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=max_len))
model.add(LSTM(128))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
model.fit(X, y, batch_size=128, epochs=70, verbose=1, callbacks=[reduce_lr])


def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # +1e-8 защита от log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))


def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_len - 1)
    sentence = words[start_index: start_index + max_len]
    generated_text = " ".join(sentence)

    for i in range(length):
        x_pred = np.array([[word_to_index[w] for w in sentence]], dtype=np.int32)
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = index_to_word[next_index]

        generated_text += " " + next_word
        sentence = sentence[1:] + [next_word]

    return generated_text


os.makedirs('result', exist_ok=True)
result = generate_text(1000, 0.2)
print(result)

with open('../result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(result)