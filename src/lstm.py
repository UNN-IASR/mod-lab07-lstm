import numpy as np
import tensorflow as tf
import random
import sys
from collections import Counter
import os
import re
from keras.optimizers import RMSprop

with open('input.txt', 'r', encoding="utf-8") as file:
    text = file.read().lower()

words = text.split()
vocabulary = sorted(list(set(words)))
word_to_indices = {word: i for i, word in enumerate(vocabulary)}
indices_to_word = {i: word for i, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)

max_length = 10
steps = 1

sentences = []
next_words = []
for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i:i + max_length])
    next_words.append(words[i + max_length])

X = np.array([[word_to_indices[word] for word in sentence] for sentence in sentences], dtype=np.int32)
y = np.array([word_to_indices[word] for word in next_words], dtype=np.int32)

print(f"Обучение на {len(X)} примерах, словарь: {vocab_size} слов")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

optimizer = RMSprop(learning_rate = 0.01)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('weights.h5', save_best_only=True, monitor='loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
]

batch_size = 128
epochs = 50

print("Начинается обучение...")

model.fit(
    X, y,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
    validation_split=0.1
)

def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(length, diversity):
    start_index = random.randint(0, len(X) - 1)
    generated = sentences[start_index].copy()
    generated_words = generated.copy()

    for _ in range(length):
        x_pred = np.array([word_to_indices[word] for word in generated_words]).reshape(1, -1)
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = indices_to_word[next_index]
        generated.append(next_word)
        generated_words = generated_words[1:] + [next_word]

    return ' '.join(generated)

print("Запущена генерация текста")
generated_text = generate_text(1500, 0.5)
with open('../result/gen.txt', 'w', encoding='utf-8') as file:
    file.write(generated_text)

print("Генерация завершена. Текст сохранен в gen.txt")
