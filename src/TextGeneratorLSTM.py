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

#текст
with open('input.txt','r',encoding='utf-8') as file: text = file.read()
#слова
words = text.split()
print(f"Загружено слов: {len(words)}")
#словарь
vocabulary = sorted(set(words))
word_to_index = {word: i for i, word in enumerate(vocabulary)}
index_to_word = {i: word for i, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)
print(f"Размер словаря: {vocab_size}")

maxlen = 10   # Длина контекста (сколько слов подаётся на вход)
step = 1     # Шаг смещения окна
sentences = []      # Входные последовательности (индексы)
next_words = []     # Целевые слова (индексы)

for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i: i + maxlen])
    next_words.append(words[i + maxlen])

print(f"Количество обучающих примеров: {len(sentences)}")

# Векторизация: превращаем слова в one-hot представления
X = np.zeros((len(sentences), maxlen, vocab_size), dtype=np.bool_)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool_)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_index[word]] = 1
    y[i, word_to_index[next_words[i]]] = 1

# 3. Построение LSTM-модели
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# 4. Вспомогательная функция для семплирования с температурой
def sample_index(preds, temperature=1.0):
    """Возвращает индекс слова с учётом температуры."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature   # +1e-8 для избежания log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 5. Генерация текста
def generate_text(length, diversity):
    """Генерирует текст заданной длины (в словах)."""
    # Случайное начало из исходного текста
    start_index = random.randint(0, len(words) - maxlen - 1)
    current_seq = words[start_index: start_index + maxlen]
    generated = current_seq[:]   # список слов

    for i in range(length):
        x_pred = np.zeros((1, maxlen, vocab_size))
        for t, word in enumerate(current_seq):
            if word in word_to_index:   # слово из обучающего словаря
                x_pred[0, t, word_to_index[word]] = 1
            # иначе остаётся нулевой вектор (слово не из словаря)

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = index_to_word[next_index]
        generated.append(next_word)

        # Сдвигаем окно: убираем первое слово, добавляем предсказанное в конец
        current_seq = current_seq[1:] + [next_word]

    # Объединяем список слов в строку через пробел
    return ' '.join(generated)

# 6. Обучение модели
print("Начинаем обучение...")
model.fit(X, y, batch_size=128, epochs=20)   # можешь увеличить число эпох при необходимости
print("Обучение завершено!")

# 7. Генерация и сохранение результата
print("Генерируем текст...")
generated_text = generate_text(1000, 0.5)   # 1000 слов, температура 0.5 (умеренная креативность)

# Сохраняем результат в папку result (создадим её автоматически)
import os
os.makedirs('result', exist_ok=True)
with open('result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

print("Текст сохранён в result/gen.txt")