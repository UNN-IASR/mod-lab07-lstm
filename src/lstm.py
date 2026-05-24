import numpy as np
import random
import re
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau

# ----------------- Загрузка текста -----------------
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(f"Общая длина текста: {len(text)} символов")

# ----------------- Токенизация -----------------
# Разбиваем текст на слова и знаки препинания как отдельные токены
# \w+ — слово (буквы/цифры), [^\w\s] — одиночный знак препинания
tokens = re.findall(r'\w+|[^\w\s]', text)
print(f"Общее количество токенов: {len(tokens)}")

# ----------------- Построение словаря -----------------
# Ограничим словарь top_n наиболее частыми токенами, чтобы модель оставалась обучаемой
top_n = 5000  # можно увеличить, если позволяет память
token_freq = {}
for t in tokens:
    token_freq[t] = token_freq.get(t, 0) + 1
sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
vocabulary = ['<UNK>'] + [t for t, _ in sorted_tokens[:top_n-1]]
print(f"Размер словаря: {len(vocabulary)}")

token_to_index = {t: i for i, t in enumerate(vocabulary)}
index_to_token = {i: t for t, i in token_to_index.items()}

# Преобразуем все токены в индексы (неизвестные — в 0, <UNK>)
token_indices = [token_to_index.get(t, 0) for t in tokens]

# ----------------- Создание последовательностей -----------------
max_length = 10  # длина цепочки токенов
step = 1         # шаг скользящего окна
sentences = []
next_tokens = []

for i in range(0, len(token_indices) - max_length, step):
    sentences.append(token_indices[i: i + max_length])
    next_tokens.append(token_indices[i + max_length])

print(f"Обучающих последовательностей: {len(sentences)}")

# ----------------- Подготовка данных -----------------
# Для Embedding на вход подаём целочисленные индексы размером (samples, max_length)
X = np.array(sentences, dtype=np.int32)

# Выходной вектор — one-hot по размеру словаря
y = np.zeros((len(sentences), len(vocabulary)), dtype=np.float32)
for i, next_tok in enumerate(next_tokens):
    y[i, next_tok] = 1.0

print(f"Форма X: {X.shape}, форма y: {y.shape}")

# ----------------- Построение модели -----------------
model = Sequential()
# Embedding вместо one-hot: кажому токену сопоставляется dense-вектор
model.add(Embedding(input_dim=len(vocabulary), output_dim=128, input_length=max_length))
model.add(LSTM(256))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

# ----------------- Температурная функция -----------------
def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    # применяем логарифмическое масштабирование
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ----------------- Колбэки -----------------
# Сохраняем лучшую модель по loss
checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)

# ----------------- Обучение -----------------
model.fit(X, y, batch_size=256, epochs=40, callbacks=[checkpoint, reduce_lr])

# ----------------- Генерация текста -----------------
def generate_text(seed_length, gen_length, diversity):
    """
    Генерация текста, начиная со случайного фрагмента из исходных токенов.
    seed_length — длина начальной цепочки токенов (обычно = max_length)
    gen_length — сколько токенов сгенерировать
    diversity — температурный коэффициент
    """
    # Выбираем случайный стартовый индекс
    start_index = random.randint(0, len(token_indices) - seed_length - 1)
    generated_indices = token_indices[start_index: start_index + seed_length]
    generated_tokens = [index_to_token[idx] for idx in generated_indices]

    for i in range(gen_length):
        # подготавливаем входную последовательность
        x_pred = np.array([generated_indices[-seed_length:]], dtype=np.int32)
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        generated_indices.append(next_index)
        generated_tokens.append(index_to_token[next_index])

    # Собираем итоговую строку: слова разделяем пробелом, знаки препинания приклеиваем к предыдущему слову
    result = generated_tokens[0]
    for t in generated_tokens[1:]:
        if re.match(r'\w+', t):  # слово
            result += ' ' + t
        else:                    # знак препинания
            result += t
    return result

# ----------------- Сохранение результата -----------------
print("Генерация текста...")
generated_text = generate_text(seed_length=max_length, gen_length=2000, diversity=0.5)
with open('../result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
print("Результат сохранён в result/gen.txt")