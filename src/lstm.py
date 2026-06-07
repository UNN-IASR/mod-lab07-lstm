import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
import random
import re

# 1. Загрузка текста
with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# 2. Токенизация по словам
tokens = re.findall(r"[А-Яа-яЁёA-Za-z']+|[^\sА-Яа-яЁёA-Za-z']", raw_text)
tokens = [t for t in tokens if t.strip()]

print(f"Всего токенов: {len(tokens)}")

# 3. Словарь
vocabulary = sorted(list(set(tokens)))
vocab_size = len(vocabulary)
print(f"Размер словаря: {vocab_size}")

word_to_index = {w: i for i, w in enumerate(vocabulary)}
index_to_word = {i: w for i, w in enumerate(vocabulary)}

# 4. Цепочки слов (скользящее окно)
max_length = 10
step = 1
sequences = []
next_words = []

for i in range(0, len(tokens) - max_length, step):
    sequences.append(tokens[i: i + max_length])
    next_words.append(tokens[i + max_length])

print(f"Обучающих примеров: {len(sequences)}")

# 5. One-hot векторизация
X = np.zeros((len(sequences), max_length, vocab_size), dtype=np.float32)
y = np.zeros((len(sequences), vocab_size), dtype=np.float32)

for i, seq in enumerate(sequences):
    for t, word in enumerate(seq):
        X[i, t, word_to_index[word]] = 1.0
    y[i, word_to_index[next_words[i]]] = 1.0

print(f"Форма X: {X.shape}, форма y: {y.shape}")

# 6. Построение LSTM-модели
model = Sequential([
    LSTM(256, input_shape=(max_length, vocab_size)),
    Dense(vocab_size),
    Activation('softmax'),
])

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

# 7. Функция температуры (sampling)
def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 8. Обучение
print("\nНачало обучения...")
model.fit(X, y, batch_size=128, epochs=50, verbose=1)
print("Обучение завершено.")

# 9. Функция генерации текста
def generate_text(num_words, diversity):
    start_index = random.randint(0, len(tokens) - max_length - 1)
    seed_seq = tokens[start_index: start_index + max_length]
    generated_tokens = list(seed_seq)
    current_seq = list(seed_seq)

    for _ in range(num_words):
        x_pred = np.zeros((1, max_length, vocab_size), dtype=np.float32)
        for t, word in enumerate(current_seq):
            x_pred[0, t, word_to_index[word]] = 1.0

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = index_to_word[next_index]

        generated_tokens.append(next_word)
        current_seq = current_seq[1:] + [next_word]

    # Восстановление текста
    text_out = ''
    for i, tok in enumerate(generated_tokens):
        if re.match(r"[А-Яа-яЁёA-Za-z']+", tok):
            if i > 0 and re.match(r"[А-Яа-яЁёA-Za-z']+",
                                   generated_tokens[i-1]):
                text_out += ' '
        text_out += tok
    return text_out

# 10. Генерация и сохранение
print("\nГенерация текста (температура=0.5)...")
result = generate_text(num_words=1100, diversity=0.5)

print("\n" + "="*60)
print(result)
print("="*60)

with open('../result/gen.txt', 'w', encoding='utf-8') as out:
    out.write(result)

print("\nРезультат сохранён в result/gen.txt")
