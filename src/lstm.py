import numpy as np
import tensorflow as tf
import random
import os

# Используем правильные импорты из tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 1. Загрузка текста
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Загружено символов: {len(text)}")
print(f"Загружено слов: {len(text.split())}")

# 2. Токенизация на уровне слов
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

print(f"Уникальных слов: {total_words}")

# Преобразуем текст в последовательность индексов слов
input_sequences = []
for sentence in text.split("\n"):
    if sentence.strip():
        token_list = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

print(f"Всего n-грамм последовательностей: {len(input_sequences)}")

if len(input_sequences) == 0:
    print("Ошибка: Не создано ни одной последовательности. Проверьте файл input.txt")
    exit(1)

# 3. Паддинг и подготовка X, y
max_sequence_len = max([len(seq) for seq in input_sequences])
print(f"Максимальная длина последовательности: {max_sequence_len}")

# Паддинг
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# X — все слова кроме последнего, y — последнее слово
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encoding для y
y = to_categorical(y, num_classes=total_words)

print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")

# 4. Построение LSTM модели
model = Sequential()
model.add(LSTM(128, input_shape=(max_sequence_len - 1, 1), return_sequences=False))
model.add(Dense(128, activation="relu"))
model.add(Dense(total_words, activation="softmax"))

# Reshape X для LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

model.summary()


# 5. Функция температуры
def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 6. Обучение модели
epochs = 20
batch_size = 32

print("\nНачинаем обучение...")
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)

# Сохраняем модель
model.save("word_lstm_model.h5")
print("Модель сохранена")


# 7. Генерация текста
def generate_text(seed_text, next_words, temperature=0.7):
    generated = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )
        token_list = token_list.reshape((1, max_sequence_len - 1, 1))

        preds = model.predict(token_list, verbose=0)[0]
        next_index = sample_index(preds, temperature)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break
        if output_word:
            generated += " " + output_word
        else:
            break
    return generated


# 8. Генерация итогового текста
# Выбираем случайный начальный фрагмент
sentences = [s for s in text.split(". ") if len(s.split()) >= 3]
if sentences:
    seed_phrase = random.choice(sentences)
    seed_phrase = " ".join(seed_phrase.split()[:3])
else:
    seed_phrase = "Linux операционная система"

print(f"\nНачальная фраза: {seed_phrase}")

# Генерируем текст (1000 слов как требуется в задании)
generated_text = generate_text(seed_phrase, next_words=1000, temperature=0.7)

print("\n" + "=" * 50)
print("Сгенерированный текст (первые 2000 символов):")
print("=" * 50)
print(generated_text[:2000])

# Сохраняем в файл
os.makedirs("../result", exist_ok=True)
with open("../result/gen.txt", "w", encoding="utf-8") as f:
    f.write(generated_text)

print(f"\nГенерация завершена. Текст сохранён в result/gen.txt")
print(f"Общее количество слов в сгенерированном тексте: {len(generated_text.split())}")
