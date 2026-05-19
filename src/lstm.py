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

with open('./src/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

words = text.split()
print(f'\n\nWords amount: {len(words)}\n\n')
if len(words) < 10000:
    print('Few words found in text. Please add more words.')

# Получаем алфавит
vocabulary = sorted(list(set(words)))
 
# Создаем словари, содержащие индекс символа и связываем их с символами
word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))

# Разбиваем текст на цепочки длины max_length
# Каждый временной шаг будет загружать очередную цепочку в сеть
max_length = 10
steps = 1
sentences = []
next_words = []

# Создаем список цепочек и список символов, которые следуют за цепочками
for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])


    # Создаем тренировочный набор
# Создаем битовые вектора для входных значений
# (Номер_цепочки-Номер_символа_в цепочке-Код_символа)
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
# Выходные данные
# (Номер_цепочки-Код_символа)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1

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
    start_index = random.randint(0, len(words) - max_length - 1)
    generated = []
    sentence = words[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_to_indices[word]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_word = indices_to_word[next_index]
 
            generated.append(next_word)
            sentence = sentence[1:] + [next_word]
    return ' '.join(generated)  # собираем слова обратно в строку

result = generate_text(1500, 0.7)

with open('./result/gen.txt', 'w', encoding='utf-8') as f:
    f.write(result)

print(result)