# Copyright 2021 GHA Test Team

import os
import random
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


INPUT_FILE = "src/input.txt"
OUTPUT_FILE = "result/gen.txt"

SEQ_LENGTH = 5
GENERATE_WORDS = 1200
EPOCHS = 20
BATCH_SIZE = 128


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def sample_index(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return int(np.argmax(probabilities))


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        text = file.read().lower()

    words = tokenize(text)

    if len(words) <= SEQ_LENGTH:
        raise ValueError("Input text is too short for training.")

    vocabulary = sorted(set(words))
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    index_to_word = {index: word for word, index in word_to_index.items()}

    sentences = []
    next_words = []

    for i in range(0, len(words) - SEQ_LENGTH):
        sentences.append(words[i:i + SEQ_LENGTH])
        next_words.append(words[i + SEQ_LENGTH])

    x = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
    y = np.zeros((len(sentences),), dtype=np.int32)

    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x[i, t] = word_to_index[word]
        y[i] = word_to_index[next_words[i]]

    model = Sequential()
    model.add(Embedding(len(vocabulary), 128, input_length=SEQ_LENGTH))
    model.add(LSTM(128))
    model.add(Dense(len(vocabulary), activation="softmax"))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer
    )

    model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

    start_index = random.randint(0, len(words) - SEQ_LENGTH - 1)
    sentence = words[start_index:start_index + SEQ_LENGTH]
    generated = list(sentence)

    for _ in range(GENERATE_WORDS):
        x_pred = np.zeros((1, SEQ_LENGTH), dtype=np.int32)

        for t, word in enumerate(sentence):
            x_pred[0, t] = word_to_index.get(word, 0)

        predictions = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(predictions, temperature=0.7)
        next_word = index_to_word[next_index]

        generated.append(next_word)
        sentence = sentence[1:] + [next_word]

    os.makedirs("result", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(detokenize(generated))

    print(f"Generated text saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
