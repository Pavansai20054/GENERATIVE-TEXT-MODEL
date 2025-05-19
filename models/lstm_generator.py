import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import time
import nltk

# Configure TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class TextTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

class LSTMModel:
    def __init__(self, tokenizer, seq_length=40):
        """Initialize GPU-optimized LSTM model"""
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.model = self._build_model()

    def _build_model(self):
        """Build LSTM model"""
        strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None

        def build_layers():
            return Sequential([
                LSTM(256, input_shape=(self.seq_length, self.tokenizer.vocab_size), 
                     return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),
                Dropout(0.2),
                LSTM(256, activation='tanh', recurrent_activation='sigmoid'),
                Dropout(0.2),
                Dense(self.tokenizer.vocab_size, activation='softmax')
            ])

        if strategy:
            with strategy.scope():
                model = build_layers()
                model.compile(loss='categorical_crossentropy', 
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        else:
            model = build_layers()
            model.compile(loss='categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        return model

    def train(self, X, y, epochs=20, batch_size=128, model_path='lstm_model.h5'):
        """Train the model"""
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )

        if len(gpus) > 1:
            batch_size = batch_size * len(gpus)

        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint],
            verbose=1
        )

    def generate_text(self, seed, length=100, temperature=1.0):
        """Generate text with trained model"""
        generated = seed
        seed = seed.lower()

        for _ in range(length):
            x = np.zeros((1, self.seq_length, self.tokenizer.vocab_size))
            for t, char in enumerate(seed[-self.seq_length:]):
                if char in self.tokenizer.char_to_idx:
                    x[0, t, self.tokenizer.char_to_idx[char]] = 1

            preds = self.model.predict(x, verbose=0)[0]
            next_idx = self._sample(preds, temperature)
            next_char = self.tokenizer.idx_to_char[next_idx]

            generated += next_char
            seed += next_char

        return generated

    def _sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)