import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, CuDNNLSTM
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Configure TensorFlow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class LSTMModel:
    def __init__(self, text_data, seq_length=40):
        """Initialize GPU-optimized LSTM model"""
        self.chars = sorted(list(set(text_data)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.seq_length = seq_length
        self.vocab_size = len(self.chars)
        self.model = self._build_model()
        
    def _build_model(self):
        """Build GPU-optimized LSTM model with CuDNNLSTM"""
        strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
        
        if strategy:
            with strategy.scope():
                model = Sequential([
                    CuDNNLSTM(256, input_shape=(self.seq_length, self.vocab_size), 
                             return_sequences=True),
                    Dropout(0.2),
                    CuDNNLSTM(256),
                    Dropout(0.2),
                    Dense(self.vocab_size, activation='softmax')
                ])
                model.compile(loss='categorical_crossentropy', 
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        else:
            model = Sequential([
                CuDNNLSTM(256, input_shape=(self.seq_length, self.vocab_size), 
                         return_sequences=True),
                Dropout(0.2),
                CuDNNLSTM(256),
                Dropout(0.2),
                Dense(self.vocab_size, activation='softmax')
            ])
            model.compile(loss='categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        
        return model
    
    def prepare_data(self, text):
        """Prepare data with GPU-friendly formats"""
        X = []
        y = []
        
        for i in range(0, len(text) - self.seq_length, 1):
            seq_in = text[i:i + self.seq_length]
            seq_out = text[i + self.seq_length]
            X.append([self.char_to_idx[char] for char in seq_in])
            y.append(self.char_to_idx[seq_out])
            
        X = np.reshape(X, (len(X), self.seq_length, 1))
        X = X / float(self.vocab_size)
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)
        
        # Convert to GPU-friendly tensors
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        return X, y
    
    def train(self, X, y, epochs=20, batch_size=128, model_path='lstm_model.h5'):
        """Train with GPU acceleration"""
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        # Use larger batch size if GPU memory allows
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
        """Generate text with GPU support"""
        generated = seed
        seed = seed.lower()
        
        for _ in range(length):
            x = np.zeros((1, self.seq_length, self.vocab_size))
            for t, char in enumerate(seed):
                x[0, t, self.char_to_idx[char]] = 1
                
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            
            preds = self.model.predict(x, verbose=0)[0]
            next_idx = self._sample(preds, temperature)
            next_char = self.idx_to_char[next_idx]
            
            generated += next_char
            seed = seed[1:] + next_char
            
        return generated
    
    def _sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return np.random.choice(len(preds), p=preds)