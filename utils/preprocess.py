import re
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Enhanced text cleaning with stopword removal"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def create_sequences(text, seq_length=50):
    """Create training sequences for LSTM"""
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length, 1):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])
    
    return sequences, next_chars

def prepare_dataset(text, tokenizer, seq_length=50, batch_size=32):
    """Create optimized TF Dataset pipeline"""
    sequences, next_chars = create_sequences(text, seq_length)
    
    # Convert to numerical tokens
    X = tokenizer.texts_to_sequences(sequences)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=seq_length)
    
    y = tf.keras.utils.to_categorical(
        [tokenizer.char_to_idx[c] for c in next_chars],
        num_classes=len(tokenizer.char_to_idx)
    )
    
    # Create optimized dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset