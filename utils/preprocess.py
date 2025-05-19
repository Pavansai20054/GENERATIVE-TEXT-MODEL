import re
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize

def clean_text_gpu(text):
    """Text cleaning that produces GPU-friendly outputs"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def batch_preprocess(texts, batch_size=1024):
    """Batch processing for GPU efficiency"""
    cleaned = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        cleaned.extend([clean_text_gpu(t) for t in batch])
    return cleaned

def create_gpu_dataset(texts, tokenizer, seq_length, batch_size=32):
    """Create TensorFlow Dataset for GPU training"""
    tokens = tokenizer.texts_to_sequences(texts)
    dataset = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=seq_length)
    
    # Create TF Dataset
    ds = tf.data.Dataset.from_tensor_slices(dataset)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return ds