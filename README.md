***COMPANY:*** CODTECH IT SOLUTIONS  
***NAME:*** RANGDAL PAVANSAI  
***INTERN ID:*** C0DF200  
***DOMAIN:*** Artificial Intelligence Markup Language (AIML Internship)  
***DURATION:*** 4 WEEKS
***MENTOR:*** NEELA SANTHOSH


# 📝 Advanced Text Generation Toolkit 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

A **GPU-accelerated text generation framework** supporting both **Transformer-based (GPT-style)** and **LSTM** models. Designed for versatility and high performance in both **research** and **production environments**.

---

## ✨ Features

- 🧠 **Dual-Model Architecture**: Plug-and-play **Transformer (GPT)** and **LSTM** models for flexible experimentation and deployment.
- ⚡ **Full CUDA Acceleration**: Leverages GPU power via both PyTorch and TensorFlow for lightning-fast training and inference.
- 🎛️ **Customizable Generation Parameters**: Tweak temperature, top-k, top-p, and more for tailored outputs.
- 📊 **Integrated Performance Benchmarking**: Built-in tools for measuring speed, memory, and quality metrics.
- 🔌 **Modular & Extensible Design**: Easily extend, swap, or stack models and components.
- 🔒 **Robust Error Handling**: Comprehensive logging and exception management for stable runs.
- 🧪 **Experiment Tracking**: Optional integration with [Weights & Biases](https://wandb.ai/) for experiment management and visualization.
- 📈 **Batch and Interactive Modes**: Use in scripts or in real-time Jupyter notebooks.

---

## 🖥️ Sample Input/Output

### GPT-2 Generation

**Command:**
```python
from models.gpt_generator import GPTGenerator
import time

print('=== GPT-2 Generation ===')
gpt = GPTGenerator()
start = time.time()
output = gpt.generate_text(
    'Artificial intelligence will', 
    max_length=150,
    temperature=0.7
)
print(f'Generated in {time.time()-start:.2f}s:\\n{output}')
```

**Output:**
```
=== GPT-2 Generation ===
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated in 2.11s:
Artificial intelligence will eventually become more versatile and more intelligent than any other form of artificial intelligence.

The rise of artificial intelligence is due in large part to the increasing presence of artificial intelligence programs. These are computers that are programmed to perform tasks they cannot perform today. These programs are programmed to perform tasks that humans cannot perform today. They learn from the past, and are not programmed to perform the same tasks that are now performed today. Therefore, the future of artificial intelligence is uncertain.

This is because the future of artificial intelligence is uncertain. Today, most of the world's population is expected to be about 75 percent autonomous. Today, most of the world's population is expected to be 100 percent autonomous. This means that the computer that
```

---

### LSTM Generation

**Command:**
```python
from models.lstm_generator import LSTMModel, TextTokenizer
from utils.preprocess import clean_text, create_sequences
import numpy as np
import tensorflow as tf
import time

print('=== LSTM Generation ===')
with open('data/sample_texts.txt', 'r', encoding='utf-8') as f:
    text = clean_text(f.read())
    
tokenizer = TextTokenizer(text)
lstm = LSTMModel(tokenizer, seq_length=50)

sequences, next_chars = create_sequences(text, seq_length=50)
X = np.array([[tokenizer.char_to_idx[c] for c in seq] for seq in sequences])
# Convert to one-hot encoding for input
X = tf.keras.utils.to_categorical(X, num_classes=tokenizer.vocab_size)
y = tf.keras.utils.to_categorical(
    [tokenizer.char_to_idx[c] for c in next_chars],
    num_classes=tokenizer.vocab_size
)

print('Training (3 epochs)...')
train_start = time.time()
lstm.train(X, y, epochs=3, batch_size=128)
train_time = time.time() - train_start

gen_start = time.time()
generated = lstm.generate_text('The future of AI', length=200, temperature=0.6)     
gen_time = time.time() - gen_start

print('\nGenerated text:')
print('-'*50)
print(generated)
print('-'*50)
print(f'Training time: {train_time:.2f}s')
print(f'Generation time: {gen_time:.2f}s')
print(f'Total time: {train_time + gen_time:.2f}s')
```

**Output:**
```
=== LSTM Generation ===
2025-05-20 00:33:44.398938: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-05-20 00:33:44.865961: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3506 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3050 6GB Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.6
Training (3 epochs)...
Epoch 1/3
2025-05-20 00:33:47.583985: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
2025-05-20 00:33:48.393887: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.    
4/6 [===================>..........] - ETA: 0s - loss: 3.1847 
Epoch 1: loss improved from inf to 3.13157, saving model to lstm_model.h5
6/6 [==============================] - 3s 26ms/step - loss: 3.1316
Epoch 2/3
5/6 [========================>.....] - ETA: 0s - loss: 2.9673
Epoch 2: loss improved from 3.13157 to 2.96774, saving model to lstm_model.h5
6/6 [==============================] - 0s 21ms/step - loss: 2.9677
Epoch 3/3
4/6 [===================>..........] - ETA: 0s - loss: 2.9411
Epoch 3: loss improved from 2.96774 to 2.93240, saving model to lstm_model.h5
6/6 [==============================] - 0s 23ms/step - loss: 2.9324

Generated text:
--------------------------------------------------
The future of AIm a e irert ecmrthecciree cacdeuineri eoee aln iniesadtedr rs slc e   ee eee ecul aoeniceei la aemu sitesepceseeera eansl o a i i r ri me naoemreai   i enipseanecoa gceee nooi  eineasceafteeee  r a ae
--------------------------------------------------
Training time: 3.73s
Generation time: 10.45s
Total time: 14.18s
```

---

## 🛠️ Prerequisites

### Hardware

- **NVIDIA GPU** (RTX 2000/3000 series recommended)
- **CUDA 11.8** compatible drivers
- **Minimum 8GB VRAM** (more recommended for longer sequences)

### Software

- **Windows 10/11** or **Linux**
- **Conda/Miniconda** (recommended for environment management)
- **NVIDIA CUDA Toolkit 11.8**
- **cuDNN 8.6**
- **Python 3.10**

---

## 📦 Installation

### 1️⃣ Create Conda Environment

```bash
conda create --prefix R:\ml_gpu_env python=3.10
conda activate R:\ml_gpu_env
```

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch==2.0.1 tensorflow-gpu==2.12.0 transformers==4.30.0 nltk==3.7 numpy>=1.23.0 tqdm>=4.65.0 jupyterlab>=3.6.0
```

---

## ⚖️ Requirements

```text
torch==2.0.1
tensorflow-gpu==2.12.0
transformers==4.30.0
nltk==3.7
numpy>=1.23.0
tqdm>=4.65.0
jupyterlab>=3.6.0
```

---

## 🎮 GPU Usage & Optimization

### Configure for Maximum Performance

#### PyTorch

```python
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.amp.autocast(enabled=True)
```

#### TensorFlow

```python
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

- 🧑‍💻 **Tip:** Monitor GPU memory with `nvidia-smi` during heavy tasks!

---

## 🚦 Quick Start

After installation, try generating text:

```bash
python generate.py --model gpt --prompt "The future of AI is"
```

Or for LSTM:

```bash
python generate.py --model lstm --seed "Deep learning is"
```

See all options:
```bash
python generate.py --help
```

---

## 📊 Benchmarking

Run the integrated benchmarking suite:

```bash
python benchmark.py --model gpt --length 256 --batch_size 8
```

---

## 🧩 Extending the Toolkit

- 📚 Add your custom model in `models/`
- 🛠️ Implement new tokenizers in `tokenizers/`
- 🔬 Contribute new evaluation metrics in `metrics/`

---

## 📝 License

Licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact & Support

| Platform    | Link                                                                          |
| ----------- | ----------------------------------------------------------------------------- |
| GitHub      | [Pavansai20054](https://github.com/Pavansai20054)                             |
| LinkedIn    | [rangdal-pavansai](https://www.linkedin.com/in/rangdal-pavansai/)             |
| Email       | [pavansai.20066@gmail.com](mailto:pavansai.20066@gmail.com)                   |
| Instagram   | [@pavansai_rangdal](https://www.instagram.com/pavansai_rangdal)               |
| Facebook    | [rangdal.pavansai](https://www.facebook.com/rangdal.pavansai)                 |

---

> 💡 **Pro Tip:** For best results, fine-tune models on your domain-specific data using the included training scripts!
>
> ⭐ **Star this repo to stay updated with the latest features and improvements!**