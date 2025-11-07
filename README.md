# 🎧 AudioNet ESC-50 Classifier

This repository contains a complete pipeline for training an audio classification model on the [ESC-50 dataset](https://github.com/karoldvl/ESC-50). It includes data preprocessing (spectrogram generation), model training, and utilities for conditional GPU usage. The model is based on EfficientNet-B4 and achieves up to **57% test accuracy**.

---

## 📁 Project Structure

* AudioNet-ESC50-Classifier/
* ├── data/
* │ ├── train/
* │ ├── valid/
* │ └── test/
* ├── precompute.py
* ├── train.py
* ├── model.py
* ├── utils.py
* └── README.md

---

## 🚀 Getting Started

### 1. Download the ESC-50 Dataset

Download the ESC-50 dataset from <https://github.com/karoldvl/ESC-50>.

Unzip it and make sure your project folder looks like this:

* AudioNet-ESC50-Classifier/
* ├── ESC-50/
* │ ├── audio/
* │ └── meta/esc50.csv
* └── ...


> ⚠️ Do not place files directly in `data/` — the `precompute.py` script will handle that automatically.

---

### 2. Precompute Spectrograms

Run the following script to generate and split the dataset into `train`, `valid`, and `test` folders:

```
python precompute.py
```

This will:

Process all audio .wav files from ESC-50 into Mel Spectrogram .png images

Split the data into training, validation, and testing folders

Populate the data/ directory accordingly

📌 Important: This step can take several hours depending on your CPU/GPU and disk speed.
🛑 You only need to run this script once unless you change the ESC-50 files or spectrogram settings.

---

### 3. Train the Model
Once preprocessing is done, train the model by running:

```
python train.py
```

The script will:

- Load the precomputed spectrogram dataset from data/

- Train the AudioNet model (EfficientNet-B4 backbone with custom ReLU head and dropout)

#### 🖥️ GPU Recommended:
Training on CPU is supported but can take a long time. Use a CUDA-enabled GPU for optimal performance.

---

## 🧠 Model Architecture
- Backbone: EfficientNet-B4 (from torchvision)

- Head:

    - Fully connected classifier

    - ReLU activation

    - Dropout for regularization

- Output: 50-class (aligned with ESC-50 labels)

---

## 📊 Performance

| Metric              | Value                      |
|---------------------|----------------------------|
| Dataset             | ESC-50                     |
| Model Architecture  | EfficientNet-B4 + ReLU Head |
| GPU Used            | NVIDIA T4                  |
| Best Test Accuracy  | ~67%                       |

---

✅ Requirements
- Python 3.8+

- PyTorch 2.x

- NumPy, Librosa, Matplotlib, Pillow, etc.

Install dependencies (with optional CUDA support):

```
pip install -r requirements.txt
```

---

## 📌 Notes
- Make sure your environment has sufficient storage (~1 GB) for spectrograms.

- You are encouraged to tweak model architecture, transforms, or hyperparameters in model.py and train.py.

- The project is configured for future experiments: model swap, augmentation, or parameter tuning.

---
## 🙌 Acknowledgements

- [ESC-50 Dataset](https://github.com/karoldvl/ESC-50) by Karol J. Piczak  
- [EfficientNet PyTorch Implementation](https://pytorch.org/vision/stable/models/efficientnet.html)

---
