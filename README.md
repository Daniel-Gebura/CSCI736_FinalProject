# Gesture Recognition with GRU and Transformer Architectures

This repository contains a gesture classification pipeline built using PyTorch. It supports training and evaluation of deep learning models on sequence data derived from hand landmarks, with a focus on real-time American Sign Language (ASL) word recognition.

Supported models:
- GRU-based classifier with attention
- Transformer encoder-based classifier

---

## Directory Structure

```

.
├── config.py                   # Configuration for training
├── train.py                   # Unified training script
├── dataloader.py              # Loads and processes hand landmark sequences
├── models/
│   ├── gru\_classifier.py      # GRU-based classifier with attention
│   └── transformer\_classifier.py  # Transformer-based sequence classifier
├── utils/
│   ├── metrics.py             # Evaluation loop and accuracy tracking
│   ├── visualization.py       # Training curve plotting
├── data/
│   └── landmarks.zip          # Zipped folder of processed landmark sequences
├── experiments/
│   └── saved\_models/          # Saved model checkpoints
│   └── training\_curves\_\*.png  # Training/validation accuracy and loss plots

```

---

## Optional: Downloading the MS-ASL Dataset

> ⚠️ **Note**: Downloading MS-ASL is not required if you are working with preprocessed landmark data. This step is only for advanced users who want to process raw videos themselves.

To download the [MS-ASL dataset](https://github.com/ycjing/ASL-Dataset), follow the instructions in their GitHub repository. Be aware that the full dataset is large (~150 GB) and contains raw video data that will need preprocessing to extract landmarks.

---

## Setting Up the Landmark Dataset

Before training, ensure you have a processed dataset of hand landmarks. This should be a `.zip` file containing `.npy` files for each sample.

1. Place your dataset at:
```

data/landmarks.zip

````
2. Inside the zip file, each file should be a NumPy array of shape `(sequence_length, 132)`, where 132 = 63 keypoints (left) + 63 (right) + 3 (face) + 3 (pose).

3. The path to this file should be reflected in `config.py`:
```python
'zip_path': './data/landmarks.zip',
'landmarks_folder': 'landmarks/'
````

---

## Training the Model

To train both the GRU and Transformer classifiers:

```bash
python train.py
```

This script will:

* Set a global seed for reproducibility
* Load the dataset from the zip file
* Train each model architecture
* Save the best checkpoint for each model in `experiments/saved_models/`
* Save training curves in `experiments/training_curves_<model_name>.png`

---

## Running the Live Demo

> ⚠️ **Note**: This section will be updated once the live demo script is finalized.

To run a real-time gesture recognition demo using your webcam:

```bash
python demo.py
```

Make sure the following are in place:

* The trained model weights are located in `experiments/saved_models/`
* Your demo script references the correct model class and preprocessing pipeline

---

## Requirements

* Python 3.8+
* PyTorch 1.10+
* NumPy
* OpenCV (for live demo)
* scikit-learn
* tqdm
* matplotlib

Install all dependencies:

```bash
pip install -r requirements.txt
```