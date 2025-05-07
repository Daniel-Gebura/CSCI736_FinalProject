# Real-Time ASL Gesture Classification

This repository provides a modular, end-to-end deep learning pipeline for classifying American Sign Language (ASL) word gestures using hand landmarks. It supports training GRU and Transformer architectures and includes tools for downloading MS-ASL video clips, extracting MediaPipe landmarks, verifying preprocessing, and running live demos.

---

## Setup & Dependencies

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .env
   source .env/bin/activate  # or .env\Scripts\activate on Windows
    ```

2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Project Directory Structure

```
.
├── ablation/                      # Scripts and data related to ablation experiments
├── data/
│   └── landmarks.zip              # Final compressed dataset used for training
├── msasl/                         # Source folder for downloaded MS-ASL clips and metadata
├── results/                       # Saved training plots and confusion matrices
├── saved_models/                  # Best model checkpoints for each classifier
├── scripts/                       # Data utilities and preprocessing tools
│   ├── download_asl100_clips.py   # Download and clip MS-ASL100 videos
│   ├── extract_landmarks.py       # Batch extract landmarks from videos
│   ├── extract_landmarks_DEMO.py  # Landmark visualizer for a single video
│   ├── verify_landmarks.py        # Visual overlay to verify .npy landmark files
│   └── dataset_analysis.py        # Prints class distribution and stats
├── src/
│   ├── models/                    # GRU and Transformer model definitions
│   ├── utils/                     # Metrics, visualization, helpers
│   ├── config.py                  # Central config for training hyperparameters
│   ├── dataloader.py              # Loads zipped landmark datasets
│   ├── evaluate_model.py          # Generates confusion matrices for best GRU model
│   ├── train.py                   # Main training script
│   └── transformer_demo.py        # Real-time webcam demo
```

---

## Step 1: (Optional) Download MS-ASL Videos and Extract Landmarks

> We recommend skipping this section if you already have `landmarks.zip` (provided in the GitHub repository) and don't want to generate your own dataset.

To download and clip ASL100 videos:

```bash
python scripts/download_asl100_clips.py
```

This will:

* Download MS-ASL metadata and videos
* Clip relevant segments
* Save them to `msasl/asl100_clips/`
* Generate `asl100_labels.csv`


To extract MediaPipe landmarks from downloaded clips:

```bash
python scripts/extract_landmarks.py
```

This will:

* Process videos in `msasl/asl100_clips/`
* Output `.npy` files to `data/landmarks/`
* You can then zip the folder to produce `landmarks.zip`

For a quick visual test on one video:

```bash
python scripts/extract_landmarks_DEMO.py
```

To verify the quality of extracted `.npy` files:

```bash
python scripts/verify_landmarks.py
```

---

## Step 2: Train a Classifier

Train the GRU classifier using:

```bash
cd src
python train.py
```

This script:

* Loads zipped hand landmark sequences from `data/landmarks.zip`
* Trains the model using defined architecture config
* Tracks validation accuracy and early stopping
* Saves best checkpoints to `saved_models/`
* Plots training loss/accuracy curves in `src/`

---

## Step 3: Evaluate a Trained Classifier

Once training is complete, you can evaluate any saved model by running:

```bash
cd src
python evaluate_model.py
```

This script:

* Loads the best saved model from `saved_models/`
* Runs evaluation on the validation set
* Computes a **confusion matrix**
* Saves the matrix plot to `src/` and prints confusion statistics

> **Note**: Pretrained model confusion matrices for both the GRU and Transformer classifiers are included in `results/` for reference.

---

## Step 4: Run the Real-Time Demo

To try out real-time gesture recognition using your webcam, run:

```bash
python src/transformer_demo.py
```

This demo script:

* Loads the best Transformer model from `saved_models/transformer_best.pth`
* Reads model architecture and label mappings from `transformer_info.json`
* Captures video frames from your webcam
* Detects up to 2 hands and extracts 132 landmark features per frame
* Buffers a sequence of 48 frames and performs a prediction when full
* Displays the predicted ASL word and confidence on screen
* Supports clearing the prediction buffer (`press 'c'`) or quitting (`press 'q'`)

As an example, `press 'c'` to clear the buffer and present a gesture from the dataset, such as `tired` or `teacher` (Instructions for performing these gestures can be found online). 

> Tip: Ensure your webcam is active and `MediaPipe`, `torch`, and `OpenCV` are correctly installed.