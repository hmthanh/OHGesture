# OHGesture Project: Architecture & Process Documentation

## Overview

OHGesture is a conversational gesture synthesis system that generates human gestures from speech, emotion, and semantic input. It leverages deep learning, multimodal data processing, and advanced audio/animation pipelines. The project is modular, supporting multiple datasets (e.g., ZEGGS, HumanAct12, UESTC) and includes tools for data preprocessing, training, sampling, and evaluation.

---

## High-Level Architecture

### Main Components

- **Data Preparation & Processing**
  - Converts raw audio and animation data into aligned, feature-rich datasets.
  - Key scripts: zeggs_data_to_h5.py, word2vec.py, data_to_h5dataset.py, data_pipeline.py, data_pipeline.py.

- **Model Training**
  - Trains gesture synthesis models using processed data.
  - Key scripts: train.py, train.py, ohgesture_finetune.py, ohgesture.py.

- **Sampling & Inference**
  - Generates gestures from new audio/text input using trained models.
  - Key scripts: sampling.py, sample.py, sampling.py.

- **Evaluation**
  - Evaluates model performance on benchmark datasets.
  - Key scripts: eval_humanml.py, eval_humanact12_uestc.py.

- **Utilities & Support**
  - Configuration, logging, seed fixing, and model utilities.
  - Key modules: utils, model, wavlm, body_models.

---

## Detailed Process & Pipeline

### 1. Data Preparation

#### a. Raw Data Extraction

- **ZEGGS Dataset**: Extract large zip files into `data/original` and `data/clean`.
- **Other Datasets**: Place raw files in designated folders.

#### b. Data Alignment & Feature Extraction

- **Audio Processing**: 
  - Extracts mel-spectrograms, energy, and other features (`audio/spectrograms.py`, `audio/audio_files.py`).
  - Uses `librosa`, `sox`, `ffmpeg` for audio manipulation.

- **Animation Processing**:
  - Loads BVH files, extracts pose, rotation, velocity, gaze, etc. (`anim/bvh.py`, `anim/quat.py`).
  - Ensures frame rate consistency (typically 60 FPS).

- **Word Embedding**:
  - Converts text to vector using FastText or Gensim (word2vec.py).

- **Pipeline Integration**:
  - data_pipeline.py orchestrates audio and animation preprocessing, label assignment, train/validation split, and feature normalization.
  - Outputs: `processed_data.npz`, `stats.npz`, `data_definition.json`, `data_pipeline_conf.json`.

#### c. HDF5 Conversion

- Converts processed data to HDF5 format for efficient training (data_to_h5dataset.py).

---

### 2. Model Training

- **Configuration**: Training options and network parameters are loaded from JSON/YAML config files.
- **Dataset Loading**: Custom PyTorch `SGDataset` loads processed data, applies windowing, style encoding, and batching.
- **Model Components**:
  - **Speech Encoder**: Extracts features from audio.
  - **Style Encoder**: Encodes gesture style (emotion, semantics).
  - **Decoder**: Generates gesture sequences.
  - **Diffusion Models**: Used for generative modeling (diffusion, mdm.py).
- **Training Loop**:
  - Iterates over batches, computes losses (e.g., KL divergence), updates model weights.
  - Logs metrics with TensorBoard.
  - Saves checkpoints and training metadata.

---

### 3. Sampling & Inference

- **Input**: New audio/text, style/emotion labels.
- **Feature Extraction**: Audio features and word embeddings are computed as in preprocessing.
- **Model Loading**: Loads trained weights.
- **Gesture Generation**: Model predicts gesture sequences, which are converted to BVH or other formats for visualization.
- **Postprocessing**: Optionally blends gestures, applies smoothing, or style transfer.

---

### 4. Evaluation

- **Benchmarking**: Evaluates generated gestures against ground truth using metrics (e.g., accuracy, diversity, realism).
- **Scripts**: eval_humanml.py, eval_humanact12_uestc.py.

---

## Key Packages & Dependencies

- **PyTorch**: Deep learning framework.
- **Librosa**: Audio feature extraction.
- **FastText/Gensim**: Word embeddings.
- **Numpy, Pandas**: Data manipulation.
- **Scipy**: Interpolation, signal processing.
- **Rich**: Console output and progress bars.
- **TensorBoard**: Training visualization.
- **BVH, Quat**: Animation and rotation handling.
- **Easydict, YAML**: Configuration management.

---

## Example Workflow

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

```bash
cd ZeroEGGSProcessing
python zeggs_data_to_h5.py
python word2vec.py --src=./data/train --dest=./processed/train/embedding --word2vec_model=./fasttext/crawl-300d-2M.vec
python data_to_h5dataset.py --config=../main/configs/OHGesture.yml
```

### 3. Run Data Pipeline

```bash
python ubisoft-laforge-ZeroEGGS/ZEGGS/data_pipeline.py
```

### 4. Train Model

```bash
python ubisoft-laforge-ZeroEGGS/ZEGGS/main.py -o "../configs/configs_v1.json" -n "zeggs_v1"
```

### 5. Sample Gestures

```bash
python main/sampling.py --input_audio your_audio.wav --output_gesture output.bvh
```

---

## Directory Structure

- main: Core scripts for training, sampling, and model management.
- diffuse_style_gesture: Alternative pipeline and models.
- ubisoft-laforge-ZeroEGGS: ZEGGS dataset and pipeline.
- ZeroEGGSProcessing: Data preprocessing utilities.
- model: Model architectures and layers.
- utils: Utility functions and configuration.
- wavlm: WavLM audio embedding.
- body_models, data_loaders, dataset: Support for other datasets.

---

## Deep Implementation Details

### Data Pipeline (data_pipeline.py)

- Loads configuration and info CSV.
- Iterates over samples:
  - Loads animation and audio.
  - Extracts features, normalizes, splits into train/validation.
  - Assigns style/emotion labels.
- Saves processed data and statistics.

### Training (train.py)

- Loads processed data and definitions.
- Initializes model components.
- Runs training loop with logging and checkpointing.

### Sampling (sampling.py, sample.py)

- Loads model and input features.
- Generates gesture sequence.
- Converts output to BVH for visualization.

---

## Customization

- **Config Files**: Change pipeline, model, and training parameters via JSON/YAML configs.
- **Modular Design**: Swap models, encoders, or datasets by editing import paths and config references.

---

## References

- See README.md and README.md for dataset and usage details.
- Each main script contains docstrings and comments for further guidance.

