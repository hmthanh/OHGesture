# OHGesture - Project Summary

**Last Updated**: December 6, 2025  
**Project Type**: Deep Learning Research - Conversational Gesture Synthesis  
**Technology Stack**: Python 3.11, PyTorch, Diffusion Models

---

## Executive Summary

OHGesture is a conversational gesture synthesis system that generates realistic human body gestures synchronized with speech, emotion, and semantic content. The system uses deep learning with diffusion models to create natural, expressive gestures from multimodal inputs (audio, text, emotion labels).

**Key Capabilities**:
- Generate full-body gestures (1141 joints) from speech audio
- Emotion-aware gesture synthesis (Happy, Sad, Neutral, Old, Angry, Relaxed)
- Semantic text understanding via word embeddings
- Style transfer and gesture conditioning
- Real-time inference with pretrained models

---

## Project Goals

### Primary Objective
Create a robust, multimodal gesture synthesis system that generates human-like gestures that are:
1. **Synchronized** with speech timing and prosody
2. **Emotionally appropriate** to the speaker's affective state
3. **Semantically aligned** with the content being spoken
4. **Naturally continuous** across long sequences

### Research Contributions
- Integration of audio (WavLM), text (Word2Vec/FastText), and emotion features
- Cross-attention mechanisms for multimodal feature fusion
- Diffusion-based generative modeling for gesture sequences
- Support for seed gesture conditioning for temporal coherence

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OHGesture Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐          │
│  │   Audio     │  │    Text      │  │   Emotion     │          │
│  │  (WavLM)    │  │ (Word2Vec)   │  │  (One-hot)    │          │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘          │
│         │                 │                   │                  │
│         └─────────────────┴───────────────────┘                  │
│                           │                                      │
│                    ┌──────▼──────────┐                          │
│                    │   DeepGesture   │                          │
│                    │     Model       │                          │
│                    │  (Transformer+  │                          │
│                    │   Diffusion)    │                          │
│                    └──────┬──────────┘                          │
│                           │                                      │
│                    ┌──────▼──────────┐                          │
│                    │  Gesture Poses  │                          │
│                    │  (1141 joints)  │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Phase 1: Raw Data Processing
**Location**: `ZeroEGGSProcessing/`

1. **Animation Data** (`zeggs_data_to_h5.py`)
   - Input: BVH animation files (60 FPS)
   - Process: Extract joint rotations, normalize with mean/std
   - Output: NPZ files with 1141 joint parameters

2. **Audio Processing**
   - Input: WAV files (16kHz mono)
   - Process: Extract WavLM features (1024-dim → 64-dim)
   - Output: Audio feature arrays synchronized to gesture frames

3. **Text Embedding** (`word2vec.py`)
   - Input: TextGrid/TSV word-level transcripts
   - Process: Convert words to FastText vectors (300-dim)
   - Output: Sequence of word embeddings aligned with audio

4. **Emotion Labels**
   - Input: Style annotations from filenames (e.g., `speaker_Happy_001`)
   - Process: One-hot encoding (6 emotion classes)
   - Output: Emotion vectors

### Phase 2: Dataset Creation
**Location**: `ZeroEGGSProcessing/data_to_h5dataset.py`

- **Windowing**: Divide long sequences into 88-frame clips (4.4 seconds at 20 FPS)
- **Stride**: Use 10-frame overlap for training continuity
- **Feature Alignment**: Sync audio (16kHz), text, and gesture (20 FPS resampled from 60 FPS)
- **Output Format**: HDF5 with keys: `gesture`, `speech`, `text`, `emotion`

### Phase 3: Training Data Loading
**Location**: `main/data_loader/deepgesture_dataset.py`

- **Dataset**: `DeepGestureDataset` class loads from HDF5
- **Batching**: Custom collate function handles variable-length sequences
- **Normalization**: Apply mean/std normalization from preprocessing

---

## Model Architecture

### DeepGesture Model
**Location**: `main/model/deepgesture.py`

```python
Components:
├── Speech Encoder (WavEncoder)
│   └── WavLM features (1024-dim) → Linear (64-dim)
├── Text Encoder (TextEncoder)  
│   └── Word2Vec features (300-dim) → Linear (64-dim)
├── Style Encoder
│   └── Emotion one-hot (6-dim) → Linear (64-dim)
├── Input Processor
│   └── Concatenate features → Linear (256-dim latent)
├── Transformer Encoder (8 layers)
│   └── Self-attention + FFN (dim=256, heads=4, ff=1024)
├── Cross-Attention Modules
│   └── Fuse audio/text/style features
├── Timestep Embedding
│   └── Diffusion timestep → Sinusoidal embedding
└── Output Processor
    └── Latent (256-dim) → Gesture (1141 joints)
```

### Diffusion Process
**Location**: `diffusion/gaussian_diffusion.py`

- **Forward Process**: Add Gaussian noise over T=1000 timesteps
- **Reverse Process**: Denoise using learned model
- **Loss**: MSE + KL divergence for variance learning
- **Sampling**: DDPM/DDIM for generation

---

## Training Pipeline

### Configuration
**Location**: `main/configs/OHGesture.yml`

Key hyperparameters:
```yaml
n_poses: 88                  # Sequence length
batch_size: 640              # Training batch size
epochs: 500000               # Training iterations
lr: 0.00003                  # Learning rate
motion_resampling_framerate: 20  # Target FPS
subdivision_stride: 10       # Window stride
```

### Training Loop
**Location**: `train/deepgesture_training_loop.py`

1. **Initialize**
   - Load DeepGesture model and Gaussian diffusion
   - Setup AdamW optimizer (β=[0.5, 0.999])
   - Configure mixed-precision trainer (optional FP16)

2. **Iteration**
   ```python
   for epoch in range(epochs):
       for batch in train_loader:
           gesture, emotion, speech, text = batch
           
           # Forward diffusion (add noise)
           t = random_timestep()
           x_t = q_sample(gesture, t)
           
           # Model prediction
           noise_pred = model(x_t, t, y={
               'audio': speech,
               'text': text,
               'style': emotion
           })
           
           # Compute loss
           loss = mse_loss(noise_pred, noise)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

3. **Checkpointing**
   - Save every 25 epochs
   - Location: `./output/checkpoint/ohgesture/model*.pt`

---

## Inference Pipeline

### Sampling Process
**Location**: `main/sampling.py`

1. **Load Model**
   ```python
   model, diffusion = create_model_and_diffusion(args)
   model.load_state_dict(torch.load(checkpoint_path))
   model.eval()
   ```

2. **Prepare Input**
   - Load audio WAV file (16kHz)
   - Extract WavLM features
   - Load/generate word embeddings
   - Set emotion style (one-hot)

3. **Generate Gestures**
   ```python
   # Start from random noise
   x_T = torch.randn(shape)
   
   # Reverse diffusion
   for t in reversed(range(T)):
       x_t = p_sample(model, x_t, t, 
                      audio=audio_feat,
                      text=text_embed,
                      style=emotion)
   
   gesture = x_0
   ```

4. **Post-processing**
   - Denormalize: `gesture = gesture * std + mean`
   - Convert to BVH format for visualization
   - Apply optional smoothing

5. **Visualization**
   - Export BVH file
   - Import to Unity 6 with deepgesture-unity viewer

---

## Key Dependencies

### Core Libraries
```python
torch==2.3.1              # Deep learning framework
numpy==1.26.4             # Numerical computing
transformers              # WavLM model
librosa==0.10.2          # Audio processing
scipy                     # Signal processing
```

### Audio Processing
```python
soundfile==0.12.1        # Audio I/O
sox==1.5.0               # Audio manipulation
ffmpeg-normalize==1.28.2 # Audio normalization
```

### Data Management
```python
h5py                      # HDF5 dataset storage
lmdb==1.5.1              # Fast key-value storage
pandas==2.2.2            # Data manipulation
```

### ML/Training
```python
tensorboard==2.17.0      # Training visualization
tqdm==4.66.4             # Progress bars
einops==0.8.0            # Tensor operations
ema-pytorch==0.5.1       # EMA for training
accelerate==0.32.1       # Distributed training
```

### Text Processing
```python
gensim                    # Word2Vec models
fasttext                  # FastText embeddings
```

---

## File Structure

```
OHGesture/
├── main/                           # Core training & inference
│   ├── ohgesture.py               # Main training script
│   ├── sampling.py                # Inference script
│   ├── configs/
│   │   └── OHGesture.yml          # Config file
│   ├── data_loader/
│   │   └── deepgesture_dataset.py # Dataset class
│   ├── model/
│   │   └── deepgesture.py         # Model architecture
│   └── output/                    # Checkpoints & results
│
├── ZeroEGGSProcessing/            # Data preprocessing
│   ├── zeggs_data_to_h5.py        # Convert BVH to NPZ
│   ├── word2vec.py                # Text embedding
│   ├── data_to_h5dataset.py       # Create HDF5 dataset
│   └── processed/                 # Processed data
│       ├── mean.npz
│       ├── std.npz
│       └── train/
│
├── train/                         # Training utilities
│   └── deepgesture_training_loop.py
│
├── diffusion/                     # Diffusion model
│   ├── gaussian_diffusion.py
│   ├── losses.py
│   └── respace.py
│
├── model/                         # Model components
│   ├── mdm.py                     # Motion diffusion base
│   └── rotation2xyz.py            # Pose conversion
│
├── wavlm/                         # Audio encoder
│   ├── wavlm_embedding.py
│   └── WavLM-Large.pt            # Pretrained weights
│
├── utils/                         # Utilities
│   ├── config.py
│   ├── model_util.py
│   └── rotation_conversions.py
│
├── process/                       # BVH processing
│   └── process_zeggs_bvh.py
│
├── eval/                          # Evaluation scripts
│   ├── eval_humanml.py
│   └── eval_humanact12_uestc.py
│
└── body_models/                   # SMPL body model
```

---

## Usage Workflows

### 1. Training from Scratch

```bash
# Step 1: Preprocess data
cd ZeroEGGSProcessing
python zeggs_data_to_h5.py
python word2vec.py --src=./data/train --dest=./processed/train/embedding --word2vec_model=./fasttext/crawl-300d-2M.vec
python data_to_h5dataset.py --config=../main/configs/OHGesture.yml

# Step 2: Train model
cd ../main
python ohgesture.py --config=./configs/OHGesture.yml --gpu cuda:0
```

### 2. Fine-tuning Pretrained Model

```bash
cd main
# Set model_finetune_path in OHGesture.yml
python ohgesture_finetune.py --config=./configs/OHGesture.yml --gpu cuda:0
```

### 3. Inference

```bash
cd main
python sampling.py \
    --config=./configs/OHGesture.yml \
    --input_audio=./input/audio.wav \
    --input_text=./input/transcript.txt \
    --emotion=Happy \
    --output=./output/gesture.bvh
```

### 4. Visualization

```bash
# Use Unity 6 with deepgesture-unity
git clone https://github.com/DeepGesture/deepgesture-unity
# Open in Unity 6 and import generated BVH file
```

---

## Research Context

### Based On
- **DiffuseStyleGesture**: Foundation for diffusion-based gesture synthesis
- **ZEGGS Dataset**: Zero-Shot Expressive Gesture Generation dataset by Ubisoft
- **MDM (Motion Diffusion Model)**: Motion generation architecture
- **WavLM**: Microsoft's speech representation model

### Key Innovations
1. **Multimodal Fusion**: Combines audio (WavLM) + text (Word2Vec) + emotion
2. **Cross-Attention**: Novel attention mechanisms for feature correlation
3. **Style Conditioning**: Emotion-aware gesture generation
4. **Seed Gesture**: Temporal coherence via initial pose conditioning

---

## Performance Characteristics

### Training
- **Dataset Size**: ~1000-5000 clips (depends on ZEGGS split)
- **Training Time**: ~1-2 weeks on single GPU (RTX 3090)
- **Batch Size**: 640 (requires 24GB+ VRAM)
- **Checkpoints**: Every 25 epochs (~50,000 iterations)

### Inference
- **Speed**: ~1-2 seconds per 4.4-second gesture clip (GPU)
- **Quality**: Realistic, synchronized gestures
- **Latency**: Near real-time with proper batching

### Resource Requirements
- **Training**: 24GB+ GPU, 32GB+ RAM
- **Inference**: 8GB+ GPU, 16GB+ RAM
- **Storage**: ~50GB for dataset + models

---

## Known Limitations

1. **Dataset Dependency**: Requires ZEGGS dataset (not publicly available in full)
2. **Computational Cost**: High VRAM requirements for training
3. **Language Support**: Currently English-only (FastText models)
4. **Body Representation**: Fixed 1141-joint skeleton (ZEGGS format)
5. **Real-time**: Not optimized for live streaming applications

---

## Future Directions

1. **Multi-language Support**: Extend to other languages
2. **Real-time Optimization**: Faster diffusion sampling (DDIM, DPM-Solver)
3. **Interactive Control**: User-guided gesture editing
4. **Multi-speaker**: Generalize across different speakers
5. **Face & Hands**: Add facial expressions and finger gestures

---

## Citation

```bibtex
@inproceedings{ohgesture2025,
  title={OHGesture: A Conversational Gesture Synthesis System},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

---

## Links

- **Homepage**: https://deepgesture.github.io
- **Pretrained Models**: [OneDrive](https://1drv.ms/f/s!AvSTDY2o11xHgalWGd7PGtdj5yOiRA?e=xek1oW)
- **Unity Viewer**: https://github.com/DeepGesture/deepgesture-unity
- **Base Repository**: https://github.com/YoungSeng/DiffuseStyleGesture
