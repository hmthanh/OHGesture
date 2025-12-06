# OHGesture: Detailed Architecture Documentation

**Version**: 1.0  
**Last Updated**: December 6, 2025  
**Project Type**: Research - Multimodal Gesture Synthesis

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline Architecture](#data-pipeline-architecture)
3. [Model Architecture](#model-architecture)
4. [Training Architecture](#training-architecture)
5. [Inference Architecture](#inference-architecture)
6. [Module Dependencies](#module-dependencies)
7. [Data Flow Diagrams](#data-flow-diagrams)

---

## System Overview

OHGesture is a conversational gesture synthesis system that generates realistic human body gestures synchronized with speech, emotion, and semantic content using deep learning and diffusion models.

### Core Capabilities

1. **Multimodal Input Processing**: Audio (WavLM) + Text (Word2Vec) + Emotion (One-hot)
2. **Generative Modeling**: Diffusion-based denoising for gesture sequence generation
3. **Style Conditioning**: Emotion-aware gesture synthesis
4. **Temporal Coherence**: Seed gesture conditioning for smooth transitions

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OHGesture System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Data Preprocessing                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ BVH → NPZ    │  │ WAV → WavLM  │  │ Text → Vec   │          │
│  │ (60fps→20fps)│  │ (16kHz mono) │  │ (FastText)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┴──────────────────┘                  │
│                           │                                      │
│                    ┌──────▼──────────┐                          │
│                    │   HDF5 Dataset  │                          │
│                    └──────┬──────────┘                          │
│                                                                  │
│  Phase 2: Training                                              │
│                    ┌──────▼──────────┐                          │
│                    │  DeepGesture    │                          │
│                    │    Model        │                          │
│                    │ (Transformer +  │                          │
│                    │  Diffusion)     │                          │
│                    └──────┬──────────┘                          │
│                           │                                      │
│                    ┌──────▼──────────┐                          │
│                    │  Checkpoints    │                          │
│                    └─────────────────┘                          │
│                                                                  │
│  Phase 3: Inference                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ New Audio    │  │ New Text     │  │ Emotion      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┴──────────────────┘                  │
│                           │                                      │
│                    ┌──────▼──────────┐                          │
│                    │ Trained Model   │                          │
│                    │ (Sampling)      │                          │
│                    └──────┬──────────┘                          │
│                           │                                      │
│                    ┌──────▼──────────┐                          │
│                    │  BVH Output     │                          │
│                    │  (Visualize)    │                          │
│                    └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline Architecture

### Phase 1: Raw Data Extraction & Preprocessing

**Location**: `ZeroEGGSProcessing/`

#### 1.1 Animation Processing (`zeggs_data_to_h5.py`)

```python
Input:  BVH files (60 FPS, ZEGGS skeleton format)
        ├── Joint rotations (quaternions)
        ├── Root position
        └── Hierarchy structure

Process:
        ├── Load BVH using process_zeggs_bvh.preprocess_animation()
        ├── Extract 1141 joint parameters (positions + rotations)
        ├── Normalize using mean/std (computed from full dataset)
        └── Save as NPZ files

Output: Normalized gesture arrays
        Shape: (n_frames, 1141)
        Format: NPZ
```

**Key Functions**:
- `preprocess_animation()`: Parse BVH, extract joints, downsample
- `pose2bvh()`: Convert numpy arrays back to BVH format

#### 1.2 Audio Processing

```python
Input:  WAV files (48kHz stereo or 16kHz mono)

Process:
        ├── Resample to 16kHz mono (if needed)
        ├── Normalize audio amplitude
        ├── Extract WavLM features (using pretrained WavLM-Large)
        │   ├── Input: 16kHz waveform
        │   ├── WavLM encoder: → 1024-dim features
        │   └── Linear projection: 1024-dim → 64-dim
        └── Interpolate to match gesture frame rate

Output: Audio feature arrays
        Shape: (n_frames, 64)
        Format: NPZ / HDF5
```

**Key Components**:
- `wavlm/wavlm_embedding.py`: WavLM model wrapper
- `wav2wavlm()`: Extract and interpolate WavLM features
- `WavEncoder`: Linear projection layer (1024 → 64)

#### 1.3 Text Embedding (`word2vec.py`)

```python
Input:  TextGrid/TSV files (word-level transcripts)
        Format: [start_time, end_time, word]

Process:
        ├── Parse TextGrid/TSV files
        ├── Load FastText model (crawl-300d-2M.vec)
        ├── For each word:
        │   ├── Get FastText embedding (300-dim)
        │   ├── Align to audio timestamp
        │   └── Replicate embedding for duration
        └── Create sequence of word vectors

Output: Text embedding arrays
        Shape: (n_words, 300) or (n_frames, 300)
        Format: NPY / HDF5
```

**Key Functions**:
- `load_word2vec_model()`: Load FastText model
- `text_grid2tsv()`: Convert TextGrid to TSV
- `load_tsv_aligned()`: Parse aligned transcripts

#### 1.4 Emotion Label Processing

```python
Input:  Filename patterns (e.g., "speaker_Happy_001")

Process:
        ├── Extract emotion from filename
        ├── Map to one-hot encoding
        │   Happy:   [1,0,0,0,0,0]
        │   Sad:     [0,1,0,0,0,0]
        │   Neutral: [0,0,1,0,0,0]
        │   Old:     [0,0,0,1,0,0]
        │   Angry:   [0,0,0,0,1,0]
        │   Relaxed: [0,0,0,0,0,1]
        └── Store as array

Output: Emotion vectors
        Shape: (6,)
        Format: NPY / HDF5
```

### Phase 2: Dataset Creation (`data_to_h5dataset.py`)

```python
Input:  Processed NPZ files from Phase 1

Process:
        ├── Load all preprocessed files
        ├── Apply windowing:
        │   ├── Window size: 88 frames (4.4 seconds at 20 FPS)
        │   ├── Stride: 10 frames (50% overlap)
        │   └── Audio window: 88/20 * 16000 = 70,400 samples
        ├── Align features:
        │   ├── Gesture: (88, 1141)
        │   ├── Audio: (70400,) → WavLM → (88, 64)
        │   ├── Text: (n_words, 300) → interpolate → (88, 300)
        │   └── Emotion: (6,) → broadcast → (88, 6)
        └── Write to HDF5

Output: HDF5 dataset
        Format: datasets_train.h5
        Structure:
            ├── clip_001
            │   ├── gesture: (88, 1141)
            │   ├── speech: (70400,) or (88, 64) if pre-extracted
            │   ├── text: (88, 300) or (n_words, 300)
            │   └── emotion: (6,)
            ├── clip_002
            └── ...
```

**Key Classes**:
- `DeepGesturePreprocessor`: Main preprocessing orchestrator
- `wavlm_init()`: Initialize WavLM model
- `wav2wavlm()`: Extract WavLM features from audio

### Phase 3: Training Data Loading

**Location**: `main/data_loader/deepgesture_dataset.py`

```python
class DeepGestureDataset(Dataset):
    """
    PyTorch Dataset for loading HDF5 training data
    
    Responsibilities:
        - Load samples from HDF5 on-the-fly
        - Return (gesture, emotion, speech, text) tuples
        - No additional preprocessing (already done)
    """
    
    def __getitem__(self, index):
        with h5py.File(self.h5_file, 'r') as h5:
            data = h5[self.keys[index]]
            return (
                data['gesture'][:],    # (88, 1141)
                data['emotion'][:],    # (6,)
                data['speech'][:],     # (88, 64) if WavLM pre-extracted
                data['text'][:]        # (88, 300) or (n_words, 300)
            )
```

**Custom Collate Function**:
```python
def custom_collate(batch):
    """
    Handle variable-length sequences (if text not pre-padded)
    
    Process:
        - Pad sequences to max length in batch
        - Create attention masks
        - Stack into tensors
    """
    gestures, emotions, speeches, texts = zip(*batch)
    
    return (
        pad_sequence(gestures, batch_first=True),   # (B, 88, 1141)
        pad_sequence(emotions, batch_first=True),   # (B, 6)
        pad_sequence(speeches, batch_first=True),   # (B, 88, 64)
        pad_sequence(texts, batch_first=True)       # (B, 88, 300)
    )
```

---

## Model Architecture

### DeepGesture Model

**Location**: `main/model/deepgesture.py`

#### Overall Architecture

```python
class DeepGesture(nn.Module):
    """
    Multimodal diffusion-based gesture synthesis model
    
    Architecture:
        Input Branch (Multi-modal encoders)
        ↓
        Feature Fusion (Concatenation/Attention)
        ↓
        Transformer Encoder (8 layers)
        ↓
        Output Decoder
        ↓
        Gesture Prediction (1141 joints)
    """
```

#### Component Breakdown

**1. Input Encoders**

```python
# Audio Encoder
self.speech_linear_encoder = WavEncoder()
    Input:  (B, 88, 1024)  # Raw WavLM features
    Output: (B, 88, 64)    # Projected features

# Text Encoder
self.text_linear_encoder = TextEncoder()
    Input:  (B, 88, 300)   # Word2Vec embeddings
    Output: (B, 88, 64)    # Projected features

# Style/Emotion Encoder
self.style_linear_encoder = nn.Linear(6, 64)
    Input:  (B, 6)         # One-hot emotion
    Output: (B, 64)        # Style embedding
```

**2. Feature Fusion**

```python
# Option A: Concatenation (style1)
if 'style1' in cond_mode:
    # Embed style as first token
    style_emb = style_linear_encoder(emotion)  # (B, 64)
    if n_seed > 0:
        seed_emb = seed_gesture_linear(seed_gesture)  # (B, 192)
        token_0 = concat([style_emb, seed_emb], dim=1)  # (B, 256)
    else:
        token_0 = style_emb  # (B, 64) → pad to (B, 256)
    
    # Concatenate with audio/text
    x_input = concat([x, audio_feat], dim=-1)  # (B, 88, 1141+64)
    x_latent = input_process(x_input)  # (B, 88, 256)
    x_full = concat([token_0.unsqueeze(1), x_latent], dim=1)  # (B, 89, 256)

# Option B: Frame-wise fusion (style2)
if 'style2' in cond_mode:
    style_emb = style_linear_encoder(emotion)  # (B, 64)
    style_broadcast = style_emb.unsqueeze(1).expand(-1, 88, -1)  # (B, 88, 64)
    
    x_input = concat([x, audio_feat, style_broadcast], dim=-1)  # (B, 88, 1141+64+64)
    x_latent = input_process(x_input)  # (B, 88, 256)
```

**3. Cross-Attention Mechanisms**

```python
# Local Cross-Attention (cross_local_attention3)
if 'cross_local_attention3' in cond_mode:
    """
    Fuse audio and gesture features with local windowed attention
    
    Process:
        1. Self-attention on gesture features
        2. Cross-attention: gesture ← audio
        3. Residual connection
    """
    self.cross_local_attention = LocalAttention(
        dim=32,
        window_size=11,
        causal=True,
        look_backward=1,
        look_forward=0
    )
    
    # In forward():
    q = gesture_latent  # (B, 88, 256)
    k = audio_latent    # (B, 88, 64)
    v = audio_latent
    
    attended = cross_local_attention(q, k, v)  # (B, 88, 256)
    gesture_latent = gesture_latent + attended  # Residual
```

**4. Transformer Encoder**

```python
self.seqTransEncoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=256,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation='gelu',
        batch_first=True
    ),
    num_layers=8
)

# In forward():
x_encoded = seqTransEncoder(x_latent)  # (B, 88, 256)
```

**5. Timestep Embedding (Diffusion)**

```python
self.embed_timestep = TimestepEmbedder(latent_dim=256, pos_encoder=...)

# In forward():
t_emb = embed_timestep(timesteps)  # (B, 256)
t_emb = t_emb.unsqueeze(1)  # (B, 1, 256)

# Add to each frame
x_encoded = x_encoded + t_emb  # Broadcast
```

**6. Output Decoder**

```python
self.output_process = OutputProcess(
    data_rep='rot6d',
    input_feats=1141,
    latent_dim=256,
    njoints=1141,
    nfeats=1
)

# In forward():
output = output_process(x_encoded)  # (B, 1141, 1, 88)
```

#### Forward Pass Summary

```python
def forward(self, x, timesteps, y=None, uncond_info=False):
    """
    Args:
        x: (B, 1141, 1, 88) - noisy gesture (diffusion input)
        timesteps: (B,) - diffusion timestep
        y: dict with keys:
            'audio': (B, 88, 64) - WavLM features
            'text': (B, 88, 300) - Word2Vec embeddings
            'style': (B, 6) - emotion one-hot
            'seed': (B, 1141, 1, n_seed) - seed gesture (optional)
        uncond_info: bool - classifier-free guidance flag
    
    Returns:
        output: (B, 1141, 1, 88) - predicted noise or x_0
    """
    # 1. Encode style/emotion
    style_emb = style_linear_encoder(y['style'])  # (B, 64)
    
    # 2. Encode seed gesture (if provided)
    if n_seed > 0:
        seed_emb = seed_gesture_linear(y['seed'])  # (B, 192)
        token_0 = concat([style_emb, seed_emb])  # (B, 256)
    
    # 3. Process input gesture
    x = x.permute(0, 3, 1, 2)  # (B, 88, 1141, 1)
    
    # 4. Encode audio/text
    audio_emb = speech_linear_encoder(y['audio'])  # (B, 88, 64)
    text_emb = text_linear_encoder(y['text'])      # (B, 88, 64)
    
    # 5. Concatenate features
    x_input = concat([x.squeeze(-1), audio_emb], dim=-1)  # (B, 88, 1141+64)
    
    # 6. Project to latent space
    x_latent = input_process(x_input)  # (B, 88, 256)
    
    # 7. Add style token
    x_full = concat([token_0.unsqueeze(1), x_latent], dim=1)  # (B, 89, 256)
    
    # 8. Embed timestep
    t_emb = embed_timestep(timesteps)  # (B, 1, 256)
    x_full = x_full + t_emb  # Broadcast
    
    # 9. Transformer encoding
    x_encoded = seqTransEncoder(x_full)  # (B, 89, 256)
    
    # 10. Remove style token
    x_encoded = x_encoded[:, 1:, :]  # (B, 88, 256)
    
    # 11. Decode to gesture
    output = output_process(x_encoded)  # (B, 1141, 1, 88)
    
    return output
```

### Diffusion Process

**Location**: `diffusion/gaussian_diffusion.py`

#### Forward Diffusion (Training)

```python
def q_sample(x_start, t, noise=None):
    """
    Add noise to clean data at timestep t
    
    Formula:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    Args:
        x_start: (B, 1141, 1, 88) - clean gesture
        t: (B,) - timestep (0 to T-1)
        noise: (B, 1141, 1, 88) - Gaussian noise
    
    Returns:
        x_t: (B, 1141, 1, 88) - noisy gesture
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    alpha_bar_t = extract(alphas_cumprod, t, x_start.shape)
    
    return (
        sqrt(alpha_bar_t) * x_start +
        sqrt(1 - alpha_bar_t) * noise
    )
```

#### Reverse Diffusion (Inference)

```python
def p_sample(model, x_t, t, y=None):
    """
    Denoise one step: x_t → x_{t-1}
    
    Formula:
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta(x_t, t))
                  + sigma_t * z
    
    Args:
        model: DeepGesture model
        x_t: (B, 1141, 1, 88) - noisy gesture at step t
        t: (B,) - current timestep
        y: dict - conditioning (audio, text, style)
    
    Returns:
        x_{t-1}: (B, 1141, 1, 88) - denoised gesture
    """
    # Predict noise
    epsilon_pred = model(x_t, t, y=y)  # (B, 1141, 1, 88)
    
    # Compute mean
    alpha_t = extract(alphas, t, x_t.shape)
    alpha_bar_t = extract(alphas_cumprod, t, x_t.shape)
    beta_t = extract(betas, t, x_t.shape)
    
    pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_bar_t)
    mean = (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred) / sqrt(alpha_t)
    
    # Add noise (except last step)
    if t[0] > 0:
        sigma_t = sqrt(beta_t)
        z = torch.randn_like(x_t)
        x_prev = mean + sigma_t * z
    else:
        x_prev = mean
    
    return x_prev
```

---

## Training Architecture

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

