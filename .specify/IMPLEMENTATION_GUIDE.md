# OHGesture: Complete Implementation Guide

**Version**: 1.0  
**Last Updated**: December 6, 2025  
**Purpose**: Step-by-step guide for implementing new tasks/features

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Adding New Audio Features](#adding-new-audio-features)
3. [Adding New Text Embeddings](#adding-new-text-embeddings)
4. [Adding New Emotion Classes](#adding-new-emotion-classes)
5. [Modifying Model Architecture](#modifying-model-architecture)
6. [Creating Custom Datasets](#creating-custom-datasets)
7. [Implementing New Sampling Strategies](#implementing-new-sampling-strategies)
8. [Adding Evaluation Metrics](#adding-evaluation-metrics)
9. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## Prerequisites

### Required Knowledge

- **Python**: Advanced (classes, decorators, context managers)
- **PyTorch**: Intermediate (models, optimizers, data loaders)
- **Deep Learning**: Transformers, attention mechanisms, diffusion models
- **Audio Processing**: Librosa, spectrograms, resampling
- **3D Animation**: BVH format, skeletal hierarchies, rotations

### Development Environment

```bash
# 1. Clone repository
git clone https://github.com/hmthanh/OHGesture.git
cd OHGesture

# 2. Create conda environment
conda create -n ohgesture python=3.11
conda activate ohgesture

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pretrained models
# Download from OneDrive link and place in ./main/

# 5. Prepare ZEGGS dataset
# Place dataset in ZeroEGGSProcessing/data/
```

### Project Structure Understanding

```
OHGesture/
├── main/                      # Core training & inference
│   ├── ohgesture.py          # Training entry point
│   ├── sampling.py           # Inference entry point
│   ├── configs/              # YAML config files
│   ├── data_loader/          # Dataset classes
│   └── model/                # Model architecture
├── ZeroEGGSProcessing/       # Data preprocessing
├── train/                    # Training loops
├── diffusion/                # Diffusion algorithms
├── model/                    # Shared model components
├── wavlm/                    # Audio encoder
└── utils/                    # Utilities
```

---

## Adding New Audio Features

### Goal

Replace WavLM with a custom audio encoder (e.g., Wav2Vec, HuBERT, MFCC+CNN).

### Steps

#### 1. Create Audio Encoder Module

**File**: `wavlm/custom_audio_encoder.py`

```python
import torch
import torch.nn as nn
import librosa

class CustomAudioEncoder(nn.Module):
    """
    Custom audio feature extractor
    
    Example: MFCC + CNN encoder
    """
    def __init__(self, 
                 n_mfcc=13, 
                 sample_rate=16000,
                 output_dim=64):
        super().__init__()
        
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        
        # CNN layers
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, output_dim, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
    
    def extract_mfcc(self, waveform):
        """
        Extract MFCC features from waveform
        
        Args:
            waveform: (batch_size, audio_samples) - 16kHz audio
        
        Returns:
            mfcc: (batch_size, n_mfcc, time_frames)
        """
        batch_size = waveform.shape[0]
        mfcc_list = []
        
        for i in range(batch_size):
            wav = waveform[i].cpu().numpy()
            mfcc = librosa.feature.mfcc(
                y=wav,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=1024,
                hop_length=160  # 10ms hop
            )
            mfcc_list.append(torch.from_numpy(mfcc))
        
        return torch.stack(mfcc_list).to(waveform.device)
    
    def forward(self, waveform, target_length=88):
        """
        Forward pass
        
        Args:
            waveform: (B, audio_samples) - e.g., (B, 70400) for 4.4s
            target_length: int - target sequence length (e.g., 88 frames)
        
        Returns:
            features: (B, target_length, output_dim)
        """
        # Extract MFCC
        mfcc = self.extract_mfcc(waveform)  # (B, n_mfcc, T)
        
        # CNN encoding
        x = self.relu(self.conv1(mfcc))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # (B, output_dim, T')
        
        # Interpolate to target length
        x = x.permute(0, 2, 1)  # (B, T', output_dim)
        x = torch.nn.functional.interpolate(
            x.permute(0, 2, 1),
            size=target_length,
            mode='linear',
            align_corners=True
        )
        x = x.permute(0, 2, 1)  # (B, target_length, output_dim)
        
        return x
```

#### 2. Integrate into Preprocessing

**File**: `ZeroEGGSProcessing/data_to_h5dataset.py`

```python
# Add import
from wavlm.custom_audio_encoder import CustomAudioEncoder

class DeepGesturePreprocessor:
    def __init__(self, args, ...):
        # Replace WavLM with custom encoder
        if args.audio_encoder == 'custom':
            self.audio_encoder = CustomAudioEncoder(
                n_mfcc=13,
                sample_rate=16000,
                output_dim=64
            ).to(self.device)
        elif args.audio_encoder == 'wavlm':
            self.audio_encoder = wavlm_init(args, self.device)
    
    def extract_audio_features(self, audio_raw):
        """Extract audio features using configured encoder"""
        audio_tensor = torch.from_numpy(audio_raw).float()
        audio_tensor = audio_tensor.to(self.device).unsqueeze(0)
        
        if args.audio_encoder == 'custom':
            features = self.audio_encoder(audio_tensor, target_length=self.n_poses)
        elif args.audio_encoder == 'wavlm':
            features = wav2wavlm(args, self.audio_encoder, audio_tensor)
        
        return features.squeeze().cpu().detach().numpy()
```

#### 3. Update Model

**File**: `main/model/deepgesture.py`

```python
class DeepGesture(nn.Module):
    def __init__(self, ..., audio_feat='wavlm', **kargs):
        super().__init__()
        
        # Configure audio encoder
        if audio_feat == 'wavlm':
            self.audio_feat_dim = 64
            self.speech_encoder = WavEncoder()
        elif audio_feat == 'custom':
            self.audio_feat_dim = 64
            self.speech_encoder = nn.Identity()  # Already extracted
        elif audio_feat == 'mfcc':
            self.audio_feat_dim = 13
            self.speech_encoder = nn.Identity()
        
        # Rest of model initialization...
```

#### 4. Update Config

**File**: `main/configs/OHGesture_custom.yml`

```yaml
# Audio configuration
audio_feat: "custom"           # Options: wavlm, custom, mfcc
audio_encoder: "custom"        # For preprocessing
audio_feat_dim: 64             # Output dimension

# Preprocessing paths
custom_audio_encoder_path: "./wavlm/custom_audio_encoder.py"
```

#### 5. Test

```bash
# Test preprocessing
cd ZeroEGGSProcessing
python data_to_h5dataset.py --config=../main/configs/OHGesture_custom.yml --audio_encoder=custom

# Test training
cd ../main
python ohgesture.py --config=./configs/OHGesture_custom.yml --gpu cuda:0
```

---

## Adding New Text Embeddings

### Goal

Replace Word2Vec with BERT/Sentence Transformers.

### Steps

#### 1. Create Text Encoder

**File**: `main/utils/text_embeddings.py`

```python
import torch
from transformers import BertTokenizer, BertModel

class BERTTextEncoder:
    """
    BERT-based text encoder for semantic understanding
    """
    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def encode_sentence(self, text):
        """
        Encode a sentence to embedding
        
        Args:
            text: str - input sentence
        
        Returns:
            embedding: (768,) - BERT embedding
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def encode_word_sequence(self, words, timestamps):
        """
        Encode sequence of words with timestamps
        
        Args:
            words: List[str] - list of words
            timestamps: List[Tuple[float, float]] - [(start, end), ...]
        
        Returns:
            embeddings: (n_words, 768)
            times: (n_words, 2)
        """
        embeddings = []
        for word in words:
            emb = self.encode_sentence(word)
            embeddings.append(emb)
        
        return np.array(embeddings), np.array(timestamps)
```

#### 2. Update Preprocessing

**File**: `ZeroEGGSProcessing/bert_embeddings.py`

```python
from main.utils.text_embeddings import BERTTextEncoder

def create_bert_embeddings(args):
    """
    Create BERT embeddings for all transcripts
    """
    encoder = BERTTextEncoder(device=args.device)
    
    # Load transcripts
    transcript_files = glob.glob(f"{args.src}/**/*.tsv", recursive=True)
    
    for tsv_file in tqdm(transcript_files):
        # Parse transcript
        words = []
        timestamps = []
        with open(tsv_file, 'r') as f:
            for line in f:
                start, end, word = line.strip().split('\t')
                words.append(word)
                timestamps.append((float(start), float(end)))
        
        # Encode
        embeddings, times = encoder.encode_word_sequence(words, timestamps)
        
        # Save
        output_path = tsv_file.replace('.tsv', '_bert.npy')
        np.save(output_path, {
            'embeddings': embeddings,  # (n_words, 768)
            'timestamps': times         # (n_words, 2)
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    create_bert_embeddings(args)
```

#### 3. Update Model

**File**: `main/model/deepgesture.py`

```python
class TextEncoder(nn.Module):
    """Project text embeddings to model dimension"""
    def __init__(self, text_feat='word2vec', latent_dim=64):
        super().__init__()
        
        if text_feat == 'word2vec':
            input_dim = 300
        elif text_feat == 'bert':
            input_dim = 768
        elif text_feat == 'sentence_bert':
            input_dim = 384
        
        self.linear = nn.Linear(input_dim, latent_dim)
    
    def forward(self, text_feat):
        return self.linear(text_feat)
```

#### 4. Update Config

```yaml
text_feat: "bert"              # Options: word2vec, bert, sentence_bert
text_feat_dim: 768             # BERT embedding dimension
text_encoder_output: 64        # Projected dimension
```

---

## Adding New Emotion Classes

### Goal

Extend from 6 emotions to custom set (e.g., add "Surprised", "Fearful").

### Steps

#### 1. Update Emotion Mapping

**File**: `ZeroEGGSProcessing/emotion_config.py`

```python
# Original
EMOTION_CLASSES_V1 = {
    'Happy': 0,
    'Sad': 1,
    'Neutral': 2,
    'Old': 3,
    'Angry': 4,
    'Relaxed': 5
}

# Extended
EMOTION_CLASSES_V2 = {
    'Happy': 0,
    'Sad': 1,
    'Neutral': 2,
    'Old': 3,
    'Angry': 4,
    'Relaxed': 5,
    'Surprised': 6,    # NEW
    'Fearful': 7       # NEW
}

def emotion_to_onehot(emotion, version='v2'):
    """Convert emotion string to one-hot vector"""
    if version == 'v1':
        mapping = EMOTION_CLASSES_V1
        n_classes = 6
    elif version == 'v2':
        mapping = EMOTION_CLASSES_V2
        n_classes = 8
    
    onehot = np.zeros(n_classes)
    if emotion in mapping:
        onehot[mapping[emotion]] = 1
    else:
        # Default to Neutral
        onehot[mapping['Neutral']] = 1
    
    return onehot
```

#### 2. Update Data Preprocessing

**File**: `ZeroEGGSProcessing/zeggs_data_to_h5.py`

```python
from emotion_config import emotion_to_onehot, EMOTION_CLASSES_V2

# In make_h5_gesture_dataset():
for bvh_file in bvh_files:
    name = os.path.split(bvh_file)[1][:-4]
    emotion_str = name.split('_')[1]  # Extract from filename
    
    # Convert to one-hot
    emotion_vector = emotion_to_onehot(emotion_str, version='v2')  # (8,)
    
    # Save
    g_data.create_dataset('emotion', data=emotion_vector)
```

#### 3. Update Model

**File**: `main/model/deepgesture.py`

```python
class DeepGesture(nn.Module):
    def __init__(self, ..., n_emotions=6, **kargs):
        super().__init__()
        
        self.n_emotions = n_emotions
        
        # Style encoder
        if 'style1' in self.cond_mode:
            self.style_linear_encoder = nn.Linear(n_emotions, self.style_dim)
```

#### 4. Update Config

```yaml
n_emotions: 8                   # Number of emotion classes
emotion_version: 'v2'           # Which emotion set to use
```

#### 5. Collect New Data

**Annotation Guidelines**:

1. Record gesture sequences for new emotions
2. Name files: `speaker_{id}_{Emotion}_{clip_id}.bvh`
3. Ensure consistent emotion labeling
4. Balance dataset (similar number of clips per emotion)

#### 6. Retrain Model

```bash
# Reprocess dataset with new emotions
cd ZeroEGGSProcessing
python zeggs_data_to_h5.py --emotion_version=v2

# Train from scratch or fine-tune
cd ../main
python ohgesture.py --config=./configs/OHGesture_v2.yml --n_emotions=8
```

---

## Modifying Model Architecture

### Goal

Add a new attention mechanism or change transformer architecture.

### Example: Add Multi-Head Cross-Attention

#### 1. Define New Attention Module

**File**: `main/model/custom_attention.py`

```python
import torch
import torch.nn as nn

class MultiModalCrossAttention(nn.Module):
    """
    Cross-attention between gesture and audio/text features
    """
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Query: from gesture
        self.q_proj = nn.Linear(dim, dim)
        
        # Key, Value: from audio/text
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, T_q, dim) - gesture features
            key: (B, T_k, dim) - audio features
            value: (B, T_v, dim) - audio features (T_v = T_k)
            mask: (B, T_q, T_k) - attention mask (optional)
        
        Returns:
            output: (B, T_q, dim) - attended features
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        # Project
        Q = self.q_proj(query)  # (B, T_q, dim)
        K = self.k_proj(key)    # (B, T_k, dim)
        V = self.v_proj(value)  # (B, T_k, dim)
        
        # Reshape to multi-head
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_q, d)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_k, d)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T_k, d)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, T_q, T_k)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T_q, d)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, -1)  # (B, T_q, dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights
```

#### 2. Integrate into DeepGesture

**File**: `main/model/deepgesture.py`

```python
from model.custom_attention import MultiModalCrossAttention

class DeepGesture(nn.Module):
    def __init__(self, ..., cond_mode='cross_attention_v2', **kargs):
        super().__init__()
        
        # ...existing code...
        
        if 'cross_attention_v2' in self.cond_mode:
            print('Using Multi-Modal Cross-Attention')
            self.cross_attention = MultiModalCrossAttention(
                dim=self.latent_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
    
    def forward(self, x, timesteps, y=None, uncond_info=False):
        # ...existing encoding...
        
        # Apply cross-attention
        if 'cross_attention_v2' in self.cond_mode:
            # Gesture features as query
            Q = x_latent  # (B, 88, 256)
            
            # Audio features as key/value
            K = audio_emb  # (B, 88, 64) → project to 256
            V = audio_emb
            
            # Project audio to same dim as gesture
            K = self.audio_proj(K)  # (B, 88, 256)
            V = self.audio_proj(V)
            
            # Cross-attention
            attended, attn_weights = self.cross_attention(Q, K, V)
            
            # Residual connection
            x_latent = x_latent + attended
        
        # ...rest of forward pass...
```

#### 3. Update Config

```yaml
# Model architecture
cond_mode: "cross_attention_v2"    # New attention mode
arch: "trans_enc"                   # Base architecture
latent_dim: 256
num_heads: 8
```

#### 4. Train and Evaluate

```bash
# Train with new architecture
python ohgesture.py --config=./configs/OHGesture_cross_attn_v2.yml

# Compare with baseline
python eval/compare_architectures.py --baseline=model_baseline.pt --new=model_cross_attn_v2.pt
```

---

## Creating Custom Datasets

### Goal

Adapt OHGesture to work with your own gesture dataset.

### Requirements

- **Animation**: BVH files or joint positions/rotations
- **Audio**: WAV files synchronized with animation
- **Text** (optional): Word-level transcripts
- **Metadata**: Emotion/style labels

### Steps

#### 1. Convert Your Data to ZEGGS Format

**File**: `data_processing/convert_to_zeggs_format.py`

```python
import numpy as np
import os

class CustomDatasetConverter:
    """
    Convert custom dataset to ZEGGS-compatible format
    """
    def __init__(self, custom_dataset_path, output_path):
        self.input_path = custom_dataset_path
        self.output_path = output_path
    
    def convert_animation(self, custom_anim_file):
        """
        Convert your animation format to ZEGGS NPZ
        
        Expected output shape: (n_frames, 1141)
        - 1141 = joint parameters (positions + rotations)
        """
        # Load your custom format
        # Example: Load from JSON, CSV, or custom binary
        anim_data = self.load_custom_animation(custom_anim_file)
        
        # Extract joint parameters
        # Adjust this based on your skeleton format
        joint_params = self.extract_joint_parameters(anim_data)
        
        # Ensure shape is (n_frames, 1141)
        assert joint_params.shape[1] == 1141, f"Expected 1141 joints, got {joint_params.shape[1]}"
        
        return joint_params
    
    def convert_audio(self, custom_audio_file):
        """
        Convert audio to 16kHz mono WAV
        """
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(custom_audio_file)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return audio
    
    def process_dataset(self):
        """
        Process entire custom dataset
        """
        # Create output directory
        os.makedirs(f"{self.output_path}/gesture_npz", exist_ok=True)
        os.makedirs(f"{self.output_path}/normalize_audio_npz", exist_ok=True)
        
        # Iterate through dataset
        for clip_id, clip_data in enumerate(self.load_dataset()):
            name = f"speaker_{clip_id}_{clip_data['emotion']}_{clip_id:03d}"
            
            # Convert animation
            gesture = self.convert_animation(clip_data['animation_file'])
            np.savez(f"{self.output_path}/gesture_npz/{name}.npz", gesture=gesture)
            
            # Convert audio
            audio = self.convert_audio(clip_data['audio_file'])
            np.savez(f"{self.output_path}/normalize_audio_npz/{name}.npz", wav=audio)
            
            print(f"Processed: {name}")
```

#### 2. Compute Dataset Statistics

```python
def compute_statistics(processed_path):
    """
    Compute mean and std for normalization
    """
    all_gestures = []
    
    gesture_files = glob.glob(f"{processed_path}/gesture_npz/*.npz")
    for file in gesture_files:
        gesture = np.load(file)['gesture']
        all_gestures.append(gesture)
    
    # Concatenate all frames
    all_frames = np.concatenate(all_gestures, axis=0)  # (total_frames, 1141)
    
    # Compute statistics
    mean = all_frames.mean(axis=0)  # (1141,)
    std = all_frames.std(axis=0)    # (1141,)
    
    # Save
    np.savez(f"{processed_path}/mean.npz", mean=mean)
    np.savez(f"{processed_path}/std.npz", std=std)
    
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
```

#### 3. Create HDF5 Dataset

Follow the same process as ZEGGS:

```bash
cd ZeroEGGSProcessing
python data_to_h5dataset.py \
    --train_data_path=../custom_dataset/processed/train \
    --valid_data_path=../custom_dataset/processed/valid \
    --output_h5=../custom_dataset/h5dataset/datasets.h5 \
    --config=../main/configs/Custom.yml
```

#### 4. Train on Custom Dataset

```yaml
# configs/Custom.yml
train_h5: "../custom_dataset/h5dataset/datasets_train.h5"
valid_h5: "../custom_dataset/h5dataset/datasets_valid.h5"
gesture_mean: "../custom_dataset/processed/mean.npz"
gesture_std: "../custom_dataset/processed/std.npz"
```

```bash
python ohgesture.py --config=./configs/Custom.yml
```

---

## Debugging & Troubleshooting

### Common Issues

#### Issue 1: GPU Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   batch_size: 320  # Down from 640
   ```

2. **Enable gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   
   # In model forward():
   x_encoded = checkpoint(self.seqTransEncoder, x_latent)
   ```

3. **Use mixed precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

#### Issue 2: Training Loss Not Decreasing

**Symptom**: Loss plateaus or increases

**Debugging Steps**:

1. **Check data loading**:
   ```python
   # Verify data shapes
   for batch in train_loader:
       gesture, emotion, speech, text = batch
       print(f"Gesture: {gesture.shape}")
       print(f"Emotion: {emotion.shape}")
       print(f"Speech: {speech.shape}")
       print(f"Text: {text.shape}")
       break
   ```

2. **Verify normalization**:
   ```python
   mean = np.load('mean.npz')['mean']
   std = np.load('std.npz')['std']
   
   print(f"Mean range: {mean.min()} to {mean.max()}")
   print(f"Std range: {std.min()} to {std.max()}")
   ```

3. **Check learning rate**:
   ```yaml
   lr: 0.0001  # Try larger LR
   ```

4. **Monitor gradients**:
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.norm()}")
   ```

#### Issue 3: Generated Gestures Look Unnatural

**Symptom**: Jittery motion, unnatural poses

**Solutions**:

1. **Increase training iterations**:
   ```yaml
   epochs: 1000000  # Train longer
   ```

2. **Apply post-processing smoothing**:
   ```python
   from scipy.signal import savgol_filter
   
   def smooth_gesture(gesture, window=11, poly=3):
       """
       Apply Savitzky-Golay filter
       
       Args:
           gesture: (n_frames, 1141)
       """
       smoothed = savgol_filter(gesture, window, poly, axis=0)
       return smoothed
   ```

3. **Use seed gestures**:
   ```python
   # In sampling:
   seed_gesture = previous_clip[-8:]  # Last 8 frames
   
   output = model.sample(
       audio=audio,
       text=text,
       emotion=emotion,
       seed=seed_gesture
   )
   ```

4. **Tune diffusion steps**:
   ```python
   # Try more denoising steps
   output = diffusion.p_sample_loop(
       model=model,
       shape=shape,
       clip_denoised=True,
       model_kwargs=model_kwargs,
       skip_timesteps=0,  # Full diffusion (was 800)
       progress=True
   )
   ```

---

## Summary

This implementation guide provides:

1. ✅ Step-by-step instructions for common customizations
2. ✅ Code examples for each task
3. ✅ Configuration management
4. ✅ Debugging strategies
5. ✅ Best practices for extending the system

For more details, refer to:
- **Architecture**: `ARCHITECTURE.md`
- **Constitution**: `.specify/memory/constitution.md`
- **Project Summary**: `.specify/PROJECT_SUMMARY.md`
