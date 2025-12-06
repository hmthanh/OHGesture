# OHGesture: Task Implementation Templates

**Purpose**: Ready-to-use templates for common implementation tasks  
**Last Updated**: December 6, 2025

---

## Table of Contents

1. [New Feature Template](#new-feature-template)
2. [New Audio Encoder Template](#new-audio-encoder-template)
3. [New Attention Mechanism Template](#new-attention-mechanism-template)
4. [Custom Dataset Template](#custom-dataset-template)
5. [Evaluation Metric Template](#evaluation-metric-template)
6. [Config File Template](#config-file-template)

---

## New Feature Template

### Feature: [Feature Name]

**Goal**: [One sentence description]

**Affected Components**:
- [ ] Data preprocessing
- [ ] Model architecture
- [ ] Training loop
- [ ] Inference pipeline
- [ ] Configuration

### Implementation Checklist

#### Phase 1: Planning
- [ ] Review `constitution.md` for constraints
- [ ] Design feature architecture
- [ ] Identify affected modules
- [ ] Update config schema

#### Phase 2: Data Pipeline (if needed)
- [ ] Modify `ZeroEGGSProcessing/*.py`
- [ ] Update HDF5 structure
- [ ] Recompute normalization stats
- [ ] Test on small dataset

#### Phase 3: Model Changes
- [ ] Create new module in `main/model/` or `model/`
- [ ] Update `DeepGesture` class
- [ ] Add forward pass logic
- [ ] Test forward/backward pass

#### Phase 4: Training
- [ ] Update training loop if needed
- [ ] Add logging for new feature
- [ ] Train baseline (1K iterations)
- [ ] Monitor loss curves

#### Phase 5: Inference
- [ ] Update `sampling.py`
- [ ] Test generation
- [ ] Verify BVH output
- [ ] Visualize in Unity

#### Phase 6: Documentation
- [ ] Update `ARCHITECTURE.md`
- [ ] Add example to `IMPLEMENTATION_GUIDE.md`
- [ ] Update config templates
- [ ] Document hyperparameters

### Code Template

```python
# File: main/model/[feature_name].py

import torch
import torch.nn as nn

class [FeatureName](nn.Module):
    """
    [Brief description]
    
    Args:
        [arg1]: [description]
        [arg2]: [description]
    """
    def __init__(self, [args]):
        super().__init__()
        
        # Initialize components
        self.[component] = nn.Linear([in_dim], [out_dim])
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: ([shape]) - [description]
        
        Returns:
            output: ([shape]) - [description]
        """
        # Implementation
        output = self.[component](x)
        return output

# Integration in DeepGesture
# File: main/model/deepgesture.py

class DeepGesture(nn.Module):
    def __init__(self, ..., use_[feature]=False, **kargs):
        super().__init__()
        
        if use_[feature]:
            self.[feature] = [FeatureName]([args])
    
    def forward(self, x, timesteps, y=None):
        # ...existing code...
        
        if hasattr(self, '[feature]'):
            x = self.[feature](x)
        
        # ...rest of forward pass...
```

### Config Template

```yaml
# configs/OHGesture_[feature].yml

# Feature configuration
use_[feature]: true
[feature]_param1: value1
[feature]_param2: value2
```

### Testing Script

```bash
# Test feature in isolation
cd main
python -c "
from model.[feature_name] import [FeatureName]
import torch

# Test instantiation
module = [FeatureName]([args])
print(f'Module created: {module}')

# Test forward pass
x = torch.randn([input_shape])
output = module(x)
print(f'Output shape: {output.shape}')
print('Test passed!')
"

# Test in full pipeline
python ohgesture.py \
    --config=./configs/OHGesture_[feature].yml \
    --gpu cuda:0 \
    --epochs 10  # Short test run
```

---

## New Audio Encoder Template

### Encoder: [Encoder Name]

**Input**: Audio waveform (16kHz mono)  
**Output**: Feature sequence (n_frames, feature_dim)

### Implementation

```python
# File: wavlm/[encoder_name]_encoder.py

import torch
import torch.nn as nn

class [EncoderName]Encoder(nn.Module):
    """
    [Description of encoder]
    
    Architecture:
        [Brief architecture description]
    
    Args:
        sample_rate: Audio sample rate (default: 16000)
        output_dim: Output feature dimension
    """
    def __init__(self, sample_rate=16000, output_dim=64):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.output_dim = output_dim
        
        # Define layers
        # Example: Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, output_dim, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
    
    def forward(self, waveform, target_length):
        """
        Extract features from waveform
        
        Args:
            waveform: (B, audio_samples) - Raw audio
            target_length: int - Target sequence length
        
        Returns:
            features: (B, target_length, output_dim)
        """
        # Add channel dimension
        x = waveform.unsqueeze(1)  # (B, 1, audio_samples)
        
        # Apply convolutions
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # (B, output_dim, T)
        
        # Interpolate to target length
        x = torch.nn.functional.interpolate(
            x,
            size=target_length,
            mode='linear',
            align_corners=True
        )
        
        # Permute to (B, T, C)
        x = x.permute(0, 2, 1)  # (B, target_length, output_dim)
        
        return x
```

### Integration

```python
# File: ZeroEGGSProcessing/data_to_h5dataset.py

from wavlm.[encoder_name]_encoder import [EncoderName]Encoder

class DeepGesturePreprocessor:
    def __init__(self, args, ...):
        if args.audio_encoder == '[encoder_name]':
            self.audio_encoder = [EncoderName]Encoder(
                sample_rate=16000,
                output_dim=args.audio_feat_dim
            ).to(self.device)
        elif args.audio_encoder == 'wavlm':
            self.audio_encoder = wavlm_init(args, self.device)
```

### Config

```yaml
audio_encoder: "[encoder_name]"
audio_feat: "[encoder_name]"
audio_feat_dim: 64
```

---

## New Attention Mechanism Template

### Mechanism: [Mechanism Name]

**Purpose**: [Description of what this attention does]

### Implementation

```python
# File: main/model/attention/[mechanism_name].py

import torch
import torch.nn as nn
import torch.nn.functional as F

class [MechanismName]Attention(nn.Module):
    """
    [Detailed description]
    
    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, T_q, dim)
            key: (B, T_k, dim)
            value: (B, T_v, dim)
            mask: (B, T_q, T_k) or None
        
        Returns:
            output: (B, T_q, dim)
            attn_weights: (B, num_heads, T_q, T_k)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        # Project and reshape
        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights
```

### Integration

```python
# File: main/model/deepgesture.py

from model.attention.[mechanism_name] import [MechanismName]Attention

class DeepGesture(nn.Module):
    def __init__(self, ..., cond_mode='[mechanism_name]', **kargs):
        super().__init__()
        
        if '[mechanism_name]' in cond_mode:
            self.[mechanism_name]_attention = [MechanismName]Attention(
                dim=self.latent_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
    
    def forward(self, x, timesteps, y=None):
        # ...encoding...
        
        if hasattr(self, '[mechanism_name]_attention'):
            # Apply attention
            attended, attn_weights = self.[mechanism_name]_attention(
                query=gesture_features,
                key=audio_features,
                value=audio_features
            )
            
            # Residual connection
            gesture_features = gesture_features + attended
        
        # ...rest of forward...
```

---

## Custom Dataset Template

### Dataset: [Dataset Name]

**Source**: [Where data comes from]  
**Format**: [Original format]  
**Size**: [Number of clips, duration]

### Conversion Script

```python
# File: data_processing/convert_[dataset_name].py

import os
import glob
import numpy as np
import soundfile as sf
from tqdm import tqdm

class [DatasetName]Converter:
    """
    Convert [Dataset Name] to ZEGGS-compatible format
    """
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
        # Create output directories
        os.makedirs(f"{output_path}/gesture_npz", exist_ok=True)
        os.makedirs(f"{output_path}/normalize_audio_npz", exist_ok=True)
        os.makedirs(f"{output_path}/embedding", exist_ok=True)
    
    def convert_animation(self, anim_file):
        """
        Convert animation to (n_frames, 1141) format
        
        Adapt this based on your skeleton format
        """
        # Load your animation format
        # ...custom loading code...
        
        # Extract joint parameters
        # Ensure output shape is (n_frames, 1141)
        joint_params = ...
        
        assert joint_params.shape[1] == 1141, \
            f"Expected 1141 joints, got {joint_params.shape[1]}"
        
        return joint_params
    
    def convert_audio(self, audio_file):
        """Convert to 16kHz mono WAV"""
        audio, sr = sf.read(audio_file)
        
        # Resample if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return audio
    
    def extract_emotion(self, filename):
        """
        Extract emotion label from filename or metadata
        
        Return one of: Happy, Sad, Neutral, Old, Angry, Relaxed
        """
        # Custom logic to extract emotion
        emotion = ...
        
        return emotion
    
    def process_all(self):
        """Process entire dataset"""
        # Get all files
        files = glob.glob(f"{self.input_path}/**/*.[extension]", recursive=True)
        
        for i, file in enumerate(tqdm(files)):
            # Extract metadata
            emotion = self.extract_emotion(file)
            
            # Create name
            name = f"speaker_{i}_{emotion}_{i:03d}"
            
            # Convert animation
            gesture = self.convert_animation(file)
            np.savez(f"{self.output_path}/gesture_npz/{name}.npz", gesture=gesture)
            
            # Convert audio
            audio_file = ...  # Get corresponding audio file
            audio = self.convert_audio(audio_file)
            np.savez(f"{self.output_path}/normalize_audio_npz/{name}.npz", wav=audio)
            
            print(f"Processed: {name}")

# Usage
if __name__ == '__main__':
    converter = [DatasetName]Converter(
        input_path='./raw_data/[dataset_name]',
        output_path='./processed/[dataset_name]'
    )
    converter.process_all()
```

### Statistics Computation

```python
# File: data_processing/compute_stats_[dataset_name].py

import numpy as np
import glob
from tqdm import tqdm

def compute_statistics(processed_path):
    """Compute mean and std for normalization"""
    all_gestures = []
    
    # Load all gesture files
    files = glob.glob(f"{processed_path}/gesture_npz/*.npz")
    for file in tqdm(files, desc="Loading gestures"):
        gesture = np.load(file)['gesture']
        all_gestures.append(gesture)
    
    # Concatenate all frames
    all_frames = np.concatenate(all_gestures, axis=0)
    print(f"Total frames: {all_frames.shape[0]}")
    
    # Compute statistics
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    
    # Clip std to avoid division by zero
    std = np.clip(std, a_min=0.01, a_max=None)
    
    # Save
    np.savez(f"{processed_path}/mean.npz", mean=mean)
    np.savez(f"{processed_path}/std.npz", std=std)
    
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")

if __name__ == '__main__':
    compute_statistics('./processed/[dataset_name]')
```

### Config

```yaml
# configs/[DatasetName].yml

name: "[DatasetName]"

# Data paths
processing_train_data_path: "./processed/[dataset_name]/train/"
processing_val_data_path: "./processed/[dataset_name]/valid/"

train_h5: "./h5dataset/[dataset_name]_train.h5"
valid_h5: "./h5dataset/[dataset_name]_valid.h5"

gesture_mean: "./processed/[dataset_name]/mean.npz"
gesture_std: "./processed/[dataset_name]/std.npz"

# Rest same as OHGesture.yml
```

---

## Evaluation Metric Template

### Metric: [Metric Name]

**Purpose**: [What does this metric measure]

### Implementation

```python
# File: eval/metrics/[metric_name].py

import torch
import numpy as np

def compute_[metric_name](predicted_gestures, ground_truth_gestures):
    """
    Compute [Metric Name]
    
    Args:
        predicted_gestures: (B, n_frames, 1141) - Generated gestures
        ground_truth_gestures: (B, n_frames, 1141) - Ground truth
    
    Returns:
        metric_value: float - [Metric Name] score
    """
    # Convert to tensors if needed
    if isinstance(predicted_gestures, np.ndarray):
        predicted_gestures = torch.from_numpy(predicted_gestures)
    if isinstance(ground_truth_gestures, np.ndarray):
        ground_truth_gestures = torch.from_numpy(ground_truth_gestures)
    
    # Compute metric
    # Example: Mean squared error
    mse = torch.mean((predicted_gestures - ground_truth_gestures) ** 2)
    
    return mse.item()

class [MetricName]Evaluator:
    """
    Evaluator class for [Metric Name]
    """
    def __init__(self):
        self.scores = []
    
    def add_batch(self, predicted, ground_truth):
        """Add a batch of predictions"""
        score = compute_[metric_name](predicted, ground_truth)
        self.scores.append(score)
    
    def get_average(self):
        """Get average score across all batches"""
        return np.mean(self.scores)
    
    def reset(self):
        """Reset scores"""
        self.scores = []
```

### Evaluation Script

```python
# File: eval/evaluate_[metric_name].py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.[metric_name] import [MetricName]Evaluator
from main.data_loader.deepgesture_dataset import DeepGestureDataset

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set using [Metric Name]
    """
    evaluator = [MetricName]Evaluator()
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            gesture, emotion, speech, text = batch
            
            # Move to device
            gesture = gesture.to(device)
            emotion = emotion.to(device)
            speech = speech.to(device)
            text = text.to(device)
            
            # Generate prediction
            predicted = model.sample(
                audio=speech,
                text=text,
                emotion=emotion,
                n_frames=gesture.shape[1]
            )
            
            # Compute metric
            evaluator.add_batch(predicted.cpu(), gesture.cpu())
    
    # Get final score
    avg_score = evaluator.get_average()
    print(f"[Metric Name]: {avg_score:.4f}")
    
    return avg_score

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # Load model
    from main.model.deepgesture import DeepGesture
    model = DeepGesture(...)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(args.device)
    
    # Load test data
    test_dataset = DeepGestureDataset(args.test_h5, ...)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    score = evaluate_model(model, test_loader, args.device)
```

---

## Config File Template

```yaml
# configs/[ConfigName].yml

# ========================
# Dataset Configuration
# ========================

name: "[ExperimentName]"

# Data paths
processing_train_data_path: "[path_to_processed_train]"
processing_val_data_path: "[path_to_processed_val]"

train_h5: "[path_to_train_h5]"
valid_h5: "[path_to_valid_h5]"

# Normalization
gesture_mean: "[path_to_mean.npz]"
gesture_std: "[path_to_std.npz]"

# ========================
# Feature Configuration
# ========================

# Audio
audio_feat: "wavlm"               # Options: wavlm, mfcc, custom
wavlm_model_path: "../wavlm/WavLM-Large.pt"
audio_feat_dim: 64                # Audio feature dimension

# Text
text_feat: "word2vec"             # Options: word2vec, bert
text_feat_dim: 300                # Text feature dimension

# Emotion
n_emotions: 6                     # Number of emotion classes

# ========================
# Model Configuration
# ========================

# Architecture
arch: "trans_enc"                 # Options: trans_enc, trans_dec, gru
cond_mode: "cross_local_attention3_style1"
latent_dim: 256
ff_size: 1024
num_layers: 8
num_heads: 4
dropout: 0.1
activation: "gelu"

# Conditioning
n_seed: 8                         # Seed gesture frames
cond_mask_prob: 0.1               # Classifier-free guidance prob

# ========================
# Training Configuration
# ========================

# Data
n_poses: 88                       # Sequence length
motion_resampling_framerate: 20   # Target FPS
subdivision_stride: 10            # Window stride
batch_size: 640                   # Batch size

# Optimization
lr: 0.00003                       # Learning rate
betas: [0.5, 0.999]               # Adam betas
weight_decay: 0.0                 # Weight decay
lr_anneal_steps: 0                # LR annealing steps

# Training loop
epochs: 500000                    # Total epochs
save_per_epochs: 25               # Save frequency
log_interval: 50                  # Log frequency

# Output
save_dir: "./output/checkpoint/[experiment_name]"

# ========================
# Data Loading
# ========================

loader_workers: 0                 # Number of workers

# ========================
# Diffusion Configuration
# ========================

diffusion_steps: 1000             # Total diffusion steps
noise_schedule: "linear"          # Options: linear, cosine
beta_start: 0.0001
beta_end: 0.02

# ========================
# Inference Configuration
# ========================

sampling_steps: 1000              # Sampling steps (can be < diffusion_steps for DDIM)
clip_denoised: true               # Clip predictions to valid range
```

---

## Summary

These templates provide:

1. ✅ **Structured approach** to implementing new features
2. ✅ **Code scaffolding** for common tasks
3. ✅ **Integration patterns** with existing codebase
4. ✅ **Configuration templates** for new experiments
5. ✅ **Testing strategies** for validation

For detailed examples, see `IMPLEMENTATION_GUIDE.md`.

For project standards, see `constitution.md`.
