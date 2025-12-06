# OHGesture Project Constitution

**Version**: 1.0.0 | **Ratified**: December 6, 2025 | **Last Amended**: December 6, 2025

---

## Core Principles

### I. Research-First Development

OHGesture is a research project focused on advancing conversational gesture synthesis. All development must maintain:

- **Reproducibility**: All experiments must be reproducible with fixed random seeds
- **Documentation**: Research decisions, hyperparameters, and architecture changes must be documented
- **Baseline Preservation**: Maintain compatibility with original DiffuseStyleGesture baseline
- **Scientific Rigor**: Changes must be validated through controlled experiments

### II. Data Pipeline Integrity

The data preprocessing pipeline is critical and must maintain:

- **Deterministic Processing**: Same input data must always produce identical outputs
- **Normalization Consistency**: Mean/std statistics must be computed once and reused
- **Frame Alignment**: Audio (16kHz) and gesture (20 FPS) must remain synchronized
- **Format Standardization**: Use HDF5 for training datasets, NPZ for intermediate processing

### III. Model Architecture Modularity

The DeepGesture model must remain modular with swappable components:

- **Encoder Independence**: Audio, text, and style encoders must be independently replaceable
- **Attention Mechanisms**: Cross-attention modules should be configurable via config files
- **Feature Dimensions**: All feature dimensions must be clearly specified in configs
- **Backward Compatibility**: Model checkpoints must load across code updates

### IV. Configuration-Driven Design

All hyperparameters and paths must be externalized:

- **YAML-Based**: Use YAML config files (e.g., `OHGesture.yml`)
- **No Hardcoding**: No hardcoded values in core training/inference scripts
- **Runtime Validation**: Check for required config keys at startup
- **Documentation**: Every config parameter must have inline comments explaining purpose

### V. Training Stability & Monitoring

Training must be stable, monitored, and resumable:

- **Gradient Handling**: Implement gradient clipping to prevent explosions
- **Mixed Precision**: Support FP16 for large batch sizes (24GB+ VRAM)
- **Checkpointing**: Save model every N epochs (default: 25)
- **Logging**: TensorBoard logging for loss, gradients, and sample outputs
- **Resumability**: Training must be resumable from any checkpoint without data loss

### VI. Code Quality Standards

All code must meet professional standards:

- **Type Hints**: Use Python type annotations where applicable
- **Docstrings**: All classes and public methods must have descriptive docstrings
- **Error Handling**: Provide graceful error messages for common failures (missing files, GPU OOM)
- **Linting**: Follow PEP 8 style guidelines
- **Code Hygiene**: Remove unused imports and commented-out code before commits

## Technical Constraints

### Hardware Requirements

**Training Environment**:
- Minimum: 1x GPU with 24GB VRAM (RTX 3090, A5000)
- Recommended: 1x GPU with 40GB+ VRAM (A100, A6000)
- CPU: 16+ cores for data loading
- RAM: 32GB+ system memory
- Storage: 100GB+ SSD for datasets and checkpoints

**Inference Environment**:
- Minimum: 1x GPU with 8GB VRAM (RTX 2080)
- CPU: 8+ cores
- RAM: 16GB+ system memory

### Software Dependencies

**Python Version**: Python 3.11 (strict requirement)

**Core Dependencies**:
- PyTorch 2.3.1+ (CUDA 11.8 or 12.1)
- NumPy 1.26.4 (exact version for compatibility)
- Transformers (for WavLM)
- Librosa 0.10.2+ (audio processing)

### Performance Standards

**Training Performance**:
- Target: 500K iterations in ≤2 weeks on single A100
- Batch size: 640 (adjust based on available VRAM)
- No training bottleneck from data loading

**Inference Performance**:
- Target: <2 seconds per 4.4-second gesture clip (GPU)
- Quality: Generated gestures must pass visual inspection

## Dataset Conventions

### ZEGGS Dataset Structure

Required directory structure for preprocessing:
```
ZeroEGGSProcessing/
├── data/
│   ├── train/              # Training split
│   │   ├── audio/          # Original WAV files
│   │   ├── bvh/            # BVH animation files (60 FPS)
│   │   └── textgrid/       # Word-level transcripts
│   └── valid/              # Validation split
├── processed/
│   ├── mean.npz            # Gesture normalization mean
│   └── std.npz             # Gesture normalization std
└── h5dataset/
    ├── datasets_train.h5   # Training HDF5 dataset
    └── datasets_valid.h5   # Validation HDF5 dataset
```

### File Naming Convention

Format: `{speaker}_{emotion}_{clip_id}`

Example: `speaker_1_Happy_001`

Valid emotions: `Happy`, `Sad`, `Neutral`, `Old`, `Angry`, `Relaxed`

## Model Checkpointing Policy

### Checkpoint Naming

Format: `model{iteration:09d}.pt`

Example: `model000450000.pt` (450K iterations)

### Retention Policy

- Save every 25 epochs during training
- Keep checkpoints at milestones: 100K, 250K, 500K iterations
- Keep best model (lowest validation loss)
- Clean up intermediate checkpoints older than 1 month (preserve milestones)

## Reproducibility Requirements

### Random Seed Management

All training/inference scripts must support `--seed` argument:

```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

### Experiment Tracking

For each experiment, record:
- Git commit hash
- Full config file (YAML)
- Random seed
- Hardware specifications (GPU model, CUDA version)
- Training duration
- Final metrics

### Result Reproducibility

To claim reproducible results, must use:
- Same code (git hash)
- Same data (dataset version)
- Same seed
- Same hardware class

Results must be within ±5% variance.

## Governance

### Constitution Authority

This constitution supersedes all other coding practices and guidelines.

### Amendment Process

To amend this constitution:
1. Propose change via GitHub issue
2. Discuss with team members  
3. Requires approval from at least 2 core maintainers
4. Update version number and "Last Amended" date
5. Document rationale for change in issue

### Compliance

- All pull requests must verify compliance with this constitution
- Reviewers must check for violations before approval
- Violations must be fixed before merging
- This is a living document and will evolve with the project
