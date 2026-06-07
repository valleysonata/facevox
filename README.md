
# FaceVox (Work In Progress)

Facial expression recognition for assistive communication. Detects expressions via 478-point face landmarks (MediaPipe) and maps them to communication intents (YES/NO/HELP/PAIN) for users with motor disabilities.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Capture training data from webcam
py main.py train --mode capture --samples 100 --subject me

# 2. Train the transformer model
py main.py train --mode train --model-type transformer

# 3. Run the demo
py main.py demo
```

## Usage

```bash
# Capture data for multiple people
py main.py train --mode capture --samples 100 --subject me
py main.py train --mode capture --samples 100 --subject brother

# Train models
py main.py train --mode train                        # RandomForest (fast, ~60%)
py main.py train --mode train --model-type transformer  # Transformer (best, ~99% val)
py main.py train --mode train --model-type temporal    # Temporal (smoother predictions)
py main.py train --mode train --model-type occlusion_aware  # Occlusion-aware attention

# Run demos
py main.py demo     # Webcam demo
py main.py server   # REST API
```

## Models

| Model | Type | Input | Val Accuracy | Notes |
|-------|------|-------|-------------|-------|
| RandomForest | ML | 10 geometric features | ~60% | Fast, no GPU needed |
| GradientBoosting | ML | 10 geometric features | ~65% | Fast, no GPU needed |
| LandmarkTransformer | DL | 478x3 normalized landmarks | ~99% | Best accuracy |
| TemporalTransformer | DL | 15-frame landmark sequences | ~99% | Smooth predictions |
| OcclusionAwareClassifier | DL | 478x3 + attention mask | ~99% | Handles mask/hand occlusion |

## Architecture

- **Face Detection**: MediaPipe FaceLandmarker (478 landmarks, 3D)
- **Landmark Normalization**: Nose-centered, inter-eye distance scaling for scale/position invariant inference
- **Spatial Transformer**: 3-layer Transformer encoder (128-dim, 4 heads) on per-frame landmarks
- **Temporal Transformer**: Cross-frame attention for smooth real-time predictions
- **Occlusion Attention**: Gated self-attention that learns to ignore occluded landmarks (mask, hand, etc.)
- **Feature Extraction**: 10 geometric features (mouth, eyes, brows, head pose) for ML models
- **Intent Mapping**: Expression → assistive communication intent
- **Occlusion Handling**: Z-depth symmetry-based mask interpolation

## Expression Mapping

| Expression | Intent |
|------------|--------|
| Happy | YES |
| Sad | NO |
| Surprised | HELP |
| Angry | PAIN |

## Colab Training

For GPU-accelerated training, upload `colab_train.ipynb` to Google Colab:

1. Upload your `data/dataset_*.json` files to Colab
2. Run all cells
3. Download the `.pt` model from `checkpoints/`
4. Place it in your local `checkpoints/` folder

## API

```bash
# Start server
py main.py server

# POST /predict with image bytes
curl -X POST http://localhost:8000/predict -F "image=@photo.jpg"
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

