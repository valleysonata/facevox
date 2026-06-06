
# FaceVox (Work In Progress)

Real-time facial expression recognition for assistive communication. Detects expressions via 478-point face landmarks (MediaPipe) and maps them to communication intents (YES/NO/HELP/PAIN) for users with motor disabilities.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Capture training data from webcam
py main.py train --mode capture --samples 50

# Capture data for a specific person
py main.py train --mode capture --samples 100 --subject me
py main.py train --mode capture --samples 100 --subject brother

# Train RF/GB model
py main.py train --mode train

# Train transformer model (deep learning, better accuracy)
py main.py train --mode train --model-type transformer

# Run webcam demo (auto-loads best available model)
py main.py demo

# Run GUI
py main.py gui

# Start API server
py main.py server
```

## Models

| Model | Type | Input | Accuracy |
|-------|------|-------|----------|
| RandomForest | ML | 10 geometric features | ~60% |
| GradientBoosting | ML | 10 geometric features | ~65% |
| LandmarkTransformer | DL | 478x3 landmark coordinates | ~85%+ |

The LandmarkTransformer uses a 3-layer Transformer encoder (128-dim, 4 heads) on raw 478-point face landmarks for significantly better accuracy.

## Architecture

- **Face Detection**: MediaPipe FaceLandmarker (478 landmarks, 3D)
- **Feature Extraction**: Geometric features (mouth, eyes, brows, head pose)
- **Classification**: Transformer encoder / RandomForest / GradientBoosting
- **Intent Mapping**: Expression → assistive communication intent
- **Occlusion Handling**: Z-depth symmetry-based mask interpolation

## Expression Mapping

| Expression | Intent |
|------------|--------|
| Happy | YES |
| Sad | NO |
| Surprised | HELP |
| Angry | PAIN |

## API

```bash
# Start server
py main.py server

# POST /predict with image bytes
curl -X POST http://localhost:8000/predict -F "image=@photo.jpg"
```
