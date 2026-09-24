# FaceVox (Work In Progress)

Real-time facial expression recognition for assistive communication. Detects expressions via 478-point face landmarks (MediaPipe) and maps them to communication intents (YES/NO/HELP/PAIN) for users with motor disabilities.

> **Status:** research prototype. The classical ML path (`rf`/`gb`) trains and runs end-to-end. The deep-learning paths (`transformer`/`temporal`/`occlusion_aware`) train end-to-end but need real captured landmark data to perform well — accuracy figures below are rough results on synthetic data, not benchmarks on real faces.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Train with synthetic data (fastest way to verify the pipeline works)
python main.py train --synthetic --samples 50 --model-type rf

# 2. Train a transformer model on synthetic raw landmarks
python main.py train --synthetic --samples 50 --model-type transformer

# 3. Run the webcam demo (rule-based fallback if no model is trained)
python main.py demo --model-type rf

# 4. Start REST API server
python main.py server

# 5. Run GUI
python main.py gui
```

## Usage

```bash
# Capture real training data from webcam (stores 10 geometric features
# plus raw 478x3 landmarks per sample, so all model types can train on it)
python main.py train --mode capture --samples 100 --subject me

# Train different model types on synthetic data (pipeline smoke test)
python main.py train --synthetic --samples 50 --model-type rf
python main.py train --synthetic --samples 50 --model-type transformer
python main.py train --synthetic --samples 50 --model-type temporal
python main.py train --synthetic --samples 50 --model-type occlusion_aware

# Train on your captured data (merges every data/dataset*.json file)
python main.py train --mode train --model-type rf
python main.py train --mode train --model-type transformer

# Evaluate a saved model without retraining (requires --model-path)
python main.py train --mode evaluate --model-type rf --model-path checkpoints/expression_model.joblib

# Run demos
python main.py demo                    # Webcam demo
python main.py gui                     # PyQt5 GUI
python main.py server                  # REST API on :8000
```

## Models

| Model | Type | Input | Notes |
|-------|------|-------|-------|
| RandomForest (`rf`) | ML | 10 geometric features | Fast, no GPU needed |
| GradientBoosting (`gb`) | ML | 10 geometric features | Fast, no GPU needed |
| LandmarkTransformer (`transformer`) | DL | 1434-dim flattened landmarks (478x3) | Needs raw-landmark data |
| TemporalTransformer (`temporal`) | DL | 15-frame sequences of 1434-dim frames | Needs raw-landmark data |
| OcclusionAwareClassifier (`occlusion_aware`) | DL | 1434-dim flattened landmarks + visibility | Needs raw-landmark data |

## Models

| Model | Type | Input | Speed | Notes |
|-------|------|-------|-------|-------|
| RandomForest | ML | 10 features | Fast | No GPU needed |
| GradientBoosting | ML | 10 features | Fast | No GPU needed |
| LandmarkTransformer | DL | 1434 features | Medium | Best accuracy |
| TemporalTransformer | DL | 15-frame sequences | Medium | Smoother predictions |
| OcclusionAwareClassifier | DL | 478 landmarks | Slow | Handles occlusions |

## Expression → Intent Mapping

| Expression | Intent |
|------------|--------|
| Happy | YES |
| Sad | NO |
| Surprised | HELP |
| Angry | PAIN |
| Fearful | HELP |

## Architecture

- **Face Detection**: MediaPipe FaceLandmarker (478 landmarks, 3D)
- **Landmark Normalization**: Nose-centered, inter-eye distance scaling
- **Spatial Transformer**: 3-layer Transformer encoder on per-frame landmarks
- **Temporal Transformer**: Cross-frame attention for smooth real-time predictions
- **Occlusion Attention**: Gated self-attention that learns to ignore occluded landmarks
- **Feature Extraction**: 10 geometric features for ML models
- **Intent Mapping**: Expression → assistive communication intent

## API

```bash
# Start server
python main.py server

# POST /predict with a JSON body containing a base64-encoded image
python -c "import base64, json, urllib.request; img = base64.b64encode(open('photo.jpg','rb').read()).decode(); req = urllib.request.Request('http://localhost:8000/predict', data=json.dumps({'image_base64': img}).encode(), headers={'Content-Type': 'application/json'}); print(urllib.request.urlopen(req).read().decode())"

# POST /predict/batch with multiple base64 images
# GET expressions list
curl http://localhost:8000/expressions

# Health check
curl http://localhost:8000/health
```

## Project Structure

```
facevox/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── face_landmarks.py      # MediaPipe 478-point detection
│   │   ├── expression_recognition.py  # Classifiers + intent mapping
│   │   ├── landmark_transformer.py    # PyTorch Transformer models
│   │   └── occlusion.py             # Occlusion detection/handling
│   ├── demo/
│   │   ├── __init__.py
│   │   ├── webcam_demo.py        # Real-time webcam demo
│   │   ├── gui.py                # PyQt5 GUI
│   │   └── server.py             # FastAPI REST API
│   └── utils/
│       ├── __init__.py
│       └── training.py           # Dataset capture + training pipeline
├── checkpoints/               # Saved models
├── data/                      # Training data
├── assets/                    # Logo, banner images
├── generate_assets.py         # Asset generation script
└── colab_train.ipynb          # Google Colab training notebook
```

## Colab Training

For GPU-accelerated training, upload `colab_train.ipynb` to Google Colab:

1. Upload your `dataset_*.json` files to Colab
2. Run all cells
3. Download the `.pt` model from `checkpoints/`
4. Place it in your local `checkpoints/` folder

## Contributing

Contributions welcome! See `CONTRIBUTING.md` for guidelines.

## License

MIT License
