# Contributing to FaceVox

Thanks for your interest in contributing! FaceVox is a real-time facial expression recognition system for assistive communication. Here's how to get started.

## Getting Started

1. Fork the repo
2. Clone your fork
3. Install dependencies: `pip install -r requirements.txt`
4. Create a branch: `git checkout -b your-feature`

## Development Setup

```bash
# Capture test data
py main.py train --mode capture --samples 50 --subject test

# Train a model
py main.py train --mode train --model-type transformer

# Run the demo
py main.py demo
```

## Project Structure

```
src/
  models/
    face_landmarks.py      # MediaPipe 478-point detection + normalization
    expression_recognition.py  # Classifiers (RF, GB, Transformer) + intent mapping
    landmark_transformer.py    # Transformer, TemporalTransformer, OcclusionAware models
    occlusion.py           # Occlusion detection and mask interpolation
  demo/
    webcam_demo.py         # Real-time webcam demo
    gui.py                 # PyQt5 GUI
    server.py              # FastAPI REST API
  utils/
    training.py            # Dataset capture, training pipeline
main.py                    # CLI entry point
colab_train.ipynb          # Colab training notebook
```

## How to Contribute

### Bug Reports
Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, and webcam model

### Feature Ideas
Open an issue first to discuss before implementing. Good first issues are tagged `good-first-issue`.

### Code Contributions
1. Create a feature branch
2. Make your changes
3. Test: `py main.py demo` (or at minimum verify imports work)
4. Submit a PR with a clear description

### Priority Areas
- **Demo GIF** — record a short clip showing the demo working live
- **New expression classes** — pain, fatigue, thirst, comfort signals
- **Mobile deployment** — TFLite or CoreML export
- **Model compression** — quantization for edge devices
- **Benchmarks** — test against AffectNet, FER2013 datasets
- **Tests** — unit tests for models, training pipeline, and API

## Code Style
- No AI-generated comments (keep it human-readable)
- Follow existing patterns in the codebase
- Keep functions focused and short

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
