# FaceVox

A Python prototype for facial-expression-based communication. MediaPipe extracts
478 face landmarks; geometric features or normalized landmarks feed an expression
classifier, and a separate mapper produces YES, NO, HELP and PAIN intents. The
webcam demo requires an intent to persist across several frames before confirming
it.

The project includes Random Forest and Gradient Boosting classifiers, experimental
PyTorch models, an OpenCV webcam demo, a Qt interface and a FastAPI server.

## Setup

Python 3.12 is used for the local smoke tests. From the repository root:

```sh
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

The first webcam or API run downloads MediaPipe's face-landmarker model into
`checkpoints/`. Captured datasets and trained checkpoints stay local and are
excluded from Git.

## Run

```sh
# Webcam demo; q quits, r resets intent confirmation
python main.py demo

# Select a camera and a trained classifier
python main.py demo --camera 0 --model checkpoints/expression_model.joblib

# Optional Qt interface
python -m pip install -r requirements-gui.txt
python main.py gui

# Local API; interactive documentation at http://127.0.0.1:8000/docs
python main.py server --host 127.0.0.1 --port 8000
```

Without a trained checkpoint, the classical classifier uses hand-written rules.
The webcam view is mirrored by default; use `--no-mirror` to disable it.
`--confirm-frames` and `--min-confidence` control intent confirmation.

## Capture and train

```sh
# Record from a webcam; use a pseudonymous subject identifier
python main.py train --mode capture --subject participant01 --samples 50

# Train on dataset*.json files in data/
python main.py train --model-type rf

# Exercise the pipeline with generated data, without a camera
python main.py train --synthetic --samples 50 --model-type rf
```

Other model choices are `gb`, `transformer`, `temporal` and `occlusion_aware`.
Classical models use geometric features; PyTorch models require raw landmarks.
The [Colab notebook](colab_train.ipynb) uses the same training code as the CLI.
Run `python main.py train --help` for checkpoint and capture options.

## Tests

```sh
python tests/test_smoke.py
```

The 14 smoke tests run without a camera, model download or network access. They
cover feature normalization, intent confirmation, model training and persistence,
PyTorch output shapes, API routes and CLI help.

## Evaluation

Synthetic data is for pipeline checks, not evidence of recognition accuracy on
real people. Training currently uses a random sample split; PyTorch also uses that
split for checkpoint selection. A subject-independent evaluation has not been
established. The intent labels are interaction mappings, not measurements of a
person's needs or a clinically validated communication method.

## Code layout

| Path | Purpose |
| --- | --- |
| `main.py` | CLI entry point |
| `src/models/face_landmarks.py` | MediaPipe detection and landmark normalization |
| `src/models/expression_recognition.py` | Classifiers and intent confirmation |
| `src/models/landmark_transformer.py` | Experimental PyTorch architectures |
| `src/models/occlusion.py` | Occlusion detection and landmark adaptation |
| `src/utils/training.py` | Dataset capture, training and evaluation |
| `src/demo/` | Webcam, Qt and API interfaces |
| `tests/test_smoke.py` | Camera-free regression checks |

Licensed under the [MIT License](LICENSE).
