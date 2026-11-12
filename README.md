# FaceVox

A Python prototype for facial-expression-based communication. MediaPipe extracts
478 face landmarks; geometric features or normalized landmarks feed an expression
classifier, and a separate mapper produces YES, NO, HELP and PAIN intents. The
webcam demo requires an intent to persist across several frames before confirming
it.

The project includes Random Forest and Gradient Boosting classifiers, experimental
PyTorch models, an OpenCV webcam demo, a Qt interface and a FastAPI server.

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
