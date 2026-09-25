# FaceVox (Work In Progress)

Real-time facial expression recognition for assistive communication. Detects expressions via 478-point face landmarks (MediaPipe) and maps them to communication intents (YES/NO/HELP/PAIN) for users with motor disabilities.

**Status:** research prototype. The classical ML path (`rf`/`gb`) trains and runs end-to-end. The deep-learning paths (`transformer`/`temporal`/`occlusion_aware`) train end-to-end but need real captured landmark data to perform well — accuracy figures below are rough results on synthetic data, not benchmarks on real faces.

## Structure

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



MIT License
