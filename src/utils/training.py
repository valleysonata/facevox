import os
import json
import time
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import mediapipe as mp

from src.models.face_landmarks import MediaPipeFaceLandmarker, normalize_landmarks
from src.models.expression_recognition import (
    ExpressionClassifier,
    ExpressionLabel,
    FeatureExtractor,
    create_classifier,
)

def _normalize_raw_landmarks(raw_list):
    lm = np.array(raw_list, dtype=np.float32).reshape(478, 3)
    normed = normalize_landmarks(lm)
    return normed.flatten().tolist()

ASSISTIVE_LABELS = {
    ExpressionLabel.NEUTRAL: 0,
    ExpressionLabel.HAPPY: 1,    # YES
    ExpressionLabel.SAD: 2,      # NO
    ExpressionLabel.SURPRISED: 3,  # HELP
    ExpressionLabel.ANGRY: 4,    # PAIN
    ExpressionLabel.DISGUSTED: 5,  # NO alternative
    ExpressionLabel.FEARFUL: 6,  # HELP alternative
    ExpressionLabel.CONFUSED: 7,
    ExpressionLabel.THINKING: 8,
}

NUM_CLASSES = len(ASSISTIVE_LABELS)

class ExpressionDatasetBuilder:
    """Build dataset from webcam captures or synthetic data."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.landmarker = None  # Lazy-loaded only when needed for capture
        self.data = []
        self.labels = []

    def capture_interactive(self, samples_per_class: int = 50):
        if self.landmarker is None:
            print("Initializing face landmark detector...")
            self.landmarker = MediaPipeFaceLandmarker(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
            )
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        expressions = [
            (ExpressionLabel.NEUTRAL, "NEUTRAL", "Relax your face. Don't make any expression."),
            (ExpressionLabel.HAPPY, "HAPPY", "Smile naturally! Show teeth if you can."),
            (ExpressionLabel.SAD, "SAD", "Frown. Turn corners of mouth down."),
            (ExpressionLabel.SURPRISED, "SURPRISED", "Open mouth wide, raise eyebrows."),
            (ExpressionLabel.ANGRY, "ANGRY", "Furrow brows, tighten mouth."),
            (ExpressionLabel.FEARFUL, "FEARFUL", "Widen eyes, open mouth slightly."),
            (ExpressionLabel.CONFUSED, "CONFUSED", "Raise one eyebrow, tilt head."),
        ]

        print("\n" + "=" * 50)
        print("  WEBCAM DATA CAPTURE")
        print("=" * 50)
        print(f"  Will capture {samples_per_class} samples per expression.")
        print(f"  Each expression: 3s countdown + auto-capture.")
        print("  Press Q at any time to stop.")
        print("=" * 50)
        print()

        for label, name, instruction in expressions:
            print(f"\n--- {name} ---")
            print(f"  {instruction}")

            countdown_duration = 3.0
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                remaining = countdown_duration - elapsed
                if remaining <= 0:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

                cv2.putText(frame, name, (w // 2 - 100, h // 2 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(frame, instruction, (w // 2 - 200, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                num = str(int(remaining) + 1)
                cv2.putText(frame, num, (w // 2 - 20, h // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
                cv2.putText(frame, "Get ready...", (w // 2 - 80, h // 2 + 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow("Capture", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            captured = 0
            failed = 0
            while captured < samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    break

                landmarks = self.landmarker.process(frame)
                face_detected = landmarks is not None

                h, w = frame.shape[:2]
                status_color = (0, 255, 0) if face_detected else (0, 0, 255)
                status_text = "Face OK" if face_detected else "No face detected!"
                cv2.circle(frame, (30, 30), 12, status_color, -1)
                cv2.putText(frame, status_text, (50, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
                cv2.putText(frame, f"{name} ({captured}/{samples_per_class})",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                bar_w = int((captured / samples_per_class) * (w - 20))
                cv2.rectangle(frame, (10, h - 30), (10 + bar_w, h - 10), (0, 200, 0), -1)
                cv2.rectangle(frame, (10, h - 30), (w - 10, h - 10), (100, 100, 100), 1)

                cv2.imshow("Capture", frame)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if face_detected:
                    features = self.landmarker.extract_features(landmarks)
                    raw_lm = normalize_landmarks(landmarks.landmarks).flatten().tolist()
                    features['_raw_landmarks'] = raw_lm
                    self.data.append(features)
                    self.labels.append(ASSISTIVE_LABELS[label])
                    captured += 1
                    if captured % 10 == 0:
                        print(f"  Captured {captured}/{samples_per_class}")
                else:
                    failed += 1

            print(f"  Done! Captured {captured} samples (skipped {failed} frames with no face)")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal dataset: {len(self.data)} samples")

    def generate_synthetic(self, samples_per_class: int = 200):
        """Generate synthetic training data based on realistic facial feature distributions."""
        np.random.seed(42)

        # Realistic feature distributions for each expression
        distributions = {
            0: {  # NEUTRAL
                'mouth_open': (3.0, 2.0),
                'mouth_width': (50.0, 8.0),
                'lip_height_avg': (2.0, 1.0),
                'left_ear': (0.25, 0.05),
                'right_ear': (0.25, 0.05),
                'ear_avg': (0.25, 0.05),
                'left_brow_height': (1.0, 2.0),
                'right_brow_height': (1.0, 2.0),
                'pitch': (0.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            1: {  # HAPPY (YES)
                'mouth_open': (8.0, 4.0),
                'mouth_width': (60.0, 10.0),
                'lip_height_avg': (5.0, 2.0),
                'left_ear': (0.22, 0.04),
                'right_ear': (0.22, 0.04),
                'ear_avg': (0.22, 0.04),
                'left_brow_height': (2.0, 2.0),
                'right_brow_height': (2.0, 2.0),
                'pitch': (0.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            2: {  # SAD (NO)
                'mouth_open': (2.0, 1.5),
                'mouth_width': (40.0, 8.0),
                'lip_height_avg': (1.5, 0.8),
                'left_ear': (0.20, 0.04),
                'right_ear': (0.20, 0.04),
                'ear_avg': (0.20, 0.04),
                'left_brow_height': (0.5, 1.5),
                'right_brow_height': (0.5, 1.5),
                'pitch': (3.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            3: {  # SURPRISED (HELP)
                'mouth_open': (15.0, 5.0),
                'mouth_width': (45.0, 8.0),
                'lip_height_avg': (8.0, 3.0),
                'left_ear': (0.35, 0.05),
                'right_ear': (0.35, 0.05),
                'ear_avg': (0.35, 0.05),
                'left_brow_height': (4.0, 2.0),
                'right_brow_height': (4.0, 2.0),
                'pitch': (0.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            4: {  # ANGRY (PAIN)
                'mouth_open': (5.0, 3.0),
                'mouth_width': (45.0, 10.0),
                'lip_height_avg': (3.0, 1.5),
                'left_ear': (0.23, 0.04),
                'right_ear': (0.23, 0.04),
                'ear_avg': (0.23, 0.04),
                'left_brow_height': (-2.0, 1.5),
                'right_brow_height': (-2.0, 1.5),
                'pitch': (-2.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            5: {  # DISGUSTED (NO alt)
                'mouth_open': (3.0, 2.0),
                'mouth_width': (42.0, 8.0),
                'lip_height_avg': (2.0, 1.0),
                'left_ear': (0.24, 0.04),
                'right_ear': (0.24, 0.04),
                'ear_avg': (0.24, 0.04),
                'left_brow_height': (0.5, 2.0),
                'right_brow_height': (0.5, 2.0),
                'pitch': (2.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            6: {  # FEARFUL (HELP alt)
                'mouth_open': (6.0, 3.0),
                'mouth_width': (48.0, 8.0),
                'lip_height_avg': (4.0, 2.0),
                'left_ear': (0.30, 0.05),
                'right_ear': (0.30, 0.05),
                'ear_avg': (0.30, 0.05),
                'left_brow_height': (3.0, 2.0),
                'right_brow_height': (3.0, 2.0),
                'pitch': (-1.0, 5.0),
                'yaw': (0.0, 3.0),
            },
            7: {  # CONFUSED
                'mouth_open': (2.0, 1.5),
                'mouth_width': (45.0, 8.0),
                'lip_height_avg': (1.5, 0.8),
                'left_ear': (0.25, 0.05),
                'right_ear': (0.25, 0.05),
                'ear_avg': (0.25, 0.05),
                'left_brow_height': (3.0, 2.0),
                'right_brow_height': (0.0, 2.0),
                'pitch': (0.0, 5.0),
                'yaw': (5.0, 5.0),
            },
            8: {  # THINKING
                'mouth_open': (1.5, 1.0),
                'mouth_width': (48.0, 8.0),
                'lip_height_avg': (1.2, 0.6),
                'left_ear': (0.24, 0.04),
                'right_ear': (0.24, 0.04),
                'ear_avg': (0.24, 0.04),
                'left_brow_height': (2.5, 2.0),
                'right_brow_height': (1.0, 2.0),
                'pitch': (0.0, 5.0),
                'yaw': (3.0, 5.0),
            },
        }

        feature_names = FeatureExtractor.FEATURE_NAMES

        for class_id, dist_params in distributions.items():
            for _ in range(samples_per_class):
                features = {}
                for name in feature_names:
                    mean, std = dist_params[name]
                    features[name] = float(np.random.normal(mean, std))
                self.data.append(features)
                self.labels.append(class_id)

    def save(self, filename: str = "dataset.json"):
        filepath = os.path.join(self.output_dir, filename)
        dataset = {
            'data': self.data,
            'labels': self.labels,
            'label_names': {v: k.value for k, v in ASSISTIVE_LABELS.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filepath}")

    def load(self, filename: str = "dataset.json"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        self.data = dataset['data']
        self.labels = dataset['labels']
        print(f"Dataset loaded from {filepath}: {len(self.data)} samples")

    def merge_all(self):
        import glob
        pattern = os.path.join(self.output_dir, "dataset*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No dataset files found in {self.output_dir}/")
            return
        self.data = []
        self.labels = []
        for f in files:
            with open(f, 'r') as fh:
                dataset = json.load(fh)
            self.data.extend(dataset['data'])
            self.labels.extend(dataset['labels'])
            name = os.path.basename(f)
            print(f"  Loaded {name}: {len(dataset['data'])} samples")
        print(f"Total: {len(self.data)} samples from {len(files)} file(s)")

class ExpressionTrainer:
    """Train and evaluate expression classifier."""

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.classifier = create_classifier(model_type=model_type)

    def prepare_data(self, dataset: ExpressionDatasetBuilder) -> Tuple[np.ndarray, np.ndarray]:
        has_raw = any('_raw_landmarks' in d for d in dataset.data)
        use_raw = self.model_type in ("transformer", "temporal", "occlusion_aware") and has_raw

        if use_raw:
            filtered = [(d, l) for d, l in zip(dataset.data, dataset.labels) if '_raw_landmarks' in d]
            dataset.data = [d for d, _ in filtered]
            dataset.labels = [l for _, l in filtered]
            raw_data = [d['_raw_landmarks'] for d in dataset.data]
            sample_len = len(raw_data[0])
            if sample_len == 478 * 3:
                raw_data = [_normalize_raw_landmarks(r) for r in raw_data]
                print(f"Normalized raw landmarks from pixel coordinates")
            X = np.array(raw_data, dtype=np.float32)
            print(f"Using raw landmarks: {X.shape[1]} features per sample ({len(X)} samples)")
        else:
            X = np.array([
                [d.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES]
                for d in dataset.data
            ])
        y = np.array(dataset.labels)
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Number of classes: {len(np.unique(y))}")

        self.classifier.train(X_train, y_train)

        y_pred = []
        for x in X_test:
            if self.model_type in ("transformer", "temporal", "occlusion_aware") and X.shape[1] > 20:
                result = self.classifier.predict({}, raw_landmarks=x.tolist())
            else:
                features = dict(zip(FeatureExtractor.FEATURE_NAMES, x))
                result = self.classifier.predict(features)
            label_name = result.label.value
            found = False
            for name, class_id in ASSISTIVE_LABELS.items():
                if name.value == label_name:
                    y_pred.append(class_id)
                    found = True
                    break
            if not found:
                y_pred.append(0)

        y_pred = np.array(y_pred)

        if self.model_type in ("transformer", "temporal", "occlusion_aware"):
            cv_mean, cv_std = 0.0, 0.0
        else:
            cv_scores = cross_val_score(
                self.classifier.clf,
                X, y,
                cv=min(5, min(np.bincount(y.astype(int))) if len(X) > 5 else 2),
                scoring='accuracy',
                n_jobs=-1,
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())

        unique_labels = sorted(np.unique(y_test))
        id_to_label = {v: k for k, v in ASSISTIVE_LABELS.items()}
        target_names = [id_to_label[l].value for l in unique_labels]

        report = classification_report(
            y_test, y_pred,
            labels=unique_labels,
            target_names=target_names,
            output_dict=True,
        )

        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

        metrics = {
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'test_accuracy': float(np.mean(y_pred == y_test)),
            'model_type': self.model_type,
        }

        print(f"\nCross-validation accuracy: {cv_mean:.3f} +/- {cv_std:.3f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

        return metrics

    def save(self, path: str):
        """Save trained model."""
        self.classifier.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        self.classifier.load(path)
        print(f"Model loaded from {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Expression Classifier")
    parser.add_argument("--mode", choices=["capture", "train", "evaluate"], default="train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--samples", type=int, default=200, help="Samples per class")
    parser.add_argument("--model-type", choices=["rf", "gb", "transformer", "temporal", "occlusion_aware"], default="rf")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="data/dataset.json")
    args = parser.parse_args()

    dataset = ExpressionDatasetBuilder()

    if args.mode == "capture":
        dataset.capture_interactive(samples_per_class=args.samples)
        dataset.save()

    elif args.mode == "train":
        if args.synthetic:
            dataset.generate_synthetic(samples_per_class=args.samples)
            dataset.save()
        else:
            dataset.load()

        trainer = ExpressionTrainer(model_type=args.model_type)
        X, y = trainer.prepare_data(dataset)
        metrics = trainer.train(X, y)

        os.makedirs("checkpoints", exist_ok=True)
        if args.model_path:
            save_path = args.model_path
        elif args.model_type == "transformer":
            save_path = "checkpoints/expression_transformer.pt"
        elif args.model_type == "temporal":
            save_path = "checkpoints/expression_temporal.pt"
        elif args.model_type == "occlusion_aware":
            save_path = "checkpoints/expression_occlusion.pt"
        else:
            save_path = "checkpoints/expression_model.joblib"
        trainer.save(save_path)

        with open("checkpoints/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    elif args.mode == "evaluate":
        dataset.load()
        trainer = ExpressionTrainer(model_type=args.model_type)
        trainer.load(args.model_path)
        X, y = trainer.prepare_data(dataset)
        metrics = trainer.train(X, y)

if __name__ == "__main__":
    main()
