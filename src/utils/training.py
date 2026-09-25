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
    create_mapper,
)


def _normalize_raw_landmarks(raw_list):
    lm = np.array(raw_list, dtype=np.float32).reshape(478, 3)
    normed = normalize_landmarks(lm)
    return normed.flatten().tolist()


ASSISTIVE_LABELS = {
    ExpressionLabel.NEUTRAL: 0,
    ExpressionLabel.HAPPY: 1,
    ExpressionLabel.SAD: 2,
    ExpressionLabel.SURPRISED: 3,
    ExpressionLabel.ANGRY: 4,
    ExpressionLabel.DISGUSTED: 5,
    ExpressionLabel.FEARFUL: 6,
    ExpressionLabel.CONFUSED: 7,
    ExpressionLabel.THINKING: 8,
}

NUM_CLASSES = len(ASSISTIVE_LABELS)


class ExpressionDatasetBuilder:
    """Build dataset from webcam captures or synthetic data."""

    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.landmarker = None
        self.data = []
        self.labels = []

    def capture_interactive(self, samples_per_class=50):
        if self.landmarker is None:
            print("Initializing face landmark detector...")
            self.landmarker = MediaPipeFaceLandmarker(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            )
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        expressions = [
            (ExpressionLabel.NEUTRAL, "NEUTRAL", "Relax your face."),
            (ExpressionLabel.HAPPY, "HAPPY", "Smile naturally!"),
            (ExpressionLabel.SAD, "SAD", "Frown."),
            (ExpressionLabel.SURPRISED, "SURPRISED", "Open mouth wide."),
            (ExpressionLabel.ANGRY, "ANGRY", "Furrow brows."),
            (ExpressionLabel.FEARFUL, "FEARFUL", "Widen eyes."),
            (ExpressionLabel.CONFUSED, "CONFUSED", "Raise one eyebrow."),
        ]

        print("\n" + "=" * 50)
        print("  WEBCAM DATA CAPTURE")
        print("=" * 50)
        print(f"  Will capture {samples_per_class} samples per expression.")
        print("  Press Q at any time to stop.")
        print("=" * 50)

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
                cv2.imshow("Capture", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return

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
                    cap.release(); cv2.destroyAllWindows(); return
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

            print(f"  Done! Captured {captured} samples (skipped {failed} frames)")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal dataset: {len(self.data)} samples")

    def generate_synthetic(self, samples_per_class=200):
        """Generate synthetic training data (scale-normalized feature units)."""
        np.random.seed(42)
        # Means/stds are fractions of the inter-eye distance, matching the
        # output of MediaPipeFaceLandmarker.extract_features.
        distributions = {
            0: {'mouth_open': (0.050, 0.033), 'mouth_width': (0.833, 0.133), 'lip_height_avg': (0.033, 0.017),
                'left_ear': (0.25, 0.05), 'right_ear': (0.25, 0.05), 'ear_avg': (0.25, 0.05),
                'left_brow_height': (0.017, 0.033), 'right_brow_height': (0.017, 0.033), 'pitch': (0.0, 0.083), 'yaw': (0.0, 0.05)},
            1: {'mouth_open': (0.133, 0.067), 'mouth_width': (1.0, 0.167), 'lip_height_avg': (0.083, 0.033),
                'left_ear': (0.22, 0.04), 'right_ear': (0.22, 0.04), 'ear_avg': (0.22, 0.04),
                'left_brow_height': (0.033, 0.033), 'right_brow_height': (0.033, 0.033), 'pitch': (0.0, 0.083), 'yaw': (0.0, 0.05)},
            2: {'mouth_open': (0.033, 0.025), 'mouth_width': (0.667, 0.133), 'lip_height_avg': (0.025, 0.013),
                'left_ear': (0.20, 0.04), 'right_ear': (0.20, 0.04), 'ear_avg': (0.20, 0.04),
                'left_brow_height': (0.008, 0.025), 'right_brow_height': (0.008, 0.025), 'pitch': (0.05, 0.083), 'yaw': (0.0, 0.05)},
            3: {'mouth_open': (0.25, 0.083), 'mouth_width': (0.75, 0.133), 'lip_height_avg': (0.133, 0.05),
                'left_ear': (0.35, 0.05), 'right_ear': (0.35, 0.05), 'ear_avg': (0.35, 0.05),
                'left_brow_height': (0.067, 0.033), 'right_brow_height': (0.067, 0.033), 'pitch': (0.0, 0.083), 'yaw': (0.0, 0.05)},
            4: {'mouth_open': (0.083, 0.05), 'mouth_width': (0.75, 0.167), 'lip_height_avg': (0.05, 0.025),
                'left_ear': (0.23, 0.04), 'right_ear': (0.23, 0.04), 'ear_avg': (0.23, 0.04),
                'left_brow_height': (-0.033, 0.025), 'right_brow_height': (-0.033, 0.025), 'pitch': (-0.033, 0.083), 'yaw': (0.0, 0.05)},
            5: {'mouth_open': (0.05, 0.033), 'mouth_width': (0.70, 0.133), 'lip_height_avg': (0.033, 0.017),
                'left_ear': (0.24, 0.04), 'right_ear': (0.24, 0.04), 'ear_avg': (0.24, 0.04),
                'left_brow_height': (0.008, 0.033), 'right_brow_height': (0.008, 0.033), 'pitch': (0.033, 0.083), 'yaw': (0.0, 0.05)},
            6: {'mouth_open': (0.10, 0.05), 'mouth_width': (0.80, 0.133), 'lip_height_avg': (0.067, 0.033),
                'left_ear': (0.30, 0.05), 'right_ear': (0.30, 0.05), 'ear_avg': (0.30, 0.05),
                'left_brow_height': (0.05, 0.033), 'right_brow_height': (0.05, 0.033), 'pitch': (-0.017, 0.083), 'yaw': (0.0, 0.05)},
            7: {'mouth_open': (0.033, 0.025), 'mouth_width': (0.75, 0.133), 'lip_height_avg': (0.025, 0.013),
                'left_ear': (0.25, 0.05), 'right_ear': (0.25, 0.05), 'ear_avg': (0.25, 0.05),
                'left_brow_height': (0.05, 0.033), 'right_brow_height': (0.0, 0.033), 'pitch': (0.0, 0.083), 'yaw': (0.083, 0.083)},
            8: {'mouth_open': (0.025, 0.017), 'mouth_width': (0.80, 0.133), 'lip_height_avg': (0.02, 0.01),
                'left_ear': (0.24, 0.04), 'right_ear': (0.24, 0.04), 'ear_avg': (0.24, 0.04),
                'left_brow_height': (0.042, 0.033), 'right_brow_height': (0.017, 0.033), 'pitch': (0.0, 0.083), 'yaw': (0.05, 0.083)},
        }
        feature_names = FeatureExtractor.FEATURE_NAMES
        rng = np.random.default_rng(42)
        for class_id, dist_params in distributions.items():
            for _ in range(samples_per_class):
                features = {}
                for name in feature_names:
                    mean, std = dist_params[name]
                    features[name] = float(np.random.normal(mean, std))
                # Synthetic raw landmarks so landmark-based (torch) models can
                # also train on synthetic data. Each class gets a distinct
                # offset so the classes are separable.
                base = rng.normal(0.0, 1.0, (478, 3)).astype(np.float32)
                base += (class_id - 4) * 0.4
                features['_raw_landmarks'] = base.flatten().tolist()
                self.data.append(features)
                self.labels.append(class_id)
        print(f"Generated {len(self.data)} synthetic samples")

    def save(self, filename="dataset.json"):
        filepath = os.path.join(self.output_dir, filename)
        dataset = {'data': self.data, 'labels': self.labels, 'label_names': {v: k.value for k, v in ASSISTIVE_LABELS.items()}}
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filepath}")

    def load(self, filename="dataset.json"):
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
            print(f"  Loaded {os.path.basename(f)}: {len(dataset['data'])} samples")
        print(f"Total: {len(self.data)} samples from {len(files)} file(s)")


class ExpressionTrainer:
    """Train and evaluate expression classifier."""

    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.classifier = create_classifier(model_type=model_type)

    def prepare_data(self, dataset):
        has_raw = any('_raw_landmarks' in d for d in dataset.data)
        use_raw = self.model_type in ("transformer", "temporal", "occlusion_aware") and has_raw

        if use_raw:
            filtered = [(d, l) for d, l in zip(dataset.data, dataset.labels) if '_raw_landmarks' in d]
            if not filtered:
                raise ValueError(
                    f"No samples with raw landmarks found, cannot train {self.model_type} model."
                )
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
            if self.model_type in ("transformer", "temporal", "occlusion_aware") and not has_raw:
                raise ValueError(
                    f"{self.model_type} model needs raw landmarks ('_raw_landmarks'), "
                    f"but the dataset only has geometric features. "
                    f"Capture data with the webcam (which stores raw landmarks) "
                    f"or train an 'rf'/'gb' model instead."
                )
            X = np.array([[d.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES] for d in dataset.data])
        y = np.array(dataset.labels)
        if self.model_type in ("transformer", "temporal", "occlusion_aware"):
            # Rebuild the torch classifier for the actual feature dimension.
            self.classifier = create_classifier(
                self.model_type, input_dim=int(X.shape[1]),
                num_classes=int(np.max(y)) + 1,
            )
        return X, y

    def _predict_labels(self, X):
        """Predict integer labels for a feature matrix."""
        label_to_id = {k.value: v for k, v in ASSISTIVE_LABELS.items()}
        y_pred = []
        if self.model_type in ("transformer", "temporal", "occlusion_aware"):
            for x in X:
                result = self.classifier.predict({}, raw_landmarks=x.tolist())
                y_pred.append(label_to_id.get(result.label.value, 0))
        else:
            for x in X:
                features = dict(zip(FeatureExtractor.FEATURE_NAMES, x))
                result = self.classifier.predict(features)
                y_pred.append(label_to_id.get(result.label.value, 0))
        return np.array(y_pred)

    def _score(self, X_test, y_test):
        """Compute accuracy, report and confusion matrix on held-out data."""
        y_pred = self._predict_labels(X_test)
        labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
        id_to_label = {v: k for k, v in ASSISTIVE_LABELS.items()}
        target_names = [id_to_label[int(l)].value if int(l) in id_to_label else str(l)
                        for l in labels]
        report = classification_report(
            y_test, y_pred, labels=labels, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        test_acc = float(np.mean(y_pred == y_test))
        return test_acc, report, cm, target_names, y_pred

    def train(self, X, y, test_size=0.2, epochs=50):
        n_classes = len(np.unique(y))
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y,
            )
        except ValueError:
            # Too few samples per class for a stratified split.
            print("Too few samples for a stratified split; using a plain shuffle split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
            )
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Number of classes: {len(np.unique(y))}")

        cv_mean = 0.0
        cv_std = 0.0

        if self.model_type in ("transformer", "temporal", "occlusion_aware"):
            # PyTorch model (X_test is genuinely held out and used for validation).
            self.classifier.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=epochs)
        else:
            self.classifier.train(X_train, y_train)
            min_class_count = int(np.min(np.bincount(y.astype(int))))
            if min_class_count >= 2:
                cv_scores = cross_val_score(
                    self.classifier.clf, X, y,
                    cv=min(5, min_class_count), scoring="accuracy", n_jobs=-1,
                )
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
                print(f"\nCross-validation accuracy: {cv_mean:.3f} +/- {cv_std:.3f}")
            else:
                print("\nSkipping cross-validation (fewer than 2 samples per class).")

        test_acc, report, cm, target_names, y_pred = self._score(X_test, y_test)

        metrics = {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "test_accuracy": test_acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "target_names": target_names,
            "model_type": self.model_type,
        }
        print(f"\nTest accuracy: {test_acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=sorted(np.unique(y_test).tolist()),
            target_names=[{v: k for k, v in ASSISTIVE_LABELS.items()}[int(l)].value
                          for l in sorted(np.unique(y_test).tolist())],
            zero_division=0,
        ))
        return metrics

    def evaluate(self, X, y):
        """Evaluate an already-loaded model without retraining."""
        if self.model_type not in ("transformer", "temporal", "occlusion_aware"):
            if not getattr(self.classifier, "is_trained", False):
                raise ValueError("No trained model loaded. Load a model before evaluating.")
        test_acc, report, cm, target_names, _ = self._score(X, y)
        metrics = {
            "test_accuracy": test_acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "target_names": target_names,
            "model_type": self.model_type,
            "num_samples": int(len(y)),
        }
        print(f"\nEval accuracy ({len(y)} samples): {test_acc:.3f}")
        return metrics

    def save(self, path):
        """Save trained model."""
        self.classifier.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load trained model."""
        self.classifier.load(path)
        print(f"Model loaded from {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Expression Classifier")
    parser.add_argument("--mode", choices=["capture", "train", "evaluate"], default="train")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
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
        dataset.merge_all()
        if not dataset.data:
            print("No dataset files found. Nothing to evaluate.")
            return
        trainer = ExpressionTrainer(model_type=args.model_type)
        if not args.model_path:
            print("--model-path is required for evaluate mode.")
            return
        trainer.load(args.model_path)
        X, y = trainer.prepare_data(dataset)
        metrics = trainer.evaluate(X, y)


if __name__ == "__main__":
    main()
