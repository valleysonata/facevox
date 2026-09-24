import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.landmark_transformer import (
    LandmarkTransformer,
    TemporalExpressionTransformer,
    OcclusionAwareClassifier,
    LandmarkSequenceTransformer,
)


class ExpressionLabel(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    CONFUSED = "confused"
    THINKING = "thinking"
    # Assistive communication intents
    YES = "yes"
    NO = "no"
    HELP = "help"
    PAIN = "pain"
    THIRSTY = "thirsty"
    HUNGRY = "hungry"
    TIRED = "tired"


@dataclass
class ExpressionResult:
    label: ExpressionLabel
    confidence: float
    probabilities: Dict[ExpressionLabel, float]
    features: Dict[str, float]


class FeatureExtractor:
    """Extract and normalize features from facial landmarks."""

    FEATURE_NAMES = [
        'mouth_open',
        'mouth_width',
        'lip_height_avg',
        'left_ear',
        'right_ear',
        'ear_avg',
        'left_brow_height',
        'right_brow_height',
        'pitch',
        'yaw',
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_history = deque(maxlen=30)

    def extract(self, landmarks) -> np.ndarray:
        """Extract raw features from landmarks."""
        return np.array([landmarks.get(name, 0.0) for name in self.FEATURE_NAMES])

    def fit(self, feature_list: List[np.ndarray]):
        """Fit scaler on training features."""
        X = np.array(feature_list)
        self.scaler.fit(X)
        self.fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        if not self.fitted:
            return features
        return self.scaler.transform(features.reshape(1, -1)).flatten()

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            self.fit([features])
        return self.transform(features)

    def add_temporal(self, features: np.ndarray) -> np.ndarray:
        """Add temporal context (current + velocity + acceleration)."""
        self.feature_history.append(features)
        if len(self.feature_history) < 3:
            return features
        curr = np.array(self.feature_history[-1])
        prev = np.array(self.feature_history[-2])
        prev2 = np.array(self.feature_history[-3])
        velocity = curr - prev
        acceleration = velocity - (prev - prev2)
        return np.concatenate([curr, velocity, acceleration])


class ExpressionClassifier:
    """Facial expression classifier with temporal smoothing for ML models."""

    def __init__(
        self,
        model_type: str = "rf",
        temporal_window: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.feature_extractor = FeatureExtractor()
        self.model_type = model_type
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=temporal_window)
        self.is_trained = False
        self.classes_ = None

        if model_type == "rf":
            self.clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1,
                ))
            ])
        elif model_type == "gb":
            self.clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=150, max_depth=8, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                ))
            ])
        else:
            raise ValueError(f"Unknown ML model type: {model_type}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        self.is_trained = True
        self.n_features = X.shape[1]

    def predict(self, features: Dict[str, float], raw_landmarks: Optional[List[float]] = None) -> ExpressionResult:
        """Predict expression. Uses ML if trained, rule-based fallback otherwise."""
        if self.is_trained:
            return self._predict_ml(features, raw_landmarks)
        return self._predict_rules(features)

    def _predict_ml(self, features: Dict[str, float], raw_landmarks: Optional[List[float]] = None) -> ExpressionResult:
        """ML-based prediction using trained model."""
        x = np.array([features.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES])
        proba = self.clf.predict_proba(x.reshape(1, -1))[0]

        id_to_label = {
            0: ExpressionLabel.NEUTRAL, 1: ExpressionLabel.HAPPY, 2: ExpressionLabel.SAD,
            3: ExpressionLabel.SURPRISED, 4: ExpressionLabel.ANGRY, 5: ExpressionLabel.DISGUSTED,
            6: ExpressionLabel.FEARFUL, 7: ExpressionLabel.CONFUSED, 8: ExpressionLabel.THINKING,
        }

        prob_dict = {}
        for i, cls in enumerate(self.classes_):
            cls_int = int(cls)
            label = id_to_label.get(cls_int, ExpressionLabel.NEUTRAL)
            prob_dict[label] = float(proba[i])

        top_label = max(prob_dict, key=prob_dict.get)
        top_conf = prob_dict[top_label]

        self.prediction_history.append((top_label, top_conf))
        if len(self.prediction_history) >= 3:
            weights = np.linspace(0.5, 1.0, len(self.prediction_history))
            vote_counts = {}
            for (label, conf), w in zip(self.prediction_history, weights):
                vote_counts[label] = vote_counts.get(label, 0) + w * conf
            top_label = max(vote_counts, key=vote_counts.get)
            top_conf = min(1.0, vote_counts[top_label] / sum(weights))

        return ExpressionResult(
            label=top_label, confidence=top_conf, probabilities=prob_dict, features=features,
        )

    def _predict_rules(self, features: Dict[str, float]) -> ExpressionResult:
        """Rule-based fallback when no trained model is available."""
        mouth_open = features.get('mouth_open', 0)
        mouth_width = features.get('mouth_width', 0)
        ear_avg = features.get('ear_avg', 0)
        left_brow = features.get('left_brow_height', 0)
        right_brow = features.get('right_brow_height', 0)
        brow_avg = (left_brow + right_brow) / 2
        brow_diff = abs(left_brow - right_brow)
        pitch = features.get('pitch', 0)
        yaw = features.get('yaw', 0)

        scores = {label: 0.0 for label in ExpressionLabel}
        if ear_avg > 0.30 and mouth_open > 15 and brow_avg > 2.5:
            scores[ExpressionLabel.SURPRISED] += 0.8
            if ear_avg > 0.34: scores[ExpressionLabel.SURPRISED] += 0.2
        if mouth_width > 50 and mouth_open > 2:
            smile_score = min(1.0, (mouth_width - 45) / 30)
            scores[ExpressionLabel.HAPPY] += smile_score * 0.7
            if ear_avg < 0.27: scores[ExpressionLabel.HAPPY] += 0.15
            if mouth_width > 55: scores[ExpressionLabel.SURPRISED] *= 0.3
        if mouth_width < 48 and brow_avg < 0.5:
            scores[ExpressionLabel.SAD] += 0.6
            if mouth_open < 5: scores[ExpressionLabel.SAD] += 0.2
        if brow_avg < -0.5:
            anger_score = min(1.0, abs(brow_avg) / 3)
            scores[ExpressionLabel.ANGRY] += anger_score * 0.7
            if mouth_width < 50: scores[ExpressionLabel.ANGRY] += 0.15
        if brow_diff > 3.5 and abs(yaw) > 8: scores[ExpressionLabel.CONFUSED] += 0.7
        if brow_diff > 3.0 and abs(yaw) > 5: scores[ExpressionLabel.THINKING] += 0.5
        if ear_avg > 0.30 and mouth_open > 5 and mouth_open < 15 and brow_avg > 1.0: scores[ExpressionLabel.FEARFUL] += 0.6
        if mouth_open > 3 and mouth_open < 10 and brow_avg < 0.3 and brow_avg > -0.5: scores[ExpressionLabel.DISGUSTED] += 0.5
        if all(v < 0.3 for v in scores.values()): scores[ExpressionLabel.NEUTRAL] = 0.5
        if mouth_open < 5: scores[ExpressionLabel.NEUTRAL] += 0.3
        if -0.5 < brow_avg < 1.0: scores[ExpressionLabel.NEUTRAL] += 0.2

        total = sum(scores.values()) + 1e-6
        prob_dict = {k: v / total for k, v in scores.items()}
        top_label = max(prob_dict, key=prob_dict.get)
        top_conf = prob_dict[top_label]

        self.prediction_history.append((top_label, top_conf))
        if len(self.prediction_history) >= 3:
            weights = np.linspace(0.5, 1.0, len(self.prediction_history))
            vote_counts = {}
            for (label, conf), w in zip(self.prediction_history, weights):
                vote_counts[label] = vote_counts.get(label, 0) + w * conf
            top_label = max(vote_counts, key=vote_counts.get)
            top_conf = min(1.0, vote_counts[top_label] / sum(weights))

        return ExpressionResult(
            label=top_label, confidence=top_conf, probabilities=prob_dict, features=features,
        )

    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'clf': self.clf, 'feature_extractor': self.feature_extractor,
            'classes_': self.classes_, 'is_trained': self.is_trained,
            'model_type': self.model_type,
        }, path)

    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.clf = data['clf']
        self.feature_extractor = data['feature_extractor']
        self.classes_ = data['classes_']
        self.is_trained = data['is_trained']
        self.model_type = data.get('model_type', 'rf')


class PyTorchExpressionClassifier:
    """Wrapper for PyTorch transformer models to provide the same interface as ExpressionClassifier."""

    def __init__(
        self,
        model_type: str = "transformer",
        input_dim: int = 1434,
        num_classes: int = 9,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.prediction_history = deque(maxlen=5)
        self._build_model()

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _build_model(self):
        """(Re)build the underlying torch module for the current config."""
        if self.model_type == "transformer":
            self.model = LandmarkTransformer(input_dim=self.input_dim, num_classes=self.num_classes).to(self.device)
        elif self.model_type == "temporal":
            self.model = TemporalExpressionTransformer(input_dim=self.input_dim, num_classes=self.num_classes, seq_len=15).to(self.device)
        elif self.model_type == "occlusion_aware":
            self.model = OcclusionAwareClassifier(input_dim=self.input_dim, num_classes=self.num_classes).to(self.device)
        elif self.model_type == "temporal_sequence":
            self.model = LandmarkSequenceTransformer(input_dim=self.input_dim, num_classes=self.num_classes).to(self.device)
        else:
            raise ValueError(f"Unknown PyTorch model type: {self.model_type}")

    def predict(self, features: Dict[str, float], raw_landmarks: Optional[List[float]] = None) -> ExpressionResult:
        """Predict expression using PyTorch model."""
        if raw_landmarks is not None:
            x = np.array(raw_landmarks, dtype=np.float32)
        else:
            x = np.array([features.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES], dtype=np.float32)

        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"{self.model_type} model expects {self.input_dim} input features, "
                f"got {x.shape[0]}. Retrain the model or pass matching features."
            )

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

        id_to_label = {
            0: ExpressionLabel.NEUTRAL, 1: ExpressionLabel.HAPPY, 2: ExpressionLabel.SAD,
            3: ExpressionLabel.SURPRISED, 4: ExpressionLabel.ANGRY, 5: ExpressionLabel.DISGUSTED,
            6: ExpressionLabel.FEARFUL, 7: ExpressionLabel.CONFUSED, 8: ExpressionLabel.THINKING,
        }

        prob_dict = {}
        for i in range(len(proba)):
            label = id_to_label.get(i, ExpressionLabel.NEUTRAL)
            prob_dict[label] = float(proba[i])

        top_label = max(prob_dict, key=prob_dict.get)
        top_conf = prob_dict[top_label]

        self.prediction_history.append((top_label, top_conf))
        if len(self.prediction_history) >= 3:
            weights = np.linspace(0.5, 1.0, len(self.prediction_history))
            vote_counts = {}
            for (label, conf), w in zip(self.prediction_history, weights):
                vote_counts[label] = vote_counts.get(label, 0) + w * conf
            top_label = max(vote_counts, key=vote_counts.get)
            top_conf = min(1.0, vote_counts[top_label] / sum(weights))

        return ExpressionResult(
            label=top_label, confidence=top_conf, probabilities=prob_dict,
            features=features if raw_landmarks is None else {'_raw_landmarks': raw_landmarks},
        )

    def train(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None,
              epochs: int = 50, batch_size: int = 64, lr: float = 1e-3):
        """Train the PyTorch model, validating on held-out data when given."""
        if X.shape[1] != self.input_dim:
            self.input_dim = int(X.shape[1])
            self._build_model()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        if self.model_type == "temporal":
            # Repeat for sequence length
            X_tensor = X_tensor.unsqueeze(1).repeat(1, 15, 1)

        if X_val is not None and y_val is not None:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
            if self.model_type == "temporal":
                X_val_t = X_val_t.unsqueeze(1).repeat(1, 15, 1)
        else:
            X_val_t, y_val_t = X_tensor, y_tensor

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0
        best_state = None
        n_train = len(X)

        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n_train)
            total_loss = 0
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X_tensor[idx]
                batch_y = y_tensor[idx]
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t).argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} — loss: {total_loss / max(1, n_train // batch_size):.4f} val_acc: {val_acc:.3f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_trained = True
        print(f"  Best val accuracy: {best_val_acc:.3f}")

    def save(self, path: str):
        """Save PyTorch model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'model_type': self.model_type,
        }, path)

    def load(self, path: str):
        """Load PyTorch model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {}) or {}
        model_type = config.get("model_type", checkpoint.get("model_type", self.model_type))
        input_dim = config.get("input_dim", checkpoint.get("input_dim", self.input_dim))
        num_classes = config.get("num_classes", checkpoint.get("num_classes", self.num_classes))
        if model_type != self.model_type or input_dim != self.input_dim or num_classes != self.num_classes:
            # Rebuild with the checkpoint's architecture before loading weights.
            self.model_type = model_type
            self.input_dim = int(input_dim)
            self.num_classes = int(num_classes)
            self._build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True


class AssistiveExpressionMapper:
    """Map expressions to assistive communication intents."""

    INTENT_MAP = {
        ExpressionLabel.HAPPY: ExpressionLabel.YES,
        ExpressionLabel.SAD: ExpressionLabel.NO,
        ExpressionLabel.SURPRISED: ExpressionLabel.HELP,
        ExpressionLabel.ANGRY: ExpressionLabel.PAIN,
        ExpressionLabel.DISGUSTED: ExpressionLabel.NO,
        ExpressionLabel.FEARFUL: ExpressionLabel.HELP,
        ExpressionLabel.CONFUSED: ExpressionLabel.HELP,
        ExpressionLabel.THINKING: ExpressionLabel.NEUTRAL,
    }

    def __init__(self):
        self.intent_history = deque(maxlen=10)

    def map_to_intent(self, result: ExpressionResult) -> ExpressionResult:
        """Map expression to communication intent (does not mutate the input)."""
        label = result.label
        features = result.features
        brow_up = features.get('left_brow_height', 0) > 0.5 or features.get('right_brow_height', 0) > 0.5
        mouth_open = features.get('mouth_open', 0) > 8.0
        mouth_width = features.get('mouth_width', 0)
        mouth_corner_down = mouth_width < 48.0
        eye_wide = features.get('ear_avg', 0) > 0.26
        brow_furrow = features.get('left_brow_height', 0) < -0.3 or features.get('right_brow_height', 0) < -0.3
        pitch = features.get('pitch', 0)

        if label in (ExpressionLabel.HAPPY, ExpressionLabel.YES):
            intent = ExpressionLabel.YES
        elif label in (ExpressionLabel.SAD, ExpressionLabel.NO):
            intent = ExpressionLabel.NO
        elif label in (ExpressionLabel.SURPRISED, ExpressionLabel.HELP):
            intent = ExpressionLabel.HELP
        elif label in (ExpressionLabel.ANGRY, ExpressionLabel.PAIN):
            intent = ExpressionLabel.PAIN
        elif label in (ExpressionLabel.FEARFUL,):
            intent = ExpressionLabel.HELP
        elif brow_furrow and mouth_open:
            intent = ExpressionLabel.PAIN
        elif eye_wide and brow_up:
            intent = ExpressionLabel.HELP
        else:
            intent = ExpressionLabel.NEUTRAL

        intent_result = ExpressionResult(
            label=intent,
            confidence=result.confidence,
            probabilities=result.probabilities,
            features=result.features,
        )
        self.intent_history.append(intent)
        return intent_result


# --- Factory Functions ---

def create_classifier(
    model_type: str = "rf",
    model_path: Optional[str] = None,
    input_dim: int = 1434,
    num_classes: int = 9,
):
    """Factory function to create the appropriate classifier."""
    if model_type in ("transformer", "temporal", "occlusion_aware", "temporal_sequence"):
        return PyTorchExpressionClassifier(
            model_type=model_type,
            input_dim=input_dim,
            num_classes=num_classes,
            model_path=model_path,
        )
    clf = ExpressionClassifier(model_type=model_type)
    if model_path and os.path.exists(model_path):
        clf.load(model_path)
    return clf


def create_mapper() -> AssistiveExpressionMapper:
    """Factory function to create an intent mapper."""
    return AssistiveExpressionMapper()
