import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
        self.feature_history = deque(maxlen=30)  # temporal smoothing
        
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
    """Facial expression classifier with temporal smoothing."""
    
    def __init__(
        self,
        model_type: str = "rf",
        temporal_window: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.feature_extractor = FeatureExtractor()
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=temporal_window)
        
        if model_type == "rf":
            self.clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                ))
            ])
        elif model_type == "gb":
            self.clf = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.is_trained = False
        self.classes_ = None
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        self.is_trained = True
        self.n_features = X.shape[1]
        
    def predict(self, features: Dict[str, float]) -> ExpressionResult:
        """Predict expression. Uses ML if trained, rule-based fallback otherwise."""
        if self.is_trained:
            return self._predict_ml(features)
        return self._predict_rules(features)
    
    def _predict_ml(self, features: Dict[str, float]) -> ExpressionResult:
        """ML-based prediction using trained model."""
        x = np.array([features.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES])
        
        proba = self.clf.predict_proba(x.reshape(1, -1))[0]
        
        id_to_label = {
            0: ExpressionLabel.NEUTRAL,
            1: ExpressionLabel.HAPPY,
            2: ExpressionLabel.SAD,
            3: ExpressionLabel.SURPRISED,
            4: ExpressionLabel.ANGRY,
            5: ExpressionLabel.DISGUSTED,
            6: ExpressionLabel.FEARFUL,
            7: ExpressionLabel.CONFUSED,
            8: ExpressionLabel.THINKING,
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
            label=top_label,
            confidence=top_conf,
            probabilities=prob_dict,
            features=features,
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

        # SURPRISED: eyes wide open + mouth open + eyebrows raised
        if ear_avg > 0.30 and mouth_open > 15 and brow_avg > 2.5:
            scores[ExpressionLabel.SURPRISED] += 0.8
            if ear_avg > 0.34:
                scores[ExpressionLabel.SURPRISED] += 0.2

        # HAPPY: mouth wide (smile), eyes slightly squinted
        if mouth_width > 50 and mouth_open > 2:
            smile_score = min(1.0, (mouth_width - 45) / 30)
            scores[ExpressionLabel.HAPPY] += smile_score * 0.7
            if ear_avg < 0.27:
                scores[ExpressionLabel.HAPPY] += 0.15
            if mouth_width > 55:
                scores[ExpressionLabel.SURPRISED] *= 0.3

        # SAD: mouth narrow/corners down, brows low
        if mouth_width < 48 and brow_avg < 0.5:
            scores[ExpressionLabel.SAD] += 0.6
            if mouth_open < 5:
                scores[ExpressionLabel.SAD] += 0.2

        # ANGRY: eyebrows furrowed (negative brow height), mouth tight
        if brow_avg < -0.5:
            anger_score = min(1.0, abs(brow_avg) / 3)
            scores[ExpressionLabel.ANGRY] += anger_score * 0.7
            if mouth_width < 50:
                scores[ExpressionLabel.ANGRY] += 0.15

        # CONFUSED: asymmetric eyebrows, head tilted
        if brow_diff > 3.5 and abs(yaw) > 8:
            scores[ExpressionLabel.CONFUSED] += 0.7

        # THINKING: one brow raised, head slightly tilted
        if brow_diff > 3.0 and abs(yaw) > 5:
            scores[ExpressionLabel.THINKING] += 0.5

        # FEARFUL: eyes wide + mouth slightly open + brows raised
        if ear_avg > 0.30 and mouth_open > 5 and mouth_open < 15 and brow_avg > 1.0:
            scores[ExpressionLabel.FEARFUL] += 0.6

        # DISGUSTED: nose wrinkle area (mouth slightly open, brow down)
        if mouth_open > 3 and mouth_open < 10 and brow_avg < 0.3 and brow_avg > -0.5:
            scores[ExpressionLabel.DISGUSTED] += 0.5

        if all(v < 0.3 for v in scores.values()):
            scores[ExpressionLabel.NEUTRAL] = 0.5

        if mouth_open < 5:
            scores[ExpressionLabel.NEUTRAL] += 0.3
        if -0.5 < brow_avg < 1.0:
            scores[ExpressionLabel.NEUTRAL] += 0.2

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
            label=top_label,
            confidence=top_conf,
            probabilities=prob_dict,
            features=features,
        )
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'clf': self.clf,
            'feature_extractor': self.feature_extractor,
            'classes_': self.classes_,
            'is_trained': self.is_trained,
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.clf = data['clf']
        self.feature_extractor = data['feature_extractor']
        self.classes_ = data['classes_']
        self.is_trained = data['is_trained']

class AssistiveExpressionMapper:
    """Map expressions to assistive communication intents."""
    
    # Mapping from facial expressions to communication intents
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
    
    # Composite expressions (combinations that indicate specific intents)
    COMPOSITE_INTENTS = {
        # (eyebrow_raised, mouth_open, head_nod) -> intent
        ('brow_up', 'mouth_open', 'nod'): ExpressionLabel.YES,
        ('brow_up', 'mouth_closed', 'shake'): ExpressionLabel.NO,
        ('brow_furrow', 'mouth_open', None): ExpressionLabel.PAIN,
        ('brow_up', 'mouth_corner_down', None): ExpressionLabel.SAD,
        ('eye_wide', 'brow_up', 'mouth_open'): ExpressionLabel.SURPRISED,
    }
    
    def __init__(self):
        self.intent_history = deque(maxlen=10)
        
    def map_to_intent(self, result: ExpressionResult) -> ExpressionResult:
        """Map expression to communication intent."""
        label = result.label
        
        features = result.features
        brow_up = features.get('left_brow_height', 0) > 0.5 or features.get('right_brow_height', 0) > 0.5
        mouth_open = features.get('mouth_open', 0) > 8.0
        mouth_width = features.get('mouth_width', 0)
        mouth_corner_down = mouth_width < 48.0
        eye_wide = features.get('ear_avg', 0) > 0.26
        brow_furrow = features.get('left_brow_height', 0) < -0.3 or features.get('right_brow_height', 0) < -0.3
        pitch = features.get('pitch', 0)
        nod = pitch < -2.0
        shake = abs(features.get('yaw', 0)) > 5.0
        
        if label == ExpressionLabel.HAPPY:
            intent = ExpressionLabel.YES
        elif label == ExpressionLabel.SAD:
            intent = ExpressionLabel.NO
        elif label == ExpressionLabel.SURPRISED:
            intent = ExpressionLabel.HELP
        elif label == ExpressionLabel.ANGRY:
            intent = ExpressionLabel.PAIN
        elif label == ExpressionLabel.FEARFUL:
            intent = ExpressionLabel.HELP
        elif brow_furrow and mouth_open:
            intent = ExpressionLabel.PAIN
        elif eye_wide and brow_up:
            intent = ExpressionLabel.HELP
        elif nod and not mouth_open:
            intent = ExpressionLabel.YES
        elif shake:
            intent = ExpressionLabel.NO
        else:
            intent = ExpressionLabel.NEUTRAL
        
        self.intent_history.append(intent)
        
        if len(self.intent_history) >= 3:
            from collections import Counter
            counts = Counter(self.intent_history)
            intent = counts.most_common(1)[0][0]
        
        return ExpressionResult(
            label=intent,
            confidence=result.confidence,
            probabilities=result.probabilities,
            features=features,
        )


class TransformerClassifier:
    def __init__(self, model_path=None, num_classes=7, input_dim=10, temporal_window=5):
        self.temporal_window = temporal_window
        self.prediction_history = deque(maxlen=temporal_window)
        self.is_trained = False
        self.num_classes = num_classes
        self.input_dim = input_dim
        self._model = None
        self._device = 'cpu'
        self._label_map = {
            0: ExpressionLabel.NEUTRAL,
            1: ExpressionLabel.HAPPY,
            2: ExpressionLabel.SAD,
            3: ExpressionLabel.SURPRISED,
            4: ExpressionLabel.ANGRY,
            5: ExpressionLabel.DISGUSTED,
            6: ExpressionLabel.FEARFUL,
        }
        if model_path:
            self.load(model_path)

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from src.models.landmark_transformer import LandmarkTransformer
        self._model = LandmarkTransformer(
            input_dim=self.input_dim, num_classes=self.num_classes,
        ).to(self._device)
        self._model.eval()

    def predict(self, features: Dict[str, float], raw_landmarks=None) -> ExpressionResult:
        if not self.is_trained or self._model is None:
            return self._predict_rules(features)

        import torch

        if raw_landmarks is not None and len(raw_landmarks) == self.input_dim:
            x = np.array(raw_landmarks, dtype=np.float32)
        else:
            from src.models.expression_recognition import FeatureExtractor
            x = np.array([features.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES])

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_dict = {}
        for i, prob in enumerate(probs):
            label = self._label_map.get(i, ExpressionLabel.NEUTRAL)
            prob_dict[label] = float(prob)

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
            label=top_label,
            confidence=top_conf,
            probabilities=prob_dict,
            features=features,
        )

    def _predict_rules(self, features):
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
            if ear_avg > 0.34:
                scores[ExpressionLabel.SURPRISED] += 0.2

        if mouth_width > 50 and mouth_open > 2:
            smile_score = min(1.0, (mouth_width - 45) / 30)
            scores[ExpressionLabel.HAPPY] += smile_score * 0.7
            if ear_avg < 0.27:
                scores[ExpressionLabel.HAPPY] += 0.15
            if mouth_width > 55:
                scores[ExpressionLabel.SURPRISED] *= 0.3

        if mouth_width < 48 and brow_avg < 0.5:
            scores[ExpressionLabel.SAD] += 0.6
            if mouth_open < 5:
                scores[ExpressionLabel.SAD] += 0.2

        if brow_avg < -0.5:
            anger_score = min(1.0, abs(brow_avg) / 3)
            scores[ExpressionLabel.ANGRY] += anger_score * 0.7
            if mouth_width < 50:
                scores[ExpressionLabel.ANGRY] += 0.15

        if brow_diff > 3.5 and abs(yaw) > 8:
            scores[ExpressionLabel.CONFUSED] += 0.7

        if brow_diff > 3.0 and abs(yaw) > 5:
            scores[ExpressionLabel.THINKING] += 0.5

        if ear_avg > 0.30 and mouth_open > 5 and mouth_open < 15 and brow_avg > 1.0:
            scores[ExpressionLabel.FEARFUL] += 0.6

        if mouth_open > 3 and mouth_open < 10 and brow_avg < 0.3 and brow_avg > -0.5:
            scores[ExpressionLabel.DISGUSTED] += 0.5

        if all(v < 0.3 for v in scores.values()):
            scores[ExpressionLabel.NEUTRAL] = 0.5

        if mouth_open < 5:
            scores[ExpressionLabel.NEUTRAL] += 0.3
        if -0.5 < brow_avg < 1.0:
            scores[ExpressionLabel.NEUTRAL] += 0.2

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
            label=top_label,
            confidence=top_conf,
            probabilities=prob_dict,
            features=features,
        )

    def train(self, X, y):
        import torch
        from src.models.landmark_transformer import (
            LandmarkTransformer, train_landmark_transformer, save_landmark_transformer,
        )

        unique_labels = np.unique(y)
        self.num_classes = len(unique_labels)

        label_remap = {old: new for new, old in enumerate(sorted(unique_labels))}
        y_remap = np.array([label_remap[int(v)] for v in y])

        n = len(X)
        n_val = max(1, int(n * 0.2))
        perm = np.random.permutation(n)
        X_train, X_val = X[perm[n_val:]], X[perm[:n_val]]
        y_train, y_val = y_remap[perm[n_val:]], y_remap[perm[:n_val]]

        id_to_expr = {v.value: k for k, v in ExpressionLabel.__members__.items()}
        self._label_map = {}
        for old_label, new_id in label_remap.items():
            expr_label = id_to_expr.get(str(old_label), None)
            if expr_label is None:
                for el in ExpressionLabel:
                    if el.value == str(old_label):
                        expr_label = el
                        break
            if expr_label is None:
                expr_label = ExpressionLabel.NEUTRAL
            self._label_map[new_id] = expr_label

        model, val_acc = train_landmark_transformer(
            X_train, y_train, X_val, y_val,
            num_classes=self.num_classes,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            device=self._device,
            input_dim=X_train.shape[1],
        )

        self._model = model
        self._model.eval()
        self.is_trained = True
        self.classes_ = sorted(unique_labels)

    def save(self, path):
        import torch
        if self._model is not None:
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'input_dim': self.input_dim,
                'num_classes': self.num_classes,
                'label_map': {int(k): v for k, v in self._label_map.items()},
                'is_trained': self.is_trained,
            }, path)
            print(f"Model saved to {path}")

    def load(self, path):
        import torch
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.num_classes = checkpoint.get('num_classes', 7)
            self.input_dim = checkpoint.get('input_dim', 10)
            self._label_map = {int(k): v for k, v in checkpoint.get('label_map', {}).items()}
            self.is_trained = checkpoint.get('is_trained', True)
            self._ensure_model()
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()
            self.classes_ = sorted(self._label_map.keys())
        except Exception as e:
            print(f"Could not load transformer model: {e}")
            self.is_trained = False


def create_classifier(model_type: str = "rf", **kwargs):
    if model_type == "transformer":
        return TransformerClassifier(**kwargs)
    return ExpressionClassifier(model_type=model_type, **kwargs)

def create_mapper() -> AssistiveExpressionMapper:
    return AssistiveExpressionMapper()