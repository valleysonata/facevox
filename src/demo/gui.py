import sys
import time
import os
import numpy as np
import cv2
from typing import Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QGroupBox, QStatusBar,
    QProgressBar, QFrame, QGridLayout, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from src.models.face_landmarks import FaceLandmarkPipeline, normalize_landmarks
from src.models.expression_recognition import (
    ExpressionClassifier,
    AssistiveExpressionMapper,
    ExpressionLabel,
    create_classifier,
    create_mapper,
)
from src.models.occlusion import RobustOcclusionHandler

EXPRESSION_COLORS = {
    ExpressionLabel.NEUTRAL: QColor(200, 200, 200),
    ExpressionLabel.HAPPY: QColor(0, 255, 0),
    ExpressionLabel.SAD: QColor(0, 0, 255),
    ExpressionLabel.SURPRISED: QColor(0, 255, 255),
    ExpressionLabel.ANGRY: QColor(255, 0, 0),
    ExpressionLabel.DISGUSTED: QColor(0, 128, 0),
    ExpressionLabel.FEARFUL: QColor(128, 0, 128),
    ExpressionLabel.CONFUSED: QColor(128, 128, 0),
    ExpressionLabel.THINKING: QColor(128, 128, 128),
    ExpressionLabel.YES: QColor(0, 255, 0),
    ExpressionLabel.NO: QColor(255, 0, 0),
    ExpressionLabel.HELP: QColor(255, 165, 0),
    ExpressionLabel.PAIN: QColor(255, 0, 255),
    ExpressionLabel.THIRSTY: QColor(0, 200, 255),
    ExpressionLabel.HUNGRY: QColor(200, 100, 0),
    ExpressionLabel.TIRED: QColor(100, 100, 200),
}

INTENT_LABELS = {
    ExpressionLabel.YES: "YES",
    ExpressionLabel.NO: "NO",
    ExpressionLabel.HELP: "HELP",
    ExpressionLabel.PAIN: "PAIN",
    ExpressionLabel.THIRSTY: "THIRSTY",
    ExpressionLabel.HUNGRY: "HUNGRY",
    ExpressionLabel.TIRED: "TIRED",
}

class VideoThread(QThread):
    """Thread for video capture and processing."""
    frame_ready = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.face_pipeline = FaceLandmarkPipeline(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.classifier = create_classifier(model_type="rf")
        self.intent_mapper = create_mapper()
        self.occlusion_handler = RobustOcclusionHandler()

        model_loaded = False
        for candidate_path, mtype in [
            ("checkpoints/expression_occlusion.pt", "occlusion_aware"),
            ("checkpoints/expression_temporal.pt", "temporal"),
            ("checkpoints/expression_transformer.pt", "transformer"),
            ("checkpoints/expression_model.joblib", "rf"),
        ]:
            if os.path.exists(candidate_path):
                try:
                    if candidate_path.endswith(".pt"):
                        self.classifier = create_classifier(model_type=mtype, model_path=candidate_path)
                    else:
                        self.classifier = create_classifier(model_type="rf")
                        self.classifier.load(candidate_path)
                    print(f"Loaded trained model from {candidate_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"Could not load {candidate_path}: {e}")

        if not model_loaded:
            print("No trained model found. Using rule-based fallback.")

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            self.error.emit(f"Could not open camera {self.camera_id}")
            return

        fps_history = []
        last_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            start_time = time.time()
            result = self._process_frame(frame)
            processing_time = time.time() - start_time

            fps_history.append(1.0 / max(processing_time, 1e-6))
            if len(fps_history) > 30:
                fps_history.pop(0)
            result['fps'] = np.mean(fps_history)

            self.frame_ready.emit(frame, result)

        cap.release()

    def _process_frame(self, frame: np.ndarray) -> dict:
        result = {
            'annotated_frame': frame.copy(),
            'expression': None,
            'intent': None,
            'occlusion': None,
            'features': {},
            'fps': 0,
        }

        face_result = self.face_pipeline.process_and_extract(frame)
        if face_result is None:
            return result

        landmarks_obj = face_result['landmarks']
        features = face_result['features']
        result['features'] = features

        adapted_landmarks, occlusion = self.occlusion_handler.process(
            landmarks_obj.landmarks,
            use_mask_adaptation=True,
        )
        result['occlusion'] = occlusion

        raw_lm = normalize_landmarks(landmarks_obj.landmarks).flatten().tolist() if hasattr(landmarks_obj, 'landmarks') else None
        expression = self.classifier.predict(features, raw_landmarks=raw_lm)
        result['expression'] = expression

        intent = self.intent_mapper.map_to_intent(expression)
        result['intent'] = intent

        result['annotated_frame'] = self.face_pipeline.detector.draw_landmarks(
            frame.copy(),
            landmarks_obj,
            draw_connections=True,
        )

        return result

    def stop(self):
        self.running = False
        self.face_pipeline.close()

class ExpressionWidget(QFrame):
    """Widget to display expression information."""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)

        layout = QVBoxLayout(self)

        title = QLabel("EXPRESSION")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.expression_label = QLabel("--")
        self.expression_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.expression_label.setAlignment(Qt.AlignCenter)
        self.expression_label.setStyleSheet("color: white;")
        layout.addWidget(self.expression_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        layout.addWidget(self.confidence_bar)

        self.confidence_label = QLabel("Confidence: 0.00")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)

        prob_group = QGroupBox("Probabilities")
        prob_layout = QGridLayout()
        self.prob_labels = {}
        row, col = 0, 0
        for label in ExpressionLabel:
            lbl = QLabel(f"{label.value}: 0.00")
            lbl.setFont(QFont("Arial", 9))
            self.prob_labels[label] = lbl
            prob_layout.addWidget(lbl, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        prob_group.setLayout(prob_layout)
        layout.addWidget(prob_group)

        self.setMinimumWidth(200)

    def update(self, expression, occlusion):
        if expression is None:
            self.expression_label.setText("--")
            self.confidence_bar.setValue(0)
            self.confidence_label.setText("Confidence: 0.00")
            return

        color = EXPRESSION_COLORS.get(expression.label, QColor(255, 255, 255))
        self.expression_label.setText(expression.label.value.upper())
        self.expression_label.setStyleSheet(f"color: {color.name()};")

        self.confidence_bar.setValue(int(expression.confidence * 100))
        self.confidence_label.setText(f"Confidence: {expression.confidence:.2f}")

        for label, prob in expression.probabilities.items():
            if label in self.prob_labels:
                self.prob_labels[label].setText(f"{label.value}: {prob:.2f}")

class IntentWidget(QFrame):
    """Widget to display intent information."""

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumWidth(150)

        layout = QVBoxLayout(self)

        title = QLabel("INTENT")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.intent_label = QLabel("--")
        self.intent_label.setFont(QFont("Arial", 32, QFont.Bold))
        self.intent_label.setAlignment(Qt.AlignCenter)
        self.intent_label.setMinimumHeight(80)
        self.intent_label.setStyleSheet("color: white; background-color: #333;")
        layout.addWidget(self.intent_label)

        self.intent_desc = QLabel("No active intent")
        self.intent_desc.setAlignment(Qt.AlignCenter)
        self.intent_desc.setWordWrap(True)
        layout.addWidget(self.intent_desc)

    def update(self, intent):
        if intent is None or intent.label not in INTENT_LABELS:
            self.intent_label.setText("--")
            self.intent_desc.setText("No active intent")
            return

        color = EXPRESSION_COLORS.get(intent.label, QColor(255, 255, 255))
        symbol = INTENT_LABELS[intent.label]
        self.intent_label.setText(symbol)
        self.intent_label.setStyleSheet(
            f"color: {color.name()}; background-color: #333;"
        )
        self.intent_desc.setText(f"Confidence: {intent.confidence:.2f}")

class ControlPanel(QFrame):
    """Control panel for settings."""

    def __init__(self, on_camera_change, on_model_load, on_reset):
        super().__init__()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setLineWidth(2)

        layout = QVBoxLayout(self)

        title = QLabel("CONTROLS")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0", "1", "2", "3"])
        self.camera_combo.currentTextChanged.connect(on_camera_change)
        camera_layout.addWidget(self.camera_combo)
        layout.addLayout(camera_layout)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(on_model_load)
        layout.addWidget(self.load_btn)

        self.reset_btn = QPushButton("Reset Intent History")
        self.reset_btn.clicked.connect(on_reset)
        layout.addWidget(self.reset_btn)

        self.show_landmarks = QCheckBox("Show Landmarks")
        self.show_landmarks.setChecked(True)
        layout.addWidget(self.show_landmarks)

        self.show_expression = QCheckBox("Show Expression")
        self.show_expression.setChecked(True)
        layout.addWidget(self.show_expression)

        layout.addStretch()

class StatusBar(QStatusBar):
    """Status bar at the bottom."""

    def __init__(self):
        super().__init__()
        self.fps_label = QLabel("FPS: --")
        self.status_label = QLabel("Ready")
        self.addWidget(self.fps_label)
        self.addWidget(self.status_label)

    def update_fps(self, fps: float):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_status(self, status: str):
        self.status_label.setText(status)

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceVox: Facial Expression Recognition")
        self.setMinimumSize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        main_layout.addLayout(video_layout, 3)

        right_layout = QVBoxLayout()

        self.control_panel = ControlPanel(
            on_camera_change=self._on_camera_change,
            on_model_load=self._on_model_load,
            on_reset=self._on_reset,
        )
        right_layout.addWidget(self.control_panel)

        self.expression_widget = ExpressionWidget()
        right_layout.addWidget(self.expression_widget)

        self.intent_widget = IntentWidget()
        right_layout.addWidget(self.intent_widget)

        main_layout.addLayout(right_layout, 1)

        self.status_bar = StatusBar()
        self.setStatusBar(self.status_bar)

        self.video_thread = VideoThread(camera_id=0)
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.error.connect(self._on_error)
        self.video_thread.start()

    def _on_frame(self, frame: np.ndarray, result: dict):
        """Handle new frame from video thread."""
        if self.control_panel.show_landmarks.isChecked():
            display = result['annotated_frame']
        else:
            display = frame.copy()

        h, w, ch = display.shape
        bytes_per_line = ch * w
        rgb_image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

        if self.control_panel.show_expression.isChecked():
            self.expression_widget.update(result['expression'], result['occlusion'])
        self.intent_widget.update(result['intent'])

        self.status_bar.update_fps(result['fps'])

    def _on_camera_change(self, camera_id: str):
        """Handle camera change."""
        self.video_thread.stop()
        self.video_thread = VideoThread(camera_id=int(camera_id))
        self.video_thread.frame_ready.connect(self._on_frame)
        self.video_thread.error.connect(self._on_error)
        self.video_thread.start()
        self.status_bar.update_status(f"Switched to camera {camera_id}")

    def _on_model_load(self):
        model_loaded = False
        for candidate_path, mtype in [
            ("checkpoints/expression_occlusion.pt", "occlusion_aware"),
            ("checkpoints/expression_temporal.pt", "temporal"),
            ("checkpoints/expression_transformer.pt", "transformer"),
            ("checkpoints/expression_model.joblib", "rf"),
        ]:
            if os.path.exists(candidate_path):
                try:
                    if candidate_path.endswith(".pt"):
                        self.video_thread.classifier = create_classifier(
                            model_type=mtype, model_path=candidate_path
                        )
                    else:
                        self.video_thread.classifier.load(candidate_path)
                    self.status_bar.update_status(f"Loaded model from {candidate_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    self.status_bar.update_status(f"Error loading {candidate_path}: {e}")
        if not model_loaded:
            self.status_bar.update_status("No trained model found")

    def _on_reset(self):
        """Handle reset button."""
        self.video_thread.intent_mapper.intent_history.clear()
        self.status_bar.update_status("Intent history reset")

    def _on_error(self, error: str):
        """Handle error from video thread."""
        self.status_bar.update_status(f"Error: {error}")

    def closeEvent(self, event):
        """Clean up on close."""
        self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QWidget {
            background-color: #333;
            color: white;
        }
        QFrame {
            background-color: #3a3a3a;
            border-radius: 5px;
        }
        QPushButton {
            background-color: #4a4a4a;
            border: 1px solid #555;
            padding: 8px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #5a5a5a;
        }
        QLabel {
            color: white;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QComboBox {
            background-color: #4a4a4a;
            border: 1px solid #555;
            padding: 5px;
        }
        QCheckBox {
            color: white;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
