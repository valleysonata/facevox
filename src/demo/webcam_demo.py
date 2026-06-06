import cv2
import time
import os
import numpy as np
from typing import Optional
from src.models.face_landmarks import FaceLandmarkPipeline
from src.models.expression_recognition import (
    ExpressionClassifier,
    AssistiveExpressionMapper,
    ExpressionLabel,
    create_classifier,
    create_mapper,
)
from src.models.occlusion import RobustOcclusionHandler

EXPRESSION_COLORS = {
    ExpressionLabel.NEUTRAL: (200, 200, 200),
    ExpressionLabel.HAPPY: (0, 255, 0),
    ExpressionLabel.SAD: (255, 0, 0),
    ExpressionLabel.SURPRISED: (0, 255, 255),
    ExpressionLabel.ANGRY: (0, 0, 255),
    ExpressionLabel.DISGUSTED: (0, 128, 0),
    ExpressionLabel.FEARFUL: (128, 0, 128),
    ExpressionLabel.CONFUSED: (128, 128, 0),
    ExpressionLabel.THINKING: (64, 64, 64),
    ExpressionLabel.YES: (0, 255, 0),
    ExpressionLabel.NO: (255, 0, 0),
    ExpressionLabel.HELP: (0, 0, 255),
    ExpressionLabel.PAIN: (255, 0, 255),
    ExpressionLabel.THIRSTY: (255, 200, 0),
    ExpressionLabel.HUNGRY: (200, 100, 0),
    ExpressionLabel.TIRED: (100, 100, 200),
}

INTENT_SYMBOLS = {
    ExpressionLabel.YES: "YES",
    ExpressionLabel.NO: "NO",
    ExpressionLabel.HELP: "HELP",
    ExpressionLabel.PAIN: "PAIN",
    ExpressionLabel.THIRSTY: "THIRSTY",
    ExpressionLabel.HUNGRY: "HUNGRY",
    ExpressionLabel.TIRED: "TIRED",
}

class WebcamDemo:
    """Real-time webcam facial expression recognition demo."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_id: int = 0,
        window_name: str = "FaceVox: Facial Expression Recognition",
        show_landmarks: bool = True,
        show_expression: bool = True,
        show_intent: bool = True,
        show_fps: bool = True,
    ):
        self.window_name = window_name
        self.show_landmarks = show_landmarks
        self.show_expression = show_expression
        self.show_intent = show_intent
        self.show_fps = show_fps

        self.face_pipeline = FaceLandmarkPipeline(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.classifier = create_classifier(model_type="rf")

        model_loaded = False
        for candidate_path in [
            model_path,
            "checkpoints/expression_transformer.pt",
            "checkpoints/expression_model.joblib",
        ]:
            if candidate_path and os.path.exists(candidate_path):
                try:
                    if candidate_path.endswith(".pt"):
                        self.classifier = create_classifier(model_type="transformer", model_path=candidate_path)
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

        self.intent_mapper = create_mapper()

        # Occlusion handler
        self.occlusion_handler = RobustOcclusionHandler()

        self.camera_id = camera_id
        self.cap = None

        self.fps_history = []
        self.last_time = time.time()

    def start(self):
        """Start the webcam demo."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return

        print("Press 'q' to quit, 's' to save screenshot, 'r' to reset intent history")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            start_time = time.time()
            result = self._process_frame(frame)
            processing_time = time.time() - start_time

            self.fps_history.append(1.0 / max(processing_time, 1e-6))
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            fps = np.mean(self.fps_history)

            display = self._draw_ui(frame, result, fps)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_screenshot(display)
            elif key == ord('r'):
                self.intent_mapper.intent_history.clear()
                print("Intent history reset")

        self.cleanup()

    def _process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame through the pipeline."""
        result = {
            'annotated_frame': frame.copy(),
            'expression': None,
            'intent': None,
            'occlusion': None,
            'features': None,
        }

        face_result = self.face_pipeline.process_and_extract(frame)
        if face_result is None:
            return result

        landmarks_obj = face_result['landmarks']
        features = face_result['features']
        annotated = face_result['annotated_frame']

        adapted_landmarks, occlusion = self.occlusion_handler.process(
            landmarks_obj.landmarks,
            use_mask_adaptation=True,
        )

        result['occlusion'] = occlusion
        result['features'] = features
        result['annotated_frame'] = annotated

        raw_lm = landmarks_obj.landmarks.flatten().tolist() if hasattr(landmarks_obj, 'landmarks') else None
        expression = self.classifier.predict(features, raw_landmarks=raw_lm)
        result['expression'] = expression

        intent = self.intent_mapper.map_to_intent(expression)
        result['intent'] = intent

        return result

    def _draw_ui(self, frame: np.ndarray, result: dict, fps: float) -> np.ndarray:
        """Draw the UI overlay."""
        display = result['annotated_frame'].copy()
        h, w = display.shape[:2]

        if self.show_fps:
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if self.show_expression and result['expression']:
            expr = result['expression']
            color = EXPRESSION_COLORS.get(expr.label, (255, 255, 255))
            text = f"Expression: {expr.label.value.upper()}"
            cv2.putText(display, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            conf_text = f"Confidence: {expr.confidence:.2f}"
            cv2.putText(display, conf_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # Top 3 probabilities
            sorted_probs = sorted(
                expr.probabilities.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            y_offset = 130
            for label, prob in sorted_probs:
                prob_text = f"{label.value}: {prob:.2f}"
                cv2.putText(display, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25

        if self.show_intent and result['intent']:
            intent = result['intent']
            if intent.label in INTENT_SYMBOLS:
                symbol = INTENT_SYMBOLS[intent.label]
                color = EXPRESSION_COLORS.get(intent.label, (255, 255, 255))

                box_x = w - 200
                box_y = 20
                box_w = 180
                box_h = 80
                cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), color, -1)
                cv2.putText(
                    display,
                    symbol,
                    (box_x + 10, box_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    display,
                    "INTENT",
                    (box_x + 10, box_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )

        if result['occlusion'] and result['occlusion'].is_occluded:
            cv2.putText(
                display,
                f"Occlusion: {result['occlusion'].occluded_regions}",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )

        cv2.putText(
            display,
            "q: quit | s: screenshot | r: reset intent",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

        return display

    def _save_screenshot(self, frame: np.ndarray):
        """Save a screenshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.face_pipeline.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FaceVox Webcam Demo")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--no-landmarks", action="store_true", help="Hide landmarks")
    parser.add_argument("--no-expression", action="store_true", help="Hide expression")
    parser.add_argument("--no-intent", action="store_true", help="Hide intent")
    args = parser.parse_args()

    demo = WebcamDemo(
        model_path=args.model,
        camera_id=args.camera,
        show_landmarks=not args.no_landmarks,
        show_expression=not args.no_expression,
        show_intent=not args.no_intent,
    )
    demo.start()

if __name__ == "__main__":
    main()
