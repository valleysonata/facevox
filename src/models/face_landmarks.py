import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import os
import urllib.request

@dataclass
class FaceLandmarks:
    landmarks: np.ndarray  # (478, 3) with x, y, z in pixel coordinates
    image_shape: Tuple[int, int]  # (height, width)
    confidence: float
    timestamp: float
    occlusion_mask: Optional[np.ndarray] = None  # (478,) boolean

class MediaPipeFaceLandmarker:
    """MediaPipe-based facial landmark detection (478 landmarks)."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        # Download model if not present
        model_path = os.path.join("checkpoints", "face_landmarker.task")
        os.makedirs("checkpoints", exist_ok=True)
        if not os.path.exists(model_path):
            print("Downloading face_landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("Model downloaded.")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO if not static_image_mode else RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
        )

        self.face_landmarker = FaceLandmarker.create_from_options(options)
        self.static_image_mode = static_image_mode
        self.timestamp_counter = 0

        # Drawing connections (manual - mediapipe new API doesn't have drawing utils)
        self.mp_face_mesh_connections = mp.tasks.vision.FaceLandmarksConnections

        # Key landmark indices for expressions
        self.LIPS_INNER = [13, 14, 15, 16, 17, 78, 80, 81, 82, 87, 88, 91, 95]
        self.LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

    def process(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        """Process image and return face landmarks."""
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        h, w = image.shape[:2]

        if self.static_image_mode:
            result = self.face_landmarker.detect(mp_image)
        else:
            self.timestamp_counter += 1
            result = self.face_landmarker.detect_for_video(mp_image, self.timestamp_counter)

        if not result.face_landmarks:
            return None

        # Take the first face
        face_lm = result.face_landmarks[0]

        # Convert to numpy array (478, 3) in pixel coordinates
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w] for lm in face_lm
        ], dtype=np.float32)

        return FaceLandmarks(
            landmarks=landmarks,
            image_shape=(h, w),
            confidence=1.0,
            timestamp=self.timestamp_counter / 30.0,
        )

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """Draw landmarks on image."""
        annotated = image.copy()
        h, w = image.shape[:2]

        # Draw individual landmarks as small circles
        for i, (x, y, z) in enumerate(landmarks.landmarks):
            cv2.circle(annotated, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Draw some key connections
        if draw_connections:
            for conn in self.mp_face_mesh_connections.FACE_LANDMARKS_LEFT_EYE:
                start, end = conn.start, conn.end
                if start < len(landmarks.landmarks) and end < len(landmarks.landmarks):
                    pt1 = (int(landmarks.landmarks[start][0]), int(landmarks.landmarks[start][1]))
                    pt2 = (int(landmarks.landmarks[end][0]), int(landmarks.landmarks[end][1]))
                    cv2.line(annotated, pt1, pt2, (0, 200, 255), 1)

            for conn in self.mp_face_mesh_connections.FACE_LANDMARKS_RIGHT_EYE:
                start, end = conn.start, conn.end
                if start < len(landmarks.landmarks) and end < len(landmarks.landmarks):
                    pt1 = (int(landmarks.landmarks[start][0]), int(landmarks.landmarks[start][1]))
                    pt2 = (int(landmarks.landmarks[end][0]), int(landmarks.landmarks[end][1]))
                    cv2.line(annotated, pt1, pt2, (0, 200, 255), 1)

            for conn in self.mp_face_mesh_connections.FACE_LANDMARKS_LIPS:
                start, end = conn.start, conn.end
                if start < len(landmarks.landmarks) and end < len(landmarks.landmarks):
                    pt1 = (int(landmarks.landmarks[start][0]), int(landmarks.landmarks[start][1]))
                    pt2 = (int(landmarks.landmarks[end][0]), int(landmarks.landmarks[end][1]))
                    cv2.line(annotated, pt1, pt2, (0, 0, 255), 1)

        return annotated

    def extract_features(self, landmarks: FaceLandmarks) -> Dict[str, float]:
        """Extract geometric features for expression recognition."""
        lm = landmarks.landmarks
        features = {}

        # Mouth opening (vertical distance between upper/lower lip)
        upper_lip = lm[13]
        lower_lip = lm[14]
        mouth_open = float(np.linalg.norm(upper_lip[:2] - lower_lip[:2]))
        features['mouth_open'] = mouth_open

        # Mouth width
        left_corner = lm[61]
        right_corner = lm[291]
        mouth_width = float(np.linalg.norm(left_corner[:2] - right_corner[:2]))
        features['mouth_width'] = mouth_width

        # Lip height (average)
        lip_heights = []
        for i in range(len(self.LIPS_INNER) // 2):
            upper = lm[self.LIPS_INNER[i]]
            lower = lm[self.LIPS_INNER[-(i + 1)]]
            lip_heights.append(float(np.linalg.norm(upper[:2] - lower[:2])))
        features['lip_height_avg'] = float(np.mean(lip_heights))

        # Eye features
        left_eye = lm[self.LEFT_EYE]
        right_eye = lm[self.RIGHT_EYE]

        def eye_aspect_ratio(eye_pts):
            v1 = np.linalg.norm(eye_pts[1][:2] - eye_pts[5][:2])
            v2 = np.linalg.norm(eye_pts[2][:2] - eye_pts[4][:2])
            h = np.linalg.norm(eye_pts[0][:2] - eye_pts[3][:2])
            return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.0

        features['left_ear'] = eye_aspect_ratio(left_eye)
        features['right_ear'] = eye_aspect_ratio(right_eye)
        features['ear_avg'] = (features['left_ear'] + features['right_ear']) / 2

        # Eyebrow features
        left_brow = lm[self.LEFT_EYEBROW]
        right_brow = lm[self.RIGHT_EYEBROW]
        left_eye_top = float(np.mean(left_eye[[1, 2]][:, 1]))
        right_eye_top = float(np.mean(right_eye[[1, 2]][:, 1]))
        left_brow_top = float(np.mean(left_brow[:, 1]))
        right_brow_top = float(np.mean(right_brow[:, 1]))

        features['left_brow_height'] = left_eye_top - left_brow_top
        features['right_brow_height'] = right_eye_top - right_brow_top

        # Head pose (approximate)
        nose_tip = lm[1]
        chin = lm[152]
        left_cheek = lm[234]
        right_cheek = lm[454]

        features['pitch'] = float(nose_tip[1] - chin[1])
        features['yaw'] = float(right_cheek[0] - left_cheek[0])

        return features

    def close(self):
        self.face_landmarker.close()

class FaceLandmarkPipeline:
    """High-level pipeline for face landmark detection and feature extraction."""

    def __init__(self, **kwargs):
        self.detector = MediaPipeFaceLandmarker(**kwargs)
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        """Process a single frame."""
        landmarks = self.detector.process(frame)
        if landmarks:
            self.frame_count += 1
        return landmarks

    def process_and_extract(self, frame: np.ndarray) -> Optional[Dict]:
        """Process frame and extract features."""
        landmarks = self.process_frame(frame)
        if landmarks is None:
            return None

        features = self.detector.extract_features(landmarks)
        annotated = self.detector.draw_landmarks(frame, landmarks)

        return {
            'landmarks': landmarks,
            'features': features,
            'annotated_frame': annotated,
        }

    def close(self):
        self.detector.close()

def create_pipeline(**kwargs) -> FaceLandmarkPipeline:
    """Factory function to create a face landmark pipeline."""
    return FaceLandmarkPipeline(**kwargs)
