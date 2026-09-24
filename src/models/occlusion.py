import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OcclusionEstimate:
    is_occluded: bool
    occlusion_ratio: float
    occluded_regions: List[str]
    confidence: float


# Facial landmark regions for occlusion checking
FACE_REGIONS = {
    'left_eye': list(range(33, 160)),
    'right_eye': list(range(362, 400)),
    'nose': list(range(1, 10)),
    'mouth': list(range(61, 68)),
    'left_cheek': [234],
    'right_cheek': [454],
    'forehead': list(range(10, 20)),
    'chin': [152],
}

REGION_CHECK = {
    'left_eye': list(range(33, 160)),
    'right_eye': list(range(362, 400)),
    'nose_bridge': [6, 168, 8, 9],
    'upper_lip': list(range(13, 18)),
    'lower_lip': list(range(14, 18)),
    'left_brow': list(range(46, 70)),
    'right_brow': list(range(276, 300)),
}


class OcclusionDetector:
    """Detect facial occlusions from landmark geometry and visibility."""

    def __init__(
        self,
        z_threshold: float = 0.06,
        visibility_threshold: float = 0.5,
        symmetry_threshold: float = 0.15,
    ):
        self.z_threshold = z_threshold
        self.visibility_threshold = visibility_threshold
        self.symmetry_threshold = symmetry_threshold

    def detect(
        self,
        landmarks: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> OcclusionEstimate:
        """Detect occlusion from landmark positions."""
        occluded_regions = []

        z = landmarks[:, 2]
        z_mean = np.median(z)
        z_std = np.std(z)

        # Symmetry check
        left_x = landmarks[234, 0]
        right_x = landmarks[454, 0]
        nose_x = landmarks[1, 0]
        nose_depth = abs(nose_x - (left_x + right_x) / 2) / (abs(right_x - left_x) + 1e-6)

        # Z-depth anomalies
        for region_name, indices in REGION_CHECK.items():
            region_z = np.abs(z[indices] - z_mean)
            if np.mean(region_z) > self.z_threshold * 3:
                occluded_regions.append(region_name)

        # Face aspect ratio check
        left_eye = landmarks[33:37]
        right_eye = landmarks[362:366]
        eye_dist = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        nose_to_chin = np.linalg.norm(landmarks[1] - landmarks[152])

        if eye_dist > 0:
            actual_ratio = eye_dist / (nose_to_chin + 1e-6)
            if actual_ratio > 0.5 * 1.3:
                occluded_regions.append('face_wide')
            elif actual_ratio < 0.5 * 0.7:
                occluded_regions.append('face_narrow')

        # Z-depth discontinuities
        z_diff = np.abs(np.diff(z))
        if len(z_diff) > 0 and np.max(z_diff) > self.z_threshold * 5:
            discontinuity_idx = np.argmax(z_diff)
            for region_name, indices in REGION_CHECK.items():
                if discontinuity_idx in indices:
                    occluded_regions.append(f'{region_name}_discontinuity')
                    break

        # Calculate occlusion ratio
        total_landmarks = len(landmarks)
        occluded_count = sum(
            len(REGION_CHECK.get(r, []))
            for r in occluded_regions
            if r in REGION_CHECK
        )
        occlusion_ratio = min(1.0, occluded_count / total_landmarks)
        is_occluded = occlusion_ratio > 0.1 or len(occluded_regions) > 0

        # Confidence
        checks = [
            nose_depth < self.symmetry_threshold,
            occlusion_ratio > 0.05,
            len(occluded_regions) > 0,
        ]
        confidence = sum(checks) / len(checks) if checks else 0.0

        return OcclusionEstimate(
            is_occluded=is_occluded,
            occlusion_ratio=occlusion_ratio,
            occluded_regions=occluded_regions,
            confidence=confidence,
        )


class MaskOcclusionAdapter:
    """Adapt landmark estimation when face is partially occluded (e.g., mask)."""

    MASK_VISIBLE = {
        'forehead': list(range(10, 20)),
        'left_brow': list(range(46, 70)),
        'right_brow': list(range(276, 300)),
        'left_eye': list(range(33, 160)),
        'right_eye': list(range(362, 400)),
        'nose_bridge': [6, 168, 8, 9],
    }

    def create_visibility_mask(self, landmarks: np.ndarray) -> np.ndarray:
        """Create visibility mask for mask occlusion."""
        visibility = np.ones(len(landmarks), dtype=np.float32)
        all_mask_indices = []
        for indices in self.MASK_VISIBLE.values():
            all_mask_indices.extend(indices)
        for i in range(len(landmarks)):
            if i not in all_mask_indices:
                visibility[i] *= 0.3
        return visibility

    def interpolate_occluded(
        self,
        landmarks: np.ndarray,
        visibility: np.ndarray,
        template: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Interpolate occluded landmarks using visible ones."""
        if template is None:
            return self._geometric_interpolate(landmarks, visibility)
        result = landmarks.copy()
        mask = visibility < 0.5
        alpha = 1.0 - visibility[mask][:, np.newaxis]
        result[mask] = landmarks[mask] * (1 - alpha) + template[mask] * alpha
        return result

    def _geometric_interpolate(
        self,
        landmarks: np.ndarray,
        visibility: np.ndarray,
    ) -> np.ndarray:
        """Simple geometric interpolation for occluded points."""
        result = landmarks.copy()
        occluded = visibility < 0.5
        visible = ~occluded
        if np.sum(visible) < 10:
            return result
        for i in range(len(landmarks)):
            if occluded[i]:
                visible_positions = landmarks[visible]
                distances = np.linalg.norm(visible_positions - landmarks[i], axis=1)
                k = min(5, len(visible_positions))
                nearest_indices = np.argsort(distances)[:k]
                nearest_weights = 1.0 / (distances[nearest_indices] + 1e-6)
                nearest_weights /= nearest_weights.sum()
                result[i] = np.sum(visible_positions[nearest_indices] * nearest_weights[:, np.newaxis], axis=0)
        return result


class RobustOcclusionHandler:
    """Combined occlusion detection and adaptation."""

    def __init__(self):
        self.detector = OcclusionDetector()
        self.adapter = MaskOcclusionAdapter()
        self.occlusion_history = []

    def process(
        self,
        landmarks: np.ndarray,
        use_mask_adaptation: bool = True,
    ) -> Tuple[np.ndarray, OcclusionEstimate]:
        """Process landmarks with occlusion handling."""
        occlusion = self.detector.detect(landmarks)
        self.occlusion_history.append(occlusion.is_occluded)
        if len(self.occlusion_history) > 30:
            self.occlusion_history.pop(0)

        if occlusion.is_occluded and use_mask_adaptation:
            visibility = self.adapter.create_visibility_mask(landmarks)
            adapted = self.adapter.interpolate_occluded(landmarks, visibility)
            return adapted, occlusion

        return landmarks, occlusion
