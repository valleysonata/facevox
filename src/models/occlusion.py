import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class OcclusionEstimate:
    is_occluded: bool
    occlusion_ratio: float
    occluded_regions: List[str]
    confidence: float

# Facial landmark regions
FACE_REGIONS = {
    'left_eye': list(range(33, 37)),
    'right_eye': list(range(362, 366)),
    'nose': list(range(1, 10)),
    'mouth': list(range(61, 68)),
    'left_cheek': [234],
    'right_cheek': [454],
    'forehead': list(range(10, 20)),
    'chin': [152],
}

# Landmark indices per region for occlusion checking
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

        # Method 1: Z-depth anomalies
        z = landmarks[:, 2]
        z_mean = np.median(z)
        z_std = np.std(z)

        # Method 2: Face symmetry check
        left_x = landmarks[234, 0]   # left cheek
        right_x = landmarks[454, 0]  # right cheek
        nose_x = landmarks[1, 0]     # nose center
        nose_depth = abs(nose_x - (left_x + right_x) / 2) / (right_x - left_x + 1e-6)

        # Method 3: Landmark visibility (if z is too large, landmark is hidden)
        for region_name, indices in REGION_CHECK.items():
            region_z = np.abs(z[indices] - z_mean)
            if np.mean(region_z) > self.z_threshold * 3:
                occluded_regions.append(region_name)

        # Method 4: Inter-landmark distance anomalies
        left_eye = landmarks[33:37]
        right_eye = landmarks[362:366]
        eye_dist = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
        nose_to_chin = np.linalg.norm(landmarks[1] - landmarks[152])

        # Expected ratio (face is roughly 1.3x wide as tall from eyes to chin)
        if eye_dist > 0:
            expected_ratio = 0.5
            actual_ratio = eye_dist / (nose_to_chin + 1e-6)
            if actual_ratio > expected_ratio * 1.3:
                occluded_regions.append('face_wide')
            elif actual_ratio < expected_ratio * 0.7:
                occluded_regions.append('face_narrow')

        # Method 5: Z-depth discontinuities
        z_diff = np.abs(np.diff(z))
        if np.max(z_diff) > self.z_threshold * 5:
            # Find where the discontinuity occurs
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

        # Determine if occluded
        is_occluded = occlusion_ratio > 0.1 or len(occluded_regions) > 0

        # Confidence based on multiple checks agreeing
        checks = [
            nose_depth < self.symmetry_threshold,
            occlusion_ratio > 0.05,
            len(occluded_regions) > 0,
        ]
        confidence = sum(checks) / len(checks)

        return OcclusionEstimate(
            is_occluded=is_occluded,
            occlusion_ratio=occlusion_ratio,
            occluded_regions=occluded_regions,
            confidence=confidence,
        )

class MaskOcclusionAdapter:
    """Adapt landmark estimation when face is partially occluded (e.g., mask)."""

    # Visible landmarks when wearing a mask
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

        # Set occluded regions to low visibility
        all_mask_indices = []
        for indices in self.MASK_VISIBLE.values():
            all_mask_indices.extend(indices)

        for i in range(len(landmarks)):
            if i not in all_mask_indices:
                visibility[i] *= 0.3  # Reduce weight for occluded regions

        return visibility

    def interpolate_occluded(
        self,
        landmarks: np.ndarray,
        visibility: np.ndarray,
        template: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Interpolate occluded landmarks using visible ones."""
        if template is None:
            # Use simple geometric interpolation
            return self._geometric_interpolate(landmarks, visibility)

        # Blend with template for occluded regions
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

        # For each occluded point, find nearest visible points
        for i in range(len(landmarks)):
            if occluded[i]:
                # Find k nearest visible landmarks
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
        # Detect occlusion
        occlusion = self.detector.detect(landmarks)

        # Track history
        self.occlusion_history.append(occlusion.is_occluded)
        if len(self.occlusion_history) > 30:
            self.occlusion_history.pop(0)

        # If occluded and mask adaptation enabled, interpolate
        if occlusion.is_occluded and use_mask_adaptation:
            visibility = self.adapter.create_visibility_mask(landmarks)
            adapted = self.adapter.interpolate_occluded(landmarks, visibility)
            return adapted, occlusion

        return landmarks, occlusion
