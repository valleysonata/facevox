"""Smoke tests for FaceVox. No camera, no network, no downloads.

Run:  python tests/test_smoke.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

passed = []


def check(name, fn):
    fn()
    passed.append(name)
    print(f"  PASS {name}")


def test_imports():
    from src.models import (  # noqa
        FaceLandmarkPipeline,
        MediaPipeFaceLandmarker,
        normalize_landmarks,
        ExpressionClassifier,
        AssistiveExpressionMapper,
        ExpressionLabel,
        ExpressionResult,
        create_classifier,
        create_mapper,
        LandmarkTransformer,
        TemporalExpressionTransformer,
        OcclusionAwareClassifier,
        LandmarkSequenceTransformer,
        RobustOcclusionHandler,
        OcclusionDetector,
    )
    from src.demo.webcam_demo import WebcamDemo  # noqa
    from src.demo.server import app  # noqa
    from src.utils.training import (  # noqa
        ExpressionDatasetBuilder,
        ExpressionTrainer,
    )


def test_synthetic_dataset():
    from src.utils.training import ExpressionDatasetBuilder
    from src.models.expression_recognition import FeatureExtractor

    ds = ExpressionDatasetBuilder(output_dir=tempfile.mkdtemp())
    ds.generate_synthetic(samples_per_class=5)
    assert len(ds.data) == 45, f"expected 45 samples, got {len(ds.data)}"
    assert set(ds.labels) == set(range(9))
    for d in ds.data:
        for name in FeatureExtractor.FEATURE_NAMES:
            assert name in d, f"missing feature {name}"
        assert "_raw_landmarks" in d
        assert len(d["_raw_landmarks"]) == 478 * 3


def test_normalize_landmarks():
    from src.models.face_landmarks import normalize_landmarks

    rng = np.random.default_rng(0)
    lm = rng.normal(0, 100, (478, 3)).astype(np.float32)
    lm[1] = [320.0, 240.0, 0.0]
    lm[33:37, 0] -= 30.0
    lm[263:267, 0] += 30.0
    out = normalize_landmarks(lm)
    assert out.shape == (478, 3)
    assert np.allclose(out[1], 0.0, atol=1e-5), "nose tip should be centered"
    eye = np.linalg.norm(out[263:267].mean(axis=0) - out[33:37].mean(axis=0))
    assert abs(eye - 1.0) < 1e-4, f"eye distance should be 1, got {eye}"


def test_rule_based_and_mapper():
    from src.models.expression_recognition import (
        ExpressionClassifier,
        ExpressionLabel,
        create_mapper,
    )

    clf = ExpressionClassifier(model_type="rf")
    assert not clf.is_trained
    features = {
        "mouth_open": 3.0, "mouth_width": 50.0, "lip_height_avg": 2.0,
        "left_ear": 0.25, "right_ear": 0.25, "ear_avg": 0.25,
        "left_brow_height": 1.0, "right_brow_height": 1.0,
        "pitch": 0.0, "yaw": 0.0,
    }
    result = clf.predict(features)
    assert isinstance(result.label, ExpressionLabel)
    assert 0.0 <= result.confidence <= 1.0

    mapper = create_mapper()
    happy = clf._predict_rules(
        {**features, "mouth_open": 8.0, "mouth_width": 62.0}
    )
    before = happy.label
    intent = mapper.map_to_intent(happy)
    assert happy.label == before, "mapper must not mutate its input"
    assert intent.label == ExpressionLabel.YES, f"happy -> {intent.label}"


def test_occlusion_detector():
    from src.models.occlusion import RobustOcclusionHandler

    rng = np.random.default_rng(1)
    lm = rng.normal(0, 50, (478, 3)).astype(np.float32)
    handler = RobustOcclusionHandler()
    adapted, estimate = handler.process(lm)
    assert adapted.shape == (478, 3)
    assert isinstance(estimate.is_occluded, bool)
    assert 0.0 <= estimate.occlusion_ratio <= 1.0


def test_rf_train_save_load_evaluate():
    from src.utils.training import ExpressionDatasetBuilder, ExpressionTrainer

    ds = ExpressionDatasetBuilder(output_dir=tempfile.mkdtemp())
    ds.generate_synthetic(samples_per_class=20)
    trainer = ExpressionTrainer(model_type="rf")
    X, y = trainer.prepare_data(dataset=ds)
    assert X.shape == (180, 10)
    metrics = trainer.train(X, y)
    # Synthetic classes overlap by design; assert well above chance (1/9).
    assert metrics["test_accuracy"] > 0.3, metrics
    assert "classification_report" in metrics

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        trainer.save(path)
        trainer2 = ExpressionTrainer(model_type="rf")
        trainer2.load(path)
        eval_metrics = trainer2.evaluate(X, y)
        assert eval_metrics["test_accuracy"] > 0.3, eval_metrics


def test_transformer_trains_on_synthetic_raw():
    from src.utils.training import ExpressionDatasetBuilder, ExpressionTrainer

    ds = ExpressionDatasetBuilder(output_dir=tempfile.mkdtemp())
    ds.generate_synthetic(samples_per_class=5)
    trainer = ExpressionTrainer(model_type="transformer")
    X, y = trainer.prepare_data(dataset=ds)
    assert X.shape == (45, 1434), X.shape
    metrics = trainer.train(X, y, epochs=2)
    assert metrics["test_accuracy"] >= 0.0

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        trainer.save(path)
        trainer2 = ExpressionTrainer(model_type="transformer")
        trainer2.load(path)
        assert trainer2.classifier.is_trained


def test_torch_forward_shapes():
    import torch
    from src.models.landmark_transformer import (
        LandmarkTransformer,
        TemporalExpressionTransformer,
        OcclusionAwareClassifier,
    )

    torch.manual_seed(0)
    assert LandmarkTransformer(input_dim=1434, num_classes=9)(
        torch.randn(2, 1434)
    ).shape == (2, 9)
    assert TemporalExpressionTransformer(input_dim=10, num_classes=9)(
        torch.randn(2, 15, 10)
    ).shape == (2, 9)
    assert OcclusionAwareClassifier(input_dim=1434, num_classes=9)(
        torch.randn(2, 1434)
    ).shape == (2, 9)


def test_demo_mirror_defaults_on():
    import inspect
    import cv2
    from src.demo.webcam_demo import WebcamDemo

    params = inspect.signature(WebcamDemo.__init__).parameters
    assert params["mirror"].default is True
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[:, :3] = 255
    flipped = cv2.flip(img, 1)
    assert flipped[:, -3:].sum() > flipped[:, :3].sum()


def test_torch_dim_mismatch_is_clear():
    from src.models.expression_recognition import create_classifier

    clf = create_classifier("transformer", input_dim=1434)
    try:
        clf.predict({"mouth_open": 1.0})
    except ValueError as e:
        assert "1434" in str(e), str(e)
    else:
        raise AssertionError("expected ValueError on dim mismatch")


def test_server_routes():
    from fastapi.testclient import TestClient
    from src.demo import server as server_module
    from src.demo.server import app

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/expressions")
    assert r.status_code == 200, r.text
    assert "happy" in r.json()["expressions"]
    r = client.get("/health")
    assert r.status_code == 200, r.text
    # No lifespan run -> pipeline not initialised -> 503, not a crash.
    r = client.post("/predict", json={"image_base64": "e30="})
    assert r.status_code in (503, 500), r.status_code
    # No-face path returns neutral without a camera.
    server_module.face_pipeline = None
    resp = server_module.ExpressionResponse(
        expression="neutral", confidence=0.0, intent="neutral",
        intent_confidence=0.0, occluded=False, features={},
    )
    assert resp.expression == "neutral"


def test_cli_help():
    import subprocess

    proc = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "demo" in proc.stdout and "train" in proc.stdout


if __name__ == "__main__":
    for name, fn in sorted(
        [(k, v) for k, v in globals().items() if k.startswith("test_")]
    ):
        check(name, fn)
    print(f"\n{len(passed)} smoke tests passed.")
