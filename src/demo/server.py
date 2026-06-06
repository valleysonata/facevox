import os
import time
import base64
import numpy as np
import cv2
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from src.models.face_landmarks import FaceLandmarkPipeline, normalize_landmarks
from src.models.expression_recognition import (
    ExpressionClassifier,
    AssistiveExpressionMapper,
    ExpressionLabel,
    create_classifier,
    create_mapper,
)
from src.models.occlusion import RobustOcclusionHandler

app = FastAPI(
    title="FaceVox API",
    description="Real-time facial expression recognition for assistive communication",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_pipeline: Optional[FaceLandmarkPipeline] = None
classifier: Optional[ExpressionClassifier] = None
intent_mapper: Optional[AssistiveExpressionMapper] = None
occlusion_handler: Optional[RobustOcclusionHandler] = None

class ExpressionResponse(BaseModel):
    expression: str
    confidence: float
    intent: str
    intent_confidence: float
    occluded: bool
    features: dict

class BatchRequest(BaseModel):
    frames: list  # base64 encoded images

class BatchResponse(BaseModel):
    results: list

@app.on_event("startup")
async def startup():
    global face_pipeline, classifier, intent_mapper, occlusion_handler

    face_pipeline = FaceLandmarkPipeline(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
    )

    classifier = create_classifier(model_type="rf")
    model_path = "checkpoints/expression_model.joblib"
    if os.path.exists(model_path):
        classifier.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found. Using default classifier.")

    intent_mapper = create_mapper()
    occlusion_handler = RobustOcclusionHandler()
    print("Server started successfully")

@app.on_event("shutdown")
async def shutdown():
    if face_pipeline:
        face_pipeline.close()

@app.get("/")
async def root():
    return {"message": "FaceVox API", "docs": "/docs"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": classifier.is_trained if classifier else False,
    }

@app.post("/predict", response_model=ExpressionResponse)
async def predict(base64_image: str):
    """Predict expression from base64 encoded image."""
    if not face_pipeline or not classifier:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        result = _process_frame(frame)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict expressions for multiple frames."""
    if not face_pipeline or not classifier:
        raise HTTPException(status_code=503, detail="Server not initialized")

    results = []
    for base64_image in request.frames:
        try:
            img_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                result = _process_frame(frame)
                results.append(result.model_dump())
            else:
                results.append({"error": "Invalid image"})
        except Exception as e:
            results.append({"error": str(e)})

    return BatchResponse(results=results)

@app.get("/expressions")
async def list_expressions():
    """List all supported expressions."""
    return {
        "expressions": [label.value for label in ExpressionLabel],
        "intents": {
            "happy": "yes",
            "sad": "no",
            "surprised": "help",
            "angry": "pain",
            "disgusted": "no",
            "fearful": "help",
        },
    }

@app.get("/stats")
async def stats():
    """Get server statistics."""
    return {
        "model_type": "RandomForest",
        "num_classes": 9,
        "features": 10,
        "temporal_window": 30,
    }

def _process_frame(frame: np.ndarray) -> ExpressionResponse:
    """Process a single frame."""
    face_result = face_pipeline.process_and_extract(frame)
    if face_result is None:
        return ExpressionResponse(
            expression="neutral",
            confidence=0.0,
            intent="neutral",
            intent_confidence=0.0,
            occluded=False,
            features={},
        )

    landmarks_obj = face_result['landmarks']
    features = face_result['features']

    adapted_landmarks, occlusion = occlusion_handler.process(
        landmarks_obj.landmarks,
        use_mask_adaptation=True,
    )

    expression = classifier.predict(features, raw_landmarks=normalize_landmarks(landmarks_obj.landmarks).flatten().tolist())

    intent = intent_mapper.map_to_intent(expression)

    return ExpressionResponse(
        expression=expression.label.value,
        confidence=expression.confidence,
        intent=intent.label.value,
        intent_confidence=intent.confidence,
        occluded=occlusion.is_occluded,
        features=features,
    )

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
