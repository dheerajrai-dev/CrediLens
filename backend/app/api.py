import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from backend.app.schemas import (
    ApplicantInput,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    EvaluationResponse,
    ExplainabilityResponse
)
from backend.app.model_service import ModelService

app = FastAPI(
    title="CrediLens Credit Risk Prediction API",
    description="Explainable Credit Risk Prediction & Decision Support System API",
    version="1.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service on startup
model_service = ModelService.get_instance()

@app.get("/")
async def root():
    return {
        "project": "CrediLens",
        "description": "Explainable Credit Risk Prediction & Decision Support System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded,
        "model_name": model_service.metadata.get("model_name", "CrediLens Credit-Default Classifier"),
        "threshold": model_service.threshold
    }

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    try:
        return model_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(applicant: ApplicantInput):
    """
    Evaluates a single applicant record across all four models,
    highlighting the calibrated XGBoost decision and local SHAP factor contributions.
    """
    try:
        return model_service.predict_single(applicant)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    """
    Runs batch predictions using exclusively the final selected calibrated model.
    Accepts CSV upload.
    """
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV format.")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        return model_service.predict_batch(df)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process batch CSV: {str(e)}")

@app.get("/evaluation", response_model=EvaluationResponse)
async def get_evaluation():
    """Returns four-model comparison table, ROC/PR curves, confusion matrices, and selection rationale."""
    try:
        return model_service.get_evaluation_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explainability", response_model=ExplainabilityResponse)
async def get_explainability():
    """Returns global TreeSHAP feature importances and methodology details."""
    try:
        return model_service.get_explainability_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
