import io
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import pandas as pd

from app.api.dependencies import get_prediction_service
from app.services.prediction_service import PredictionService
from app.core.exceptions import InvalidApplicantDataError, PredictionError, ModelNotLoadedError
from app.schemas.credit import (
    ApplicantInput,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    EvaluationResponse,
    ExplainabilityResponse,
)

router = APIRouter()

@router.get("/health")
async def health_check(service: PredictionService = Depends(get_prediction_service)):
    return {
        "status": "healthy",
        "model_loaded": service.is_loaded,
        "model_name": service.metadata.get("model_name", "CrediLens Credit-Default Classifier"),
        "threshold": service.threshold,
    }

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(service: PredictionService = Depends(get_prediction_service)):
    try:
        return service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    applicant: ApplicantInput,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Evaluates a single applicant record across all four models,
    highlighting the calibrated XGBoost decision and local SHAP factor contributions.
    """
    try:
        return service.predict_single(applicant)
    except (InvalidApplicantDataError, ValueError) as ve:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except ModelNotLoadedError as mne:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(mne)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    file: UploadFile = File(...),
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Runs batch predictions using exclusively the final selected calibrated model.
    Accepts CSV upload.
    """
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV format.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        return service.predict_batch(df)
    except (InvalidApplicantDataError, ValueError) as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except ModelNotLoadedError as mne:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(mne)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process batch CSV: {str(e)}")

@router.get("/evaluation", response_model=EvaluationResponse)
async def get_evaluation(service: PredictionService = Depends(get_prediction_service)):
    """Returns four-model comparison table, ROC/PR curves, confusion matrices, and selection rationale."""
    try:
        return service.get_evaluation_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explainability", response_model=ExplainabilityResponse)
async def get_explainability(service: PredictionService = Depends(get_prediction_service)):
    """Returns global TreeSHAP feature importances and methodology details."""
    try:
        return service.get_explainability_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
