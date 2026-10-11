import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"

FINAL_MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
