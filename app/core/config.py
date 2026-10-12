import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Models directory: prefer models/, fallback to model/
MODEL_DIR = BASE_DIR / "models"
if not MODEL_DIR.exists() and (BASE_DIR / "model").exists():
    MODEL_DIR = BASE_DIR / "model"

DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"

# Model artifact paths
FINAL_MODEL_PATH = MODEL_DIR / "Credit_risk_model.pkl"
BEST_THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"
BASELINES_PATH = MODEL_DIR / "baselines.joblib"

# API Server configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
try:
    API_PORT = int(os.getenv("API_PORT", 8000))
except (ValueError, TypeError):
    API_PORT = 8000
