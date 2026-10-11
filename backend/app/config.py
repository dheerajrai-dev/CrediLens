import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"

FINAL_MODEL_PATH = MODEL_DIR / "Credit_risk_model.pkl"
BEST_THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
