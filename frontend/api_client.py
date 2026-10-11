import logging
import socket
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import requests
import pandas as pd

from backend.app.config import API_HOST, API_PORT
from backend.app.schemas import ApplicantInput
from backend.app.model_service import ModelService

logger = logging.getLogger("credilens.api_client")

class ApiClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or f"http://{API_HOST}:{API_PORT}"
        parsed = urlparse(self.base_url)
        self.host = parsed.hostname or API_HOST
        self.port = parsed.port or API_PORT
        self._direct_service = None
        self._api_reachable_cache: Optional[bool] = None
        self._last_checked_time: float = 0.0

    def _get_direct_service(self) -> ModelService:
        if self._direct_service is None:
            self._direct_service = ModelService.get_instance()
        return self._direct_service

    def is_api_reachable(self) -> bool:
        now = time.time()
        if self._api_reachable_cache is not None and (now - self._last_checked_time) < 2.0:
            return self._api_reachable_cache

        reachable = False
        try:
            # Fast TCP probe to avoid HTTP socket timeout hangs when server is offline
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.2)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            if result == 0:
                resp = requests.get(f"{self.base_url}/health", timeout=1.0)
                reachable = (resp.status_code == 200)
        except Exception:
            reachable = False

        self._api_reachable_cache = reachable
        self._last_checked_time = now
        return reachable

    def get_health(self) -> Dict[str, Any]:
        if self.is_api_reachable():
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API health request failed: {e}")
        
        # In-process fallback
        srv = self._get_direct_service()
        return {
            "status": "healthy (in-process fallback)",
            "model_loaded": srv.is_loaded,
            "model_name": srv.metadata.get("model_name", "CrediLens Credit-Default Classifier"),
            "threshold": srv.threshold
        }

    def get_model_info(self) -> Dict[str, Any]:
        if self.is_api_reachable():
            try:
                resp = requests.get(f"{self.base_url}/model-info", timeout=3.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API model-info request failed: {e}")

        # In-process fallback
        return self._get_direct_service().get_model_info().model_dump()

    def predict(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        applicant_obj = ApplicantInput(**applicant_data)
        if self.is_api_reachable():
            try:
                resp = requests.post(f"{self.base_url}/predict", json=applicant_obj.model_dump(), timeout=5.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API predict request failed: {e}")

        # In-process fallback
        return self._get_direct_service().predict_single(applicant_obj).model_dump()

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.is_api_reachable():
            try:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                files = {"file": ("batch_applicants.csv", csv_bytes, "text/csv")}
                resp = requests.post(f"{self.base_url}/batch-predict", files=files, timeout=30.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API batch-predict request failed: {e}")

        # In-process fallback
        return self._get_direct_service().predict_batch(df).model_dump()

    def get_evaluation(self) -> Dict[str, Any]:
        if self.is_api_reachable():
            try:
                resp = requests.get(f"{self.base_url}/evaluation", timeout=3.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API evaluation request failed: {e}")

        # In-process fallback
        return self._get_direct_service().get_evaluation_data().model_dump()

    def get_explainability(self) -> Dict[str, Any]:
        if self.is_api_reachable():
            try:
                resp = requests.get(f"{self.base_url}/explainability", timeout=3.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.warning(f"API explainability request failed: {e}")

        # In-process fallback
        return self._get_direct_service().get_explainability_data().model_dump()
