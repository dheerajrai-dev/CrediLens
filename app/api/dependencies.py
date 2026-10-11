from fastapi import Request
from app.services.prediction_service import PredictionService

def get_prediction_service(request: Request) -> PredictionService:
    """Dependency that retrieves the singleton PredictionService from app.state."""
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        service = PredictionService.get_instance()
        request.app.state.prediction_service = service
    return service
