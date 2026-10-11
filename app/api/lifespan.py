import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.prediction_service import PredictionService

logger = logging.getLogger("credilens.lifespan")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager.
    Loads models, preprocessors, thresholds, and baselines once at startup.
    Cleans up references upon application shutdown.
    """
    logger.info("Application lifespan startup: Loading model artifacts...")
    service = PredictionService.get_instance()
    service.ensure_loaded()
    app.state.prediction_service = service
    app.state.model_service = service

    yield

    logger.info("Application lifespan shutdown: Releasing model resources...")
    service.cleanup()
    app.state.prediction_service = None
    app.state.model_service = None
