from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.lifespan import lifespan
from app.api.routes.prediction import router as prediction_router

app = FastAPI(
    title="CrediLens Credit Risk Prediction API",
    description="Explainable Credit Risk Prediction & Decision Support System API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and cross-origin communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "project": "CrediLens",
        "description": "Explainable Credit Risk Prediction & Decision Support System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health",
    }

# Include routers
app.include_router(prediction_router)
