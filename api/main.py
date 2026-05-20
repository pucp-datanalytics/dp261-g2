# ============================================================
# Hotel Booking Cancellation API - FastAPI App
# ============================================================

from fastapi import FastAPI, HTTPException

from api.inference import get_model_info, predict_from_raw
from api.schemas import (
    HealthResponse,
    PredictionResponse,
    RawPredictionRequest,
    VersionResponse,
)


app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    description=(
        "API for predicting hotel booking cancellations using a full "
        "production pipeline with preprocessing and XGBoost classifier."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    model_info = get_model_info()

    return VersionResponse(
        api_version="1.0.0",
        model_name="Tuned XGBoost Full Pipeline",
        model_version="1.0.0",
        model_file=model_info["model_file"],
        model_type=model_info["model_type"],
        pipeline_steps=model_info["pipeline_steps"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RawPredictionRequest) -> PredictionResponse:
    try:
        result = predict_from_raw(request.model_dump())
        return PredictionResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Required artifact not found: {str(exc)}",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )