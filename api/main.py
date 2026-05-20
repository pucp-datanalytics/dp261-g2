# ============================================================
# Hotel Booking Cancellation API - FastAPI App
# ============================================================
# Endpoints:
# - GET  /health
# - GET  /version
# - POST /predict
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
        "API for predicting hotel booking cancellations using a final "
        "Tuned XGBoost model and a saved preprocessing pipeline."
    ),
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check endpoint.

    Used by local tests, Docker HEALTHCHECK, ECS or load balancers
    to verify that the API is alive.
    """
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """
    Return API, model and preprocessor metadata.
    """
    model_info = get_model_info()

    return VersionResponse(
        api_version="1.0.0",
        model_name="Tuned XGBoost",
        model_version="1.0.0",
        model_file=model_info["model_file"],
        preprocessor_file=model_info["preprocessor_file"],
        expected_input_features=model_info["expected_input_features"],
        model_type=model_info["model_type"],
        preprocessor_type=model_info["preprocessor_type"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RawPredictionRequest) -> PredictionResponse:
    """
    Predict whether a hotel booking will be canceled.

    The endpoint receives user-friendly raw booking data.
    Internally, the API:
    - creates engineered features,
    - reconstructs the 64 expected input columns,
    - applies the saved preprocessor,
    - calls the final trained model.
    """
    try:
        payload = request.model_dump()
        result = predict_from_raw(payload)

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