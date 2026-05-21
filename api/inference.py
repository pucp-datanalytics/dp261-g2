# ============================================================
# Hotel Booking Cancellation API - Inference Module
# ============================================================
# Responsibilities:
# - Load final full pipeline once at startup.
# - Convert user-friendly raw input into the feature format expected
#   by the full pipeline.
# - Generate prediction, probability, risk level and recommendation.
# ============================================================

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PATH = BASE_DIR / "models" / "final_pipeline.pkl"


# ============================================================
# Load artifact once at process startup
# ============================================================

MODEL_PIPELINE = joblib.load(PIPELINE_PATH)


# ============================================================
# Constants
# ============================================================

MONTH_MAP = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

ADR_CAP = 252.0


# ============================================================
# Feature preparation
# ============================================================

def build_pipeline_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert user-friendly raw input into the dataframe expected by
    final_pipeline.pkl.

    The final pipeline already contains:
    - ColumnTransformer
    - numeric preprocessing
    - categorical encoding
    - XGBoost classifier

    Therefore, this function only applies the cleaning and feature
    engineering logic that happened before the sklearn pipeline.
    """

    arrival_date = pd.to_datetime(payload["arrival_date"])
    arrival_month_name = MONTH_MAP[int(arrival_date.month)]

    lead_time = int(payload["lead_time"])
    adr = float(payload["adr"])

    stays_in_weekend_nights = int(payload["stays_in_weekend_nights"])
    stays_in_week_nights = int(payload["stays_in_week_nights"])

    adults = int(payload["adults"])
    children = int(payload.get("children", 0))
    babies = int(payload.get("babies", 0))

    total_guests = adults + children + babies

    if total_guests <= 0:
        raise ValueError(
            "La reserva debe tener al menos un huésped. "
            "La suma adults + children + babies debe ser mayor a 0."
        )

    if children == 10:
        raise ValueError(
            "El valor children=10 fue considerado inválido durante el pipeline de entrenamiento."
        )

    if adr < 0:
        raise ValueError("La variable adr no puede ser negativa.")

    adr = min(adr, ADR_CAP)

    row = {
        # Original variables kept after cleaning
        "hotel": payload["hotel"],
        "arrival_date_year": int(arrival_date.year),
        "arrival_date_month": arrival_month_name,
        "arrival_date_week_number": int(arrival_date.isocalendar().week),
        "arrival_date_day_of_month": int(arrival_date.day),
        "meal": payload["meal"],
        "market_segment": payload["market_segment"],
        "distribution_channel": payload["distribution_channel"],
        "is_repeated_guest": int(payload["is_repeated_guest"]),
        "previous_cancellations": int(payload["previous_cancellations"]),
        "previous_bookings_not_canceled": int(
            payload["previous_bookings_not_canceled"]
        ),
        "reserved_room_type": payload["reserved_room_type"],
        "assigned_room_type": payload["assigned_room_type"],
        "booking_changes": int(payload["booking_changes"]),
        "deposit_type": payload["deposit_type"],
        "days_in_waiting_list": int(payload["days_in_waiting_list"]),
        "customer_type": payload["customer_type"],
        "required_car_parking_spaces": int(
            payload["required_car_parking_spaces"]
        ),
        "total_of_special_requests": int(
            payload["total_of_special_requests"]
        ),

        # Feature engineering
        "total_nights": stays_in_week_nights + stays_in_weekend_nights,
        "total_guests": total_guests,
        "is_high_season": 1 if arrival_month_name in ["July", "August"] else 0,
        "adr_log": float(np.log1p(adr)),
        "lead_time_log": float(np.log1p(lead_time)),
    }

    df_input = pd.DataFrame([row])

    return df_input


# ============================================================
# Business output helpers
# ============================================================

def get_risk_level(probability: float) -> str:
    """Convierte la probabilidad de cancelación en un nivel de riesgo."""
    if probability >= 0.70:
        return "alto"
    if probability >= 0.40:
        return "medio"
    return "bajo"


def get_recommendation(risk_level: str) -> str:
    """Devuelve una recomendación de negocio en español."""
    recommendations = {
        "alto": "Priorizar contacto preventivo con el cliente.",
        "medio": "Monitorear la reserva y considerar una acción preventiva ligera.",
        "bajo": "No se requiere acción urgente.",
    }
    return recommendations[risk_level]


# ============================================================
# Main prediction function
# ============================================================

def predict_from_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function used by the API.

    Flow:
    raw user-friendly input
    -> cleaning + feature engineering
    -> final_pipeline.pkl
    -> prediction response
    """

    df_input = build_pipeline_input(payload)

    prediction = int(MODEL_PIPELINE.predict(df_input)[0])

    if hasattr(MODEL_PIPELINE, "predict_proba"):
        probability = float(MODEL_PIPELINE.predict_proba(df_input)[0, 1])
    else:
        probability = float(prediction)

    risk_level = get_risk_level(probability)

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "recommendation": get_recommendation(risk_level),
    }


def get_model_info() -> Dict[str, Any]:
    """Return model metadata for /version endpoint."""
    return {
        "model_file": PIPELINE_PATH.name,
        "model_type": type(MODEL_PIPELINE).__name__,
        "pipeline_steps": list(MODEL_PIPELINE.named_steps.keys())
        if hasattr(MODEL_PIPELINE, "named_steps")
        else [],
    }