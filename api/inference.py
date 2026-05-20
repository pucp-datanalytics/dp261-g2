# ============================================================
# Hotel Booking Cancellation API - Inference Module
# ============================================================
# Responsibilities:
# - Load model and preprocessor artifacts once at startup.
# - Convert user-friendly raw input into the 64 expected features.
# - Apply the saved preprocessing pipeline.
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

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


# ============================================================
# Load artifacts once at process startup
# ============================================================

MODEL = joblib.load(MODEL_PATH)

_preprocessor_artifact = joblib.load(PREPROCESSOR_PATH)

if isinstance(_preprocessor_artifact, dict) and "preprocessor" in _preprocessor_artifact:
    PREPROCESSOR = _preprocessor_artifact["preprocessor"]
else:
    PREPROCESSOR = _preprocessor_artifact


# The preprocessor was fitted with 64 input columns.
EXPECTED_COLUMNS = list(PREPROCESSOR.feature_names_in_)


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

ADR_CAP = 252.0  # Same cap used in the data cleaning notebook.


# ============================================================
# Helper functions
# ============================================================

def _to_int(value: Any) -> int:
    """Safely convert a value to int."""
    return int(value)


def _to_float(value: Any) -> float:
    """Safely convert a value to float."""
    return float(value)


def _set_one_hot_feature(
    row: Dict[str, Any],
    prefix: str,
    value: str,
) -> None:
    """
    Set one-hot encoded feature values.

    Example:
    prefix = "market_segment_"
    value = "Online TA"

    If EXPECTED_COLUMNS contains "market_segment_Online TA",
    that column is set to 1. Otherwise all related columns remain 0.
    """
    target_column = f"{prefix}{value}"

    if target_column in row:
        row[target_column] = 1


def build_model_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw user-friendly input into the 64-column dataframe expected
    by the saved preprocessor.

    Input example:
    {
        "hotel": "City Hotel",
        "lead_time": 120,
        "arrival_date": "2017-07-15",
        ...
    }

    Output:
    DataFrame with exactly PREPROCESSOR.feature_names_in_ columns.
    """

    # --------------------------------------------------------
    # Initialize all expected columns with 0
    # --------------------------------------------------------
    row: Dict[str, Any] = {column: 0 for column in EXPECTED_COLUMNS}

    # --------------------------------------------------------
    # Parse arrival date
    # --------------------------------------------------------
    arrival_date = pd.to_datetime(payload["arrival_date"])
    arrival_month_name = MONTH_MAP[int(arrival_date.month)]

    row["arrival_date_year"] = int(arrival_date.year)
    row["arrival_date_week_number"] = int(arrival_date.isocalendar().week)
    row["arrival_date_day_of_month"] = int(arrival_date.day)

    # --------------------------------------------------------
    # Raw numeric variables
    # --------------------------------------------------------
    lead_time = _to_int(payload["lead_time"])
    adr = _to_float(payload["adr"])

    stays_in_week_nights = _to_int(payload["stays_in_week_nights"])
    stays_in_weekend_nights = _to_int(payload["stays_in_weekend_nights"])

    adults = _to_int(payload["adults"])
    children = _to_int(payload.get("children", 0))
    babies = _to_int(payload.get("babies", 0))

    total_guests = adults + children + babies

    if total_guests <= 0:
        raise ValueError("A booking must have at least one guest.")

    if children == 10:
        raise ValueError("children=10 was considered an invalid value in the training pipeline.")

    if adr < 0:
        raise ValueError("adr cannot be negative.")

    # Apply same ADR cap used during training data cleaning.
    adr = min(adr, ADR_CAP)

    row["is_repeated_guest"] = _to_int(payload["is_repeated_guest"])
    row["previous_cancellations"] = _to_int(payload["previous_cancellations"])
    row["previous_bookings_not_canceled"] = _to_int(
        payload["previous_bookings_not_canceled"]
    )
    row["booking_changes"] = _to_int(payload["booking_changes"])
    row["days_in_waiting_list"] = _to_int(payload["days_in_waiting_list"])
    row["required_car_parking_spaces"] = _to_int(
        payload["required_car_parking_spaces"]
    )
    row["total_of_special_requests"] = _to_int(
        payload["total_of_special_requests"]
    )

    # --------------------------------------------------------
    # Feature engineering from notebook 06_feature_eng.ipynb
    # --------------------------------------------------------
    row["total_nights"] = stays_in_week_nights + stays_in_weekend_nights
    row["total_guests"] = total_guests
    row["is_high_season"] = 1 if arrival_month_name in ["July", "August"] else 0
    row["adr_log"] = float(np.log1p(adr))
    row["lead_time_log"] = float(np.log1p(lead_time))

    # --------------------------------------------------------
    # Manual one-hot encoding
    # Important:
    # We do NOT use pd.get_dummies(drop_first=True) here because
    # a single-row request can lose the dummy column. Instead,
    # we directly activate the expected column if it exists.
    # --------------------------------------------------------
    _set_one_hot_feature(row, "hotel_", payload["hotel"])
    _set_one_hot_feature(row, "arrival_date_month_", arrival_month_name)
    _set_one_hot_feature(row, "meal_", payload["meal"])
    _set_one_hot_feature(row, "market_segment_", payload["market_segment"])
    _set_one_hot_feature(row, "distribution_channel_", payload["distribution_channel"])
    _set_one_hot_feature(row, "deposit_type_", payload["deposit_type"])
    _set_one_hot_feature(row, "customer_type_", payload["customer_type"])
    _set_one_hot_feature(row, "reserved_room_type_", payload["reserved_room_type"])
    _set_one_hot_feature(row, "assigned_room_type_", payload["assigned_room_type"])

    # --------------------------------------------------------
    # Build dataframe in the exact expected order
    # --------------------------------------------------------
    df_model = pd.DataFrame([row], columns=EXPECTED_COLUMNS)

    return df_model


def get_risk_level(probability: float) -> str:
    """Convert probability into a simple business risk level."""
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def get_recommendation(risk_level: str) -> str:
    """Return a user-friendly business recommendation."""
    recommendations = {
        "high": "Priorizar contacto preventivo con el cliente.",
        "medium": "Monitorear la reserva y considerar una acción preventiva ligera.",
        "low": "No se requiere acción urgente.",
    }
    return recommendations[risk_level]


def predict_from_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function used by the API.

    Flow:
    raw input
    -> 64 engineered features
    -> saved preprocessor
    -> final trained model
    -> prediction response
    """

    df_model = build_model_input(payload)

    X_processed = PREPROCESSOR.transform(df_model)

    prediction = int(MODEL.predict(X_processed)[0])

    if hasattr(MODEL, "predict_proba"):
        probability = float(MODEL.predict_proba(X_processed)[0, 1])
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
    """Return model and preprocessor metadata for /version endpoint."""
    return {
        "model_file": MODEL_PATH.name,
        "preprocessor_file": PREPROCESSOR_PATH.name,
        "expected_input_features": len(EXPECTED_COLUMNS),
        "model_type": type(MODEL).__name__,
        "preprocessor_type": type(PREPROCESSOR).__name__,
    }