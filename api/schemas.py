# ============================================================
# Hotel Booking Cancellation API - Pydantic Schemas
# ============================================================

from datetime import date
from typing import Literal, List

from pydantic import BaseModel, Field


class RawPredictionRequest(BaseModel):
    hotel: Literal["Resort Hotel", "City Hotel"]

    lead_time: int = Field(..., ge=0)
    arrival_date: date

    stays_in_weekend_nights: int = Field(..., ge=0)
    stays_in_week_nights: int = Field(..., ge=0)

    adults: int = Field(..., ge=0)
    children: int = Field(0, ge=0)
    babies: int = Field(0, ge=0)

    meal: Literal["BB", "FB", "HB", "SC", "Undefined"]

    market_segment: Literal[
        "Direct",
        "Corporate",
        "Online TA",
        "Offline TA/TO",
        "Groups",
        "Complementary",
    ]

    distribution_channel: Literal[
        "Direct",
        "TA/TO",
        "Corporate",
        "GDS",
        "Undefined",
    ]

    is_repeated_guest: int = Field(..., ge=0, le=1)
    previous_cancellations: int = Field(..., ge=0)
    previous_bookings_not_canceled: int = Field(..., ge=0)

    reserved_room_type: Literal[
        "A", "B", "C", "D", "E", "F", "G", "H", "L"
    ]

    assigned_room_type: Literal[
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"
    ]

    booking_changes: int = Field(..., ge=0)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]

    days_in_waiting_list: int = Field(..., ge=0)

    customer_type: Literal[
        "Contract",
        "Group",
        "Transient",
        "Transient-Party",
    ]

    adr: float = Field(..., ge=0)
    required_car_parking_spaces: int = Field(..., ge=0)
    total_of_special_requests: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    recommendation: str


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    api_version: str
    model_name: str
    model_version: str
    model_file: str
    model_type: str
    pipeline_steps: List[str]