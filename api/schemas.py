# ============================================================
# Hotel Booking Cancellation API - Pydantic Schemas
# ============================================================

from datetime import date
from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


class RawPredictionRequest(BaseModel):
    hotel: Literal["Resort Hotel", "City Hotel"] = Field(
        ...,
        description="Tipo de hotel. Opciones válidas: Resort Hotel, City Hotel."
    )

    lead_time: int = Field(
        ...,
        ge=0,
        description="Días entre la fecha de reserva y la llegada. Debe ser un entero mayor o igual a 0."
    )

    arrival_date: date = Field(
        ...,
        description="Fecha de llegada en formato YYYY-MM-DD. Ejemplo: 2017-07-15."
    )

    stays_in_weekend_nights: int = Field(
        ...,
        ge=0,
        description="Noches de fin de semana. Debe ser un entero mayor o igual a 0."
    )

    stays_in_week_nights: int = Field(
        ...,
        ge=0,
        description="Noches entre semana. Debe ser un entero mayor o igual a 0."
    )

    adults: int = Field(
        ...,
        ge=0,
        description="Número de adultos. Debe ser un entero mayor o igual a 0."
    )

    children: int = Field(
        0,
        ge=0,
        description="Número de niños. Debe ser un entero mayor o igual a 0. El valor 10 no está permitido."
    )

    babies: int = Field(
        0,
        ge=0,
        description="Número de bebés. Debe ser un entero mayor o igual a 0."
    )

    meal: Literal["BB", "FB", "HB", "SC", "Undefined"] = Field(
        ...,
        description="Tipo de comida. Opciones válidas: BB, FB, HB, SC, Undefined."
    )

    market_segment: Literal[
        "Direct",
        "Corporate",
        "Online TA",
        "Offline TA/TO",
        "Groups",
        "Complementary",
    ] = Field(
        ...,
        description="Segmento de mercado. Opciones válidas: Direct, Corporate, Online TA, Offline TA/TO, Groups, Complementary."
    )

    distribution_channel: Literal[
        "Direct",
        "TA/TO",
        "Corporate",
        "GDS",
        "Undefined",
    ] = Field(
        ...,
        description="Canal de distribución. Opciones válidas: Direct, TA/TO, Corporate, GDS, Undefined."
    )

    is_repeated_guest: int = Field(
        ...,
        ge=0,
        le=1,
        description="Cliente repetido. Valores válidos: 0 = no, 1 = sí."
    )

    previous_cancellations: int = Field(
        ...,
        ge=0,
        description="Cancelaciones previas. Debe ser un entero mayor o igual a 0."
    )

    previous_bookings_not_canceled: int = Field(
        ...,
        ge=0,
        description="Reservas previas no canceladas. Debe ser un entero mayor o igual a 0."
    )

    reserved_room_type: Literal[
        "A", "B", "C", "D", "E", "F", "G", "H", "L"
    ] = Field(
        ...,
        description="Tipo de habitación reservada. Opciones válidas: A, B, C, D, E, F, G, H, L."
    )

    assigned_room_type: Literal[
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L"
    ] = Field(
        ...,
        description="Tipo de habitación asignada. Opciones válidas: A, B, C, D, E, F, G, H, I, K, L."
    )

    booking_changes: int = Field(
        ...,
        ge=0,
        description="Cambios en la reserva. Debe ser un entero mayor o igual a 0."
    )

    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"] = Field(
        ...,
        description="Tipo de depósito. Opciones válidas: No Deposit, Non Refund, Refundable."
    )

    days_in_waiting_list: int = Field(
        ...,
        ge=0,
        description="Días en lista de espera. Debe ser un entero mayor o igual a 0."
    )

    customer_type: Literal[
        "Contract",
        "Group",
        "Transient",
        "Transient-Party",
    ] = Field(
        ...,
        description="Tipo de cliente. Opciones válidas: Contract, Group, Transient, Transient-Party."
    )

    adr: float = Field(
        ...,
        ge=0,
        description="Tarifa diaria promedio. Debe ser un número mayor o igual a 0."
    )

    required_car_parking_spaces: int = Field(
        ...,
        ge=0,
        description="Espacios de estacionamiento requeridos. Debe ser un entero mayor o igual a 0."
    )

    total_of_special_requests: int = Field(
        ...,
        ge=0,
        description="Total de solicitudes especiales. Debe ser un entero mayor o igual a 0."
    )

    @model_validator(mode="after")
    def validate_business_rules(self):
        total_guests = self.adults + self.children + self.babies

        if total_guests <= 0:
            raise ValueError(
                "La reserva debe tener al menos un huésped. "
                "La suma adults + children + babies debe ser mayor a 0."
            )

        if self.children == 10:
            raise ValueError(
                "El valor children=10 fue considerado inválido durante el pipeline de entrenamiento."
            )

        return self


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Predicción del modelo. 0 = no cancelación, 1 = cancelación."
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilidad estimada de cancelación."
    )
    risk_level: str = Field(
        ...,
        description="Nivel de riesgo: low, medium o high."
    )
    recommendation: str = Field(
        ...,
        description="Recomendación de negocio basada en el nivel de riesgo."
    )


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    api_version: str
    model_name: str
    model_version: str
    model_file: str
    model_type: str
    pipeline_steps: List[str]