## Modelo Final

El modelo final seleccionado es Tuned XGBoost con las siguientes características:
- Nombre: XGBoost Classifier
- Accuracy: 0.867
- Precision: 0.854
- Recall: 0.775
- F1-score: 0.813
- ROC-AUC: 0.936

## Pipeline de Limpieza y Transformación de Datos

El pipeline de limpieza y transformación de datos es el siguiente:
1. Carga de datos
2. Limpieza de datos
3. Transformación de datos
4. Ingeniería de características
5. Construcción del dataset final

## Dataset Final

El dataset final está disponible en el archivo '../data/interim/hotel_bookings_fe.csv'  

## Lista de Features

| Feature | Descripcion |
|---------|-------------|
| deposit_type_Non Refund | Depósito no reembolsable |
| total_of_special_requests | Número total de solicitudes especiales |
| lead_time_log | Logaritmo natural del tiempo entre la reserva y la llegada |
| required_car_parking_spaces | Número de espacios de estacionamiento requeridos |
| adr_log | Logaritmo natural del precio promedio por día |
| previous_cancellations | Número de cancelaciones previas |
| assigned_room_type_D | Tipo de habitación asignada |
| customer_type_Transient | Tipo de cliente |
| booking_changes | Número de cambios en la reserva |
| reserved_room_type_D | Tipo de habitación reservada |
| arrival_date_week_number | Número de semana de la fecha de llegada |
| customer_type_Transient-Party | Tipo de cliente |
| hotel_Resort | Tipo de hotel |
| market_segment_Online TA | Segmento de mercado |
| arrival_date_year | Año de la fecha de llegada |
| arrival_date_day_of_month | Día del mes de la fecha de llegada |
| total_nights | Número total de noches |
| market_segment_Offline TA/TO | Segmento de mercado |
| previous_bookings_not_canceled | Número de reservas previas no canceladas |
| assigned_room_type_E | Tipo de habitación asignada |

## Model_metadata
{
    "model_path": "../models/tuned_xgb.pkl",
    "model_type": "XGBoost Classifier",
    "training_date": "2026-05-07",
    "target_variable": "is_canceled",
    "features": ["deposit_type_Non Refund", "total_of_special_requests", "lead_time_log", "required_car_parking_spaces", "adr_log", "previous_cancellations", "assigned_room_type_D", "customer_type_Transient", "booking_changes", "reserved_room_type_D", "arrival_date_week_number", "customer_type_Transient-Party", "hotel_Resort", "market_segment_Online TA", "arrival_date_year", "arrival_date_day_of_month", "total_nights", "market_segment_Offline TA/TO", "previous_bookings_not_canceled", "assigned_room_type_E"],
    "model_version": "1.0.0",
    "model_description": "Tuned XGBoost Classifier for hotel booking cancellation prediction",
    "model_performance": {
        "accuracy": 0.867,
        "precision": 0.854,
        "recall": 0.775,
        "f1": 0.813,
        "roc_auc": 0.936
    }
}

## contracts/

input_schema:
  type: object
  properties:
    deposit_type_Non Refund: { type: number }
    total_of_special_requests: { type: number }
    lead_time_log: { type: number }
    required_car_parking_spaces: { type: number }
    adr_log: { type: number }
    previous_cancellations: { type: number }
    assigned_room_type_D: { type: number }
    customer_type_Transient: { type: number }
    booking_changes: { type: number }
    reserved_room_type_D: { type: number }
    arrival_date_week_number: { type: number }
    customer_type_Transient-Party: { type: number }
    hotel_Resort: { type: number }
    market_segment_Online TA: { type: number }
    arrival_date_year: { type: number }
    arrival_date_day_of_month: { type: number }
    total_nights: { type: number }
    market_segment_Offline TA/TO: { type: number }
    previous_bookings_not_canceled: { type: number }
    assigned_room_type_E: { type: number }
  required:
    - deposit_type_Non Refund
    - total_of_special_requests
    - lead_time_log
    - required_car_parking_spaces
    - adr_log
    - previous_cancellations
    - assigned_room_type_D
    - customer_type_Transient
    - booking_changes
    - reserved_room_type_D
    - arrival_date_week_number
    - customer_type_Transient-Party
    - hotel_Resort
    - market_segment_Online TA
    - arrival_date_year
    - arrival_date_day_of_month
    - total_nights
    - market_segment_Offline TA/TO
    - previous_bookings_not_canceled
    - assigned_room_type_E

output_schema:
  type: object
  properties:
    prediction: { type: number }
    probability: { type: number }
  required:
    - prediction
    - probability   

## requirements

- python 3.10
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- dvc
- fastapi
- uvicorn
- requests

## KPIs_targets_SLA

| KPI | Target | SLA |
|-----|--------|-----|
| Accuracy | 0.867 | 0.867 |
| Precision | 0.854 | 0.854 |
| Recall | 0.775 | 0.775 |
| F1 | 0.813 | 0.813 |
| ROC-AUC | 0.936 | 0.936 |
