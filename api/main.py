# ============================================================
# Hotel Booking Cancellation API - FastAPI App
# ============================================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from api.logging_config import log_event
import time

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

log_event(
    event="api_startup",
    status="running"
)

# ============================================================
# Friendly validation error messages
# ============================================================

FIELD_HINTS = {
    "hotel": "Opciones válidas: Resort Hotel, City Hotel.",
    "lead_time": "Debe ser un número entero mayor o igual a 0.",
    "arrival_date": "Debe tener formato YYYY-MM-DD. Ejemplo: 2017-07-15.",
    "stays_in_weekend_nights": "Debe ser un número entero mayor o igual a 0.",
    "stays_in_week_nights": "Debe ser un número entero mayor o igual a 0.",
    "adults": "Debe ser un número entero mayor o igual a 0.",
    "children": "Debe ser un número entero mayor o igual a 0. El valor 10 no está permitido.",
    "babies": "Debe ser un número entero mayor o igual a 0.",
    "meal": "Opciones válidas: BB, FB, HB, SC, Undefined.",
    "market_segment": "Opciones válidas: Direct, Corporate, Online TA, Offline TA/TO, Groups, Complementary.",
    "distribution_channel": "Opciones válidas: Direct, TA/TO, Corporate, GDS, Undefined.",
    "is_repeated_guest": "Valores válidos: 0 = no, 1 = sí.",
    "previous_cancellations": "Debe ser un número entero mayor o igual a 0.",
    "previous_bookings_not_canceled": "Debe ser un número entero mayor o igual a 0.",
    "reserved_room_type": "Opciones válidas: A, B, C, D, E, F, G, H, L.",
    "assigned_room_type": "Opciones válidas: A, B, C, D, E, F, G, H, I, K, L.",
    "booking_changes": "Debe ser un número entero mayor o igual a 0.",
    "deposit_type": "Opciones válidas: No Deposit, Non Refund, Refundable.",
    "days_in_waiting_list": "Debe ser un número entero mayor o igual a 0.",
    "customer_type": "Opciones válidas: Contract, Group, Transient, Transient-Party.",
    "adr": "Debe ser un número mayor o igual a 0.",
    "required_car_parking_spaces": "Debe ser un número entero mayor o igual a 0.",
    "total_of_special_requests": "Debe ser un número entero mayor o igual a 0.",
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Convierte errores de validación de Pydantic en mensajes claros en español.

    HTTP 422 se usa cuando faltan campos, hay tipos incorrectos,
    categorías inválidas o valores fuera de rango.
    """

    friendly_errors = []

    for error in exc.errors():
        loc = error.get("loc", [])
        field_parts = [str(item) for item in loc if item != "body"]
        field = ".".join(field_parts) if field_parts else "request_body"

        error_type = error.get("type", "validation_error")

        if error_type == "missing":
            message = "Campo requerido. Debes enviar este valor en el formulario o JSON."
        elif error_type == "literal_error":
            message = "Valor no permitido para este campo."
        elif "greater_than_equal" in error_type:
            message = "Valor fuera de rango. Debe ser mayor o igual al mínimo permitido."
        elif "less_than_equal" in error_type:
            message = "Valor fuera de rango. Debe ser menor o igual al máximo permitido."
        elif "int" in error_type:
            message = "Tipo de dato inválido. Debes enviar un número entero."
        elif "float" in error_type:
            message = "Tipo de dato inválido. Debes enviar un número."
        elif "date" in error_type:
            message = "Fecha inválida. Debes usar el formato YYYY-MM-DD."
        elif "value_error" in error_type:
            message = "El valor enviado no cumple una regla de negocio."
        else:
            message = "El valor enviado no es válido."

        friendly_errors.append(
            {
                "campo": field,
                "tipo_error": error_type,
                "mensaje": message,
                "ayuda": FIELD_HINTS.get(
                    field,
                    "Revisa que el campo exista, tenga el tipo correcto y respete las opciones o rangos permitidos.",
                ),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "detalle": "Error de validación en los datos de entrada.",
            "errores": friendly_errors,
        },
    )


# ============================================================
# Friendly web form
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """
    Interfaz amigable para usuarios no técnicos.

    Esta pantalla genera internamente el JSON requerido por /predict,
    sin exponer al usuario final la estructura técnica del contrato.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Predicción de Cancelación Hotelera</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                margin: 0;
                padding: 0;
                color: #1f2937;
            }

            .container {
                max-width: 1100px;
                margin: 32px auto;
                padding: 24px;
            }

            .header {
                background: white;
                padding: 28px;
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
                margin-bottom: 24px;
            }

            .header h1 {
                margin: 0 0 8px;
                font-size: 30px;
                color: #1e3a5f;
            }

            .header p {
                margin: 0;
                color: #52606d;
                font-size: 15px;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
            }

            .card {
                background: white;
                border-radius: 18px;
                padding: 24px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }

            .card h2 {
                margin-top: 0;
                font-size: 20px;
                color: #1e3a5f;
            }

            .field {
                display: flex;
                flex-direction: column;
                gap: 6px;
                margin-bottom: 14px;
            }

            label {
                font-weight: 700;
                font-size: 13px;
                color: #334155;
            }

            input, select {
                padding: 10px 12px;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                font-size: 14px;
                background: #ffffff;
            }

            input:focus, select:focus {
                outline: none;
                border-color: #2563eb;
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
            }

            .actions {
                margin-top: 22px;
                display: flex;
                gap: 12px;
                align-items: center;
            }

            button {
                background: #1e3a5f;
                color: white;
                border: none;
                padding: 12px 18px;
                border-radius: 12px;
                cursor: pointer;
                font-size: 15px;
                font-weight: 700;
            }

            button:hover {
                background: #142940;
            }

            .secondary {
                background: #e2e8f0;
                color: #1f2937;
            }

            .secondary:hover {
                background: #cbd5e1;
            }

            .result {
                display: none;
                margin-top: 24px;
                background: white;
                border-radius: 18px;
                padding: 24px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }

            .result.ok {
                border-left: 8px solid #16a34a;
            }

            .result.warn {
                border-left: 8px solid #f59e0b;
            }

            .result.high {
                border-left: 8px solid #dc2626;
            }

            .result.error {
                border-left: 8px solid #dc2626;
                background: #fff7f7;
            }

            .result h2 {
                margin-top: 0;
                color: #1e3a5f;
            }

            .metric {
                font-size: 34px;
                font-weight: 800;
                margin: 8px 0;
            }

            .small {
                color: #64748b;
                font-size: 13px;
            }

            .error-list {
                margin-top: 8px;
                padding-left: 18px;
            }

            .footer {
                margin-top: 20px;
                color: #64748b;
                font-size: 13px;
            }

            .footer a {
                color: #1e3a5f;
                font-weight: 700;
            }

            @media (max-width: 800px) {
                .grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>

    <body>
        <div class="container">
            <div class="header">
                <h1>Predicción de cancelación hotelera</h1>
                <p>
                    Completa los datos de la reserva. La aplicación generará internamente
                    el JSON requerido por la API y mostrará el resultado en lenguaje de negocio.
                </p>
            </div>

            <form id="predictionForm">
                <div class="grid">
                    <div class="card">
                        <h2>Datos principales</h2>

                        <div class="field">
                            <label>Tipo de hotel</label>
                            <select name="hotel" required>
                                <option value="City Hotel">City Hotel</option>
                                <option value="Resort Hotel">Resort Hotel</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Fecha de llegada</label>
                            <input name="arrival_date" type="date" value="2017-07-15" required />
                        </div>

                        <div class="field">
                            <label>Días entre reserva y llegada</label>
                            <input name="lead_time" type="number" min="0" value="120" required />
                        </div>

                        <div class="field">
                            <label>Noches de fin de semana</label>
                            <input name="stays_in_weekend_nights" type="number" min="0" value="1" required />
                        </div>

                        <div class="field">
                            <label>Noches entre semana</label>
                            <input name="stays_in_week_nights" type="number" min="0" value="3" required />
                        </div>

                        <div class="field">
                            <label>ADR / Tarifa diaria promedio</label>
                            <input name="adr" type="number" min="0" step="0.01" value="95.5" required />
                        </div>
                    </div>

                    <div class="card">
                        <h2>Huéspedes y reserva</h2>

                        <div class="field">
                            <label>Adultos</label>
                            <input name="adults" type="number" min="0" value="2" required />
                        </div>

                        <div class="field">
                            <label>Niños</label>
                            <input name="children" type="number" min="0" value="0" required />
                        </div>

                        <div class="field">
                            <label>Bebés</label>
                            <input name="babies" type="number" min="0" value="0" required />
                        </div>

                        <div class="field">
                            <label>Cliente repetido</label>
                            <select name="is_repeated_guest" required>
                                <option value="0">No</option>
                                <option value="1">Sí</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Cancelaciones previas</label>
                            <input name="previous_cancellations" type="number" min="0" value="0" required />
                        </div>

                        <div class="field">
                            <label>Reservas previas no canceladas</label>
                            <input name="previous_bookings_not_canceled" type="number" min="0" value="0" required />
                        </div>
                    </div>

                    <div class="card">
                        <h2>Canal y tipo de cliente</h2>

                        <div class="field">
                            <label>Tipo de comida</label>
                            <select name="meal" required>
                                <option value="BB">BB</option>
                                <option value="FB">FB</option>
                                <option value="HB">HB</option>
                                <option value="SC">SC</option>
                                <option value="Undefined">Undefined</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Segmento de mercado</label>
                            <select name="market_segment" required>
                                <option value="Online TA">Online TA</option>
                                <option value="Offline TA/TO">Offline TA/TO</option>
                                <option value="Direct">Direct</option>
                                <option value="Corporate">Corporate</option>
                                <option value="Groups">Groups</option>
                                <option value="Complementary">Complementary</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Canal de distribución</label>
                            <select name="distribution_channel" required>
                                <option value="TA/TO">TA/TO</option>
                                <option value="Direct">Direct</option>
                                <option value="Corporate">Corporate</option>
                                <option value="GDS">GDS</option>
                                <option value="Undefined">Undefined</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Tipo de cliente</label>
                            <select name="customer_type" required>
                                <option value="Transient">Transient</option>
                                <option value="Transient-Party">Transient-Party</option>
                                <option value="Contract">Contract</option>
                                <option value="Group">Group</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Tipo de depósito</label>
                            <select name="deposit_type" required>
                                <option value="No Deposit">No Deposit</option>
                                <option value="Non Refund">Non Refund</option>
                                <option value="Refundable">Refundable</option>
                            </select>
                        </div>
                    </div>

                    <div class="card">
                        <h2>Habitación y solicitudes</h2>

                        <div class="field">
                            <label>Habitación reservada</label>
                            <select name="reserved_room_type" required>
                                <option value="A">A</option>
                                <option value="B">B</option>
                                <option value="C">C</option>
                                <option value="D">D</option>
                                <option value="E">E</option>
                                <option value="F">F</option>
                                <option value="G">G</option>
                                <option value="H">H</option>
                                <option value="L">L</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Habitación asignada</label>
                            <select name="assigned_room_type" required>
                                <option value="A">A</option>
                                <option value="B">B</option>
                                <option value="C">C</option>
                                <option value="D">D</option>
                                <option value="E">E</option>
                                <option value="F">F</option>
                                <option value="G">G</option>
                                <option value="H">H</option>
                                <option value="I">I</option>
                                <option value="K">K</option>
                                <option value="L">L</option>
                            </select>
                        </div>

                        <div class="field">
                            <label>Cambios en la reserva</label>
                            <input name="booking_changes" type="number" min="0" value="0" required />
                        </div>

                        <div class="field">
                            <label>Días en lista de espera</label>
                            <input name="days_in_waiting_list" type="number" min="0" value="0" required />
                        </div>

                        <div class="field">
                            <label>Espacios de estacionamiento</label>
                            <input name="required_car_parking_spaces" type="number" min="0" value="0" required />
                        </div>

                        <div class="field">
                            <label>Solicitudes especiales</label>
                            <input name="total_of_special_requests" type="number" min="0" value="1" required />
                        </div>
                    </div>
                </div>

                <div class="actions">
                    <button type="submit">Generar predicción</button>
                    <button type="reset" class="secondary">Restablecer</button>
                </div>
            </form>

            <div id="result" class="result"></div>

            <div class="footer">
                Vista amigable para usuarios no técnicos. Para pruebas técnicas del contrato REST, usar
                <a href="/docs" target="_blank">/docs</a>.
            </div>
        </div>

        <script>
            const form = document.getElementById("predictionForm");
            const resultDiv = document.getElementById("result");

            const numericFields = [
                "lead_time",
                "stays_in_weekend_nights",
                "stays_in_week_nights",
                "adults",
                "children",
                "babies",
                "is_repeated_guest",
                "previous_cancellations",
                "previous_bookings_not_canceled",
                "booking_changes",
                "days_in_waiting_list",
                "required_car_parking_spaces",
                "total_of_special_requests"
            ];

            const floatFields = ["adr"];

            function buildPayload(formData) {
                const payload = {};

                for (const [key, value] of formData.entries()) {
                    if (numericFields.includes(key)) {
                        payload[key] = parseInt(value, 10);
                    } else if (floatFields.includes(key)) {
                        payload[key] = parseFloat(value);
                    } else {
                        payload[key] = value;
                    }
                }

                return payload;
            }

            function renderSuccess(data) {
                const probability = (data.probability * 100).toFixed(2);
                const risk = data.risk_level;

                let cssClass = "ok";
                if (risk === "medio") cssClass = "warn";
                if (risk === "alto") cssClass = "high";

                const predictionText = data.prediction === 1
                    ? "La reserva podría cancelarse"
                    : "La reserva probablemente no se cancelará";

                resultDiv.className = "result " + cssClass;
                resultDiv.style.display = "block";
                resultDiv.innerHTML = `
                    <h2>Resultado de la predicción</h2>
                    <div class="small">Probabilidad estimada de cancelación</div>
                    <div class="metric">${probability}%</div>
                    <p><strong>Predicción:</strong> ${predictionText}</p>
                    <p><strong>Nivel de riesgo:</strong> ${risk}</p>
                    <p><strong>Recomendación:</strong> ${data.recommendation}</p>
                `;
            }

            function renderError(errorData) {
                resultDiv.className = "result error";
                resultDiv.style.display = "block";

                let html = "<h2>No se pudo generar la predicción</h2>";

                if (errorData.errores) {
                    html += "<p>Revisa los siguientes campos:</p><ul class='error-list'>";
                    errorData.errores.forEach(err => {
                        html += `<li><strong>${err.campo}:</strong> ${err.mensaje}<br><span class="small">${err.ayuda}</span></li>`;
                    });
                    html += "</ul>";
                } else if (errorData.detail && typeof errorData.detail === "object") {
                    html += `<p>${errorData.detail.mensaje || "Ocurrió un error."}</p>`;
                    if (errorData.detail.ayuda) {
                        html += `<p class="small">${errorData.detail.ayuda}</p>`;
                    }
                } else {
                    html += "<p>Ocurrió un error inesperado. Revisa los datos ingresados.</p>";
                }

                resultDiv.innerHTML = html;
            }

            form.addEventListener("submit", async (event) => {
                event.preventDefault();

                resultDiv.className = "result";
                resultDiv.style.display = "block";
                resultDiv.innerHTML = "<h2>Procesando predicción...</h2>";

                const formData = new FormData(form);
                const payload = buildPayload(formData);

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        renderError(data);
                        return;
                    }

                    renderSuccess(data);
                } catch (error) {
                    renderError({
                        detail: {
                            mensaje: "No se pudo conectar con la API.",
                            ayuda: "Verifica que el servicio esté activo e intenta nuevamente."
                        }
                    });
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ============================================================
# API endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:

    log_event(
        event="healthcheck",
        status_code=200
    )

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


import time

@app.post("/predict", response_model=PredictionResponse)
def predict(request: RawPredictionRequest) -> PredictionResponse:

    t0 = time.time()

    try:

        payload = request.model_dump()

        result = predict_from_raw(payload)

        latency_ms = (time.time() - t0) * 1000

        # Log estructurado
        log_event(
            event="prediction_success",
            status_code=200,
            latency_ms=round(latency_ms, 2),

            # drift monitoring
            lead_time=payload.get("lead_time"),
            adr=payload.get("adr"),
            adults=payload.get("adults"),
            children=payload.get("children"),

            prediction=result.get("prediction"),
            probability=result.get("probability")
        )

        return PredictionResponse(**result)

    except ValueError as exc:

        latency_ms = (time.time() - t0) * 1000

        log_event(
            event="validation_error",
            status_code=400,
            latency_ms=round(latency_ms, 2),
            error=str(exc)
        )

        raise HTTPException(
            status_code=400,
            detail={
                "mensaje": str(exc),
                "ayuda": "Revisa las reglas de negocio de la reserva enviada.",
            },
        )

    except FileNotFoundError as exc:

        latency_ms = (time.time() - t0) * 1000

        log_event(
            event="model_file_missing",
            status_code=500,
            latency_ms=round(latency_ms, 2),
            error=str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "mensaje": f"No se encontró un artefacto requerido: {str(exc)}",
                "ayuda": "Verifica que models/final_pipeline.pkl exista en la ruta esperada.",
            },
        )

    except Exception as exc:

        latency_ms = (time.time() - t0) * 1000

        log_event(
            event="prediction_failure",
            status_code=500,
            latency_ms=round(latency_ms, 2),
            error=str(exc)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "mensaje": f"Falló la predicción: {str(exc)}",
                "ayuda": "Revisa el formato de entrada, el pipeline del modelo o los logs del servidor.",
            },
        )