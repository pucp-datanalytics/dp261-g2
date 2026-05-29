# =========================================================
# HOTEL CANCELLATION INTELLIGENCE PLATFORM
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import joblib
import shap

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score
)

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Hotel Cancellation Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: #F8FAFC;
}

h1, h2, h3, h4 {
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #1E1E2F;
}

/* Títulos sidebar */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #F8FAFC !important;
}

.kpi-card {
    background-color: #1C2333;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #2A3447;
}

.kpi-title {
    font-size: 15px;
    color: #9CA3AF;
}

.kpi-value {
    font-size: 34px;
    font-weight: bold;
    color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
}
            
.custom-card-red {
    background-color: rgba(239,85,59,0.18);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(239,85,59,0.40);
}
.custom-card-yellow {
    background-color: rgba(254,203,82,0.18);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(254,203,82,0.40);
}
.custom-card-green {
    background-color: rgba(0,204,150,0.18);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(0,204,150,0.40);
}
.card-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 15px;
    color: #F8FAFC;
}
.card-text {
    font-size: 16px;
    margin-bottom: 10px;
    color: #F8FAFC;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>

    .custom-card-red {
        background-color: rgba(239,85,59,0.18);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(239,85,59,0.40);
    }

    .custom-card-yellow {
        background-color: rgba(254,203,82,0.18);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(254,203,82,0.40);
    }

    .custom-card-green {
        background-color: rgba(0,204,150,0.18);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(0,204,150,0.40);
    }

    .card-title {
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #F8FAFC;
    }

    .card-text {
        font-size: 16px;
        margin-bottom: 10px;
        color: #F8FAFC;
    }

    </style>
    """, unsafe_allow_html=True)

# =========================================================
# MODELS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

XGB_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost.pkl"
)

RF_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest.pkl"
)

xgb_model = joblib.load(XGB_MODEL_PATH)

rf_model = joblib.load(RF_MODEL_PATH)

# =========================================================
# CLEANING
# =========================================================

def clean_data(df):

    df = df.copy()

    cols_drop = [
        "reservation_status",
        "reservation_status_date",
        "agent",
        "company",
        "country"
    ]

    df.drop(
        columns=cols_drop,
        inplace=True,
        errors="ignore"
    )

    if "children" in df.columns:

        df["children"] = (
            df["children"]
            .fillna(0)
            .astype(int)
        )

    if "adr" in df.columns:

        df = df[df["adr"] >= 0]

        p99 = df["adr"].quantile(0.99)

        df["adr"] = df["adr"].clip(
            upper=p99
        )

    return df

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def feature_engineering(df):

    df = df.copy()

    df["total_nights"] = (
        df["stays_in_week_nights"]
        +
        df["stays_in_weekend_nights"]
    )

    df["total_guests"] = (
        df["adults"]
        +
        df["children"]
        +
        df["babies"]
    )

    high_season = [
        "July",
        "August"
    ]

    df["is_high_season"] = (
        df["arrival_date_month"]
        .isin(high_season)
        .astype(int)
    )

    df["adr_log"] = np.log1p(df["adr"])

    df["lead_time_log"] = np.log1p(
        df["lead_time"]
    )

    return df

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ Panel de Control"
)

threshold = st.sidebar.slider(
    "Threshold de decisión",
    0.0,
    1.0,
    0.35
)

# =========================================================
# BUSINESS PARAMETERS
# =========================================================

st.sidebar.subheader(
    "💰 Business Parameters"
)

ADR_PROMEDIO = st.sidebar.slider(
    "ADR promedio",
    50,
    500,
    102
)

EFECTIVIDAD_OPERATIVA = st.sidebar.slider(
    "Efectividad operativa",
    0.0,
    1.0,
    0.25
)

COSTO_FP = st.sidebar.slider(
    "Costo por falso positivo",
    0.0,
    50.0,
    5.0
)

BENEFICIO_TP = (
    ADR_PROMEDIO
    * EFECTIVIDAD_OPERATIVA
)

PERDIDA_FN = ADR_PROMEDIO

# =========================================================
# HEADER
# =========================================================

st.markdown("""
# 🏨 Dashboard de cancelaciones hoteleras

Sistema inteligente de predicción de cancelaciones hoteleras
para revenue management y toma de decisiones estratégicas.
""")

st.divider()

# =========================================================
# MODE SELECTION
# =========================================================

modo = st.radio(
    "Seleccione modo de uso",
    [
        "📂 Predicción Masiva",
        "🧾 Simulación Manual"
    ],
    horizontal=True
)

# =========================================================
# MASSIVE PREDICTION
# =========================================================

if modo == "📂 Predicción Masiva":

    uploaded = st.file_uploader(
        "📂 Suba un archivo",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded is None:

        st.warning(
            "Suba un dataset para comenzar."
        )

        st.stop()

    if uploaded.name.endswith(".csv"):

        df_eval = pd.read_csv(uploaded)
        df_original = df_eval.copy()

    else:

        df_eval = pd.read_excel(uploaded)
        df_original = df_eval.copy()

    st.success(
        "✅ Dataset cargado correctamente"
    )

    with st.expander(
        "📄 Vista previa del dataset"
    ):

        st.dataframe(
            df_eval.head(),
            width="stretch"
        )

# =========================================================
# MANUAL SIMULATION
# =========================================================

else:

    st.subheader(
        "🧾 Simulación manual de reserva"
    )

    with st.form("manual_form"):

        col1, col2, col3 = st.columns(3)

        # =====================================================
        # RESERVA
        # =====================================================

        with col1:

            st.markdown("## 🏨 Reserva")

            hotel = st.selectbox(
                "🏨 Hotel",
                [
                    "Resort Hotel",
                    "City Hotel"
                ]
            )

            lead_time = st.slider(
                "⏳ Lead Time - días entre reserva y llegada",
                0,
                700,
                120
            )

            adr = st.number_input(
                "💰 ADR - Tarifa diaria promedio",
                0.0,
                1000.0,
                120.0
            )

            month = st.selectbox(
                "📅 Mes llegada",
                [
                    "January","February","March",
                    "April","May","June",
                    "July","August","September",
                    "October","November","December"
                ]
            )

            arrival_week = st.slider(
                "🗓️ Semana llegada",
                1,
                53,
                30
            )

            arrival_day = st.slider(
                "📍 Día llegada",
                1,
                31,
                15
            )

            week_nights = st.slider(
                "🌙 Noches entre semana",
                0,
                20,
                2
            )

            weekend_nights = st.slider(
                "🎉 Noches fin semana",
                0,
                10,
                1
            )

            reserved_room_type = st.selectbox(
                "🛏️ Habitación reservada",
                ["A","B","C","D","E","F","G"]
            )

            assigned_room_type = st.selectbox(
                "🔑 Habitación asignada",
                ["A","B","C","D","E","F","G"]
            )

        # =====================================================
        # CLIENTE
        # =====================================================

        with col2:

            st.markdown("## 👤 Cliente")

            adults = st.number_input(
                "🧑 Número de adultos",
                min_value=1,
                max_value=20,
                value=2,
                step=1
            )

            children = st.number_input(
                "🧒 Número de niños",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

            babies = st.number_input(
                "👶 Número de bebés",
                min_value=0,
                max_value=20,
                value=0,
                step=1
            )

            is_repeated_guest = st.selectbox(
                "🔁 Cliente recurrente",
                [0,1]
            )

            previous_cancellations = st.slider(
                "❌ Cancelaciones previas",
                0,
                10,
                0
            )

            previous_bookings_not_canceled = st.slider(
                "✅ Reservas previas no canceladas",
                0,
                50,
                0
            )

            customer_type = st.selectbox(
                "🪪 Tipo cliente",
                [
                    "Transient",
                    "Contract",
                    "Transient-Party",
                    "Group"
                ]
            )

            parking_spaces = st.slider(
                "🚗 Parking requerido",
                0,
                5,
                0
            )

        # =====================================================
        # COMERCIAL
        # =====================================================

        with col3:

            st.markdown("## 💼 Comercial")

            market_segment = st.selectbox(
                "📊 Segmento",
                [
                    "Online TA",
                    "Offline TA/TO",
                    "Direct",
                    "Groups",
                    "Corporate"
                ]
            )

            distribution_channel = st.selectbox(
                "🌐 Canal distribución",
                [
                    "TA/TO",
                    "Direct",
                    "Corporate",
                    "GDS"
                ]
            )

            deposit_type = st.selectbox(
                "💳 Depósito",
                [
                    "No Deposit",
                    "Refundable",
                    "Non Refund"
                ]
            )

            special_requests = st.slider(
                "⭐ Solicitudes especiales",
                0,
                5,
                1
            )

            booking_changes = st.slider(
                "🔄 Cambios de reserva",
                0,
                10,
                0
            )

            waiting_list = st.slider(
                "⏱️ Waiting list",
                0,
                400,
                0
            )

        submitted = st.form_submit_button(
            "🔍 Predecir cancelación"
        )

    if not submitted:

        st.stop()

    # =====================================================
    # DATAFRAME MANUAL
    # =====================================================

    df_eval = pd.DataFrame([{

        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": 2017,
        "arrival_date_month": month,
        "arrival_date_week_number": arrival_week,
        "arrival_date_day_of_month": arrival_day,

        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,

        "adults": adults,
        "children": children,
        "babies": babies,

        # EL MODELO NECESITA ESTA VARIABLE
        "meal": "BB",

        "market_segment": market_segment,
        "distribution_channel": distribution_channel,

        "is_repeated_guest": is_repeated_guest,

        "previous_cancellations":
            previous_cancellations,

        "previous_bookings_not_canceled":
            previous_bookings_not_canceled,

        "reserved_room_type":
            reserved_room_type,

        "assigned_room_type":
            assigned_room_type,

        "booking_changes":
            booking_changes,

        "deposit_type":
            deposit_type,

        "days_in_waiting_list":
            waiting_list,

        "customer_type":
            customer_type,

        "adr": adr,

        "required_car_parking_spaces":
            parking_spaces,

        "total_of_special_requests":
            special_requests

    }])

# =========================================================
# PROCESSING
# =========================================================

df_eval = clean_data(df_eval)

df_eval = feature_engineering(df_eval)

# =========================================================
# X / Y
# =========================================================

if "is_canceled" in df_eval.columns:

    X_eval = df_eval.drop(
        "is_canceled",
        axis=1
    )

    y_eval = df_eval["is_canceled"]

else:

    X_eval = df_eval.copy()

    y_eval = None

# =========================================================
# PREDICTIONS
# =========================================================

import requests
import os

API_URL = os.getenv(
    "API_URL",
    "http://34.204.98.17:8000"
)

def api_disponible():

    try:

        r = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        return r.status_code == 200

    except Exception:

        return False

# =========================================================
# XGBOOST
# =========================================================

try:

    y_proba_xgb = xgb_model.predict_proba(
        X_eval
    )[:,1]

except Exception as e:

    st.error(f"Error XGBoost: {e}")
    st.stop()

y_pred_xgb = (
    y_proba_xgb >= threshold
).astype(int)

# =========================================================
# RANDOM FOREST
# =========================================================

try:

    y_proba_rf = rf_model.predict_proba(
        X_eval
    )[:,1]

except Exception as e:

    st.error(f"Error Random Forest: {e}")
    st.stop()

y_pred_rf = (
    y_proba_rf >= threshold
).astype(int)

# =========================================================
# RESULTS XGB
# =========================================================

results_xgb = X_eval.copy()

results_xgb["Probabilidad"] = np.round(
    y_proba_xgb,
    3
)

results_xgb["Predicción"] = np.where(
    y_pred_xgb == 1,
    "❌ Cancelará",
    "✅ No cancelará"
)

# =========================================================
# RESULTS RF
# =========================================================

results_rf = X_eval.copy()

results_rf["Probabilidad"] = np.round(
    y_proba_rf,
    3
)

results_rf["Predicción"] = np.where(
    y_pred_rf == 1,
    "❌ Cancelará",
    "✅ No cancelará"
)

# =========================================================
# RISK LEVEL
# =========================================================

def riesgo(p):

    if p > 0.7:
        return "🔴 Riesgo Alto"

    elif p > 0.4:
        return "🟡 Riesgo Medio"

    else:
        return "🟢 Riesgo Bajo"

results_xgb["Nivel de Riesgo"] = results_xgb[
    "Probabilidad"
].apply(riesgo)

results_rf["Nivel de Riesgo"] = results_rf[
    "Probabilidad"
].apply(riesgo)

# =========================================================
# ACTIONS
# =========================================================

def accion_recomendada(r):

    if r == "🔴 Riesgo Alto":

        return (
            "🚨 Solicitar depósito | "
            "📞 Reconfirmar reserva | "
            "👀 Monitoreo prioritario"
        )

    elif r == "🟡 Riesgo Medio":

        return (
            "📧 Enviar recordatorio | "
            "📊 Seguimiento preventivo"
        )

    else:

        return (
            "✅ Flujo normal de atención"
        )

results_xgb["Acción Recomendada"] = results_xgb[
    "Nivel de Riesgo"
].apply(accion_recomendada)

results_rf["Acción Recomendada"] = results_rf[
    "Nivel de Riesgo"
].apply(accion_recomendada)

# =========================================================
# BUSINESS VALUE
# =========================================================

from sklearn.metrics import confusion_matrix


def calcular_business_value(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    gain = BENEFICIO_TP

    business_value = (
        gain * tp
        -
        COSTO_FP * (tp + fp)
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Gain": gain,
        "Business Value": business_value
    }

# =========================================================
# COLORS
# =========================================================

risk_colors = {
    "🟢 Riesgo Bajo": "#00CC96",
    "🟡 Riesgo Medio": "#FECB52",
    "🔴 Riesgo Alto": "#EF553B"
}

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Resumen Ejecutivo",
    "🤖 Performance ML",
    "💰 Business Value",
    "📋 Predicciones",
    "🔍 Explainability"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    total = len(results_xgb)

    cancelaciones = (
        results_xgb["Predicción"]
        ==
        "❌ Cancelará"
    ).sum()

    riesgo_prom = results_xgb[
        "Probabilidad"
    ].mean()

    pct_alto = (
        results_xgb["Nivel de Riesgo"]
        ==
        "🔴 Riesgo Alto"
    ).mean() * 100

    # =========================================================
    # KPIs
    # =========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🏨 Reservas",
        total
    )

    col2.metric(
        "❌ Cancelaciones",
        cancelaciones
    )

    col3.metric(
        "📊 Riesgo Promedio",
        f"{riesgo_prom:.1%}"
    )

    col4.metric(
        "🚨 % Riesgo Alto",
        f"{pct_alto:.1f}%"
    )

    st.divider()

    # =========================================================
    # RECOMENDACIONES OPERATIVAS
    # =========================================================

    st.subheader(
        "🎯 Recomendaciones Operativas"
    )

    alto = (
        results_xgb["Nivel de Riesgo"]
        == "🔴 Riesgo Alto"
    ).sum()

    medio = (
        results_xgb["Nivel de Riesgo"]
        == "🟡 Riesgo Medio"
    ).sum()

    bajo = (
        results_xgb["Nivel de Riesgo"]
        == "🟢 Riesgo Bajo"
    ).sum()

    
    c1, c2, c3 = st.columns(3)

    with c1:
        st.error(f"🔴 Riesgo Alto\n\nReservas: {alto}\n\nAcción: Solicitar depósito y reconfirmar reserva.")

    with c2:
        st.warning(f"🟡 Riesgo Medio\n\nReservas: {medio}\n\nAcción: Seguimiento preventivo.")

    with c3:
        st.success(f"🟢 Riesgo Bajo\n\nReservas: {bajo}\n\nAcción: Flujo normal.")

    st.divider()

    
    # =========================================================
    # GRÁFICOS
    # =========================================================

    col1, col2 = st.columns(2)

    with col1:

        fig_hist = px.histogram(
            results_xgb,
            x="Probabilidad",
            color="Nivel de Riesgo",
            template="plotly_dark",
            nbins=30,
            color_discrete_map=risk_colors
        )

        fig_hist.update_layout(
            title="📈 Distribución de Probabilidades",
            title_x=0.25,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="#FFFFFF", size=14),
            title_font=dict(color="#FFFFFF", size=20),

            xaxis=dict(
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF")
            ),

            yaxis=dict(
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF")
            ),

            legend=dict(
                font=dict(color="#FFFFFF")
            )
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    with col2:

        riesgo_counts = (
            results_xgb["Nivel de Riesgo"]
            .value_counts()
            .reset_index()
        )

        riesgo_counts.columns = ["Nivel de Riesgo", "Cantidad"]

        fig_pie = px.pie(
            riesgo_counts,
            names="Nivel de Riesgo",
            values="Cantidad",
            hole=0.5,
            template="plotly_dark",
            color="Nivel de Riesgo",
            color_discrete_map=risk_colors
        )

        fig_pie.update_layout(
            title="🥧 Distribución de Riesgo",
            title_x=0.25,
            paper_bgcolor="#0E1117",
            font=dict(color="#FFFFFF", size=14),
            title_font=dict(color="#FFFFFF", size=20),
            legend=dict(font=dict(color="#FFFFFF"))
        )

        fig_pie.update_traces(
            textfont=dict(color="#0008ff", size=14)
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )
# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader(
        "🤖 Comparación de Modelos"
    )

    if y_eval is not None:

        # =====================================================
        # METRICS
        # =====================================================

        metrics_df = pd.DataFrame({

            "Métrica": [
                "Accuracy",
                "Recall",
                "F1 Score",
                "AUC ROC"
            ],

            "XGBoost": [

                accuracy_score(
                    y_eval,
                    y_pred_xgb
                ),

                recall_score(
                    y_eval,
                    y_pred_xgb
                ),

                f1_score(
                    y_eval,
                    y_pred_xgb
                ),

                roc_auc_score(
                    y_eval,
                    y_proba_xgb
                )
            ],

            "Random Forest": [

                accuracy_score(
                    y_eval,
                    y_pred_rf
                ),

                recall_score(
                    y_eval,
                    y_pred_rf
                ),

                f1_score(
                    y_eval,
                    y_pred_rf
                ),

                roc_auc_score(
                    y_eval,
                    y_proba_rf
                )
            ]
        })

        st.dataframe(
            metrics_df.style.format({
                "XGBoost":"{:.2%}",
                "Random Forest":"{:.2%}"
            }),
            width="stretch"
        )

        st.divider()

        # =====================================================
        # ROC CURVES
        # =====================================================

        fpr_xgb, tpr_xgb, _ = roc_curve(
            y_eval,
            y_proba_xgb
        )

        fpr_rf, tpr_rf, _ = roc_curve(
            y_eval,
            y_proba_rf
        )

        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=fpr_xgb,
                y=tpr_xgb,
                mode="lines",
                name="XGBoost"
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=fpr_rf,
                y=tpr_rf,
                mode="lines",
                name="Random Forest"
            )
        )

        fig_roc.add_trace(
            go.Scatter(
                x=[0,1],
                y=[0,1],
                mode="lines",
                line=dict(
                    dash="dash"
                ),
                showlegend=False
            )
        )

        fig_roc.update_layout(
            template="plotly_dark",
            title="📈 ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        st.plotly_chart(
            fig_roc,
            width="stretch"
        )

        st.divider()

        # =====================================================
        # CONFUSION MATRICES
        # =====================================================

        st.subheader(
            "🧩 Matrices de Confusión"
        )

        cm_xgb = confusion_matrix(
            y_eval,
            y_pred_xgb
        )

        cm_rf = confusion_matrix(
            y_eval,
            y_pred_rf
        )

        col1, col2 = st.columns(2)

        # =========================================
        # XGBOOST
        # =========================================

        with col1:

            fig_cm_xgb = px.imshow(
                cm_xgb,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(
                    x="Predicción",
                    y="Valor Real",
                    color="Cantidad"
                ),
                x=[
                    "No Cancel",
                    "Cancel"
                ],
                y=[
                    "No Cancel",
                    "Cancel"
                ],
                title="🎯 XGBoost"
            )

            fig_cm_xgb.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(
                    color="#FFFFFF",
                    size=14
                ),
                title_font=dict(
                    color="#FFFFFF",
                    size=20
                )
            )

            st.plotly_chart(
                fig_cm_xgb,
                width="stretch"
            )

        # =========================================
        # RANDOM FOREST
        # =========================================

        with col2:

            fig_cm_rf = px.imshow(
                cm_rf,
                text_auto=True,
                color_continuous_scale="Greens",
                labels=dict(
                    x="Predicción",
                    y="Valor Real",
                    color="Cantidad"
                ),
                x=[
                    "No Cancel",
                    "Cancel"
                ],
                y=[
                    "No Cancel",
                    "Cancel"
                ],
                title="🌲 Random Forest"
            )

            fig_cm_rf.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(
                    color="#FFFFFF",
                    size=14
                ),
                title_font=dict(
                    color="#FFFFFF",
                    size=20
                )
            )

            st.plotly_chart(
                fig_cm_rf,
                width="stretch"
            )

    else:

        st.info(
            "📂 Suba dataset con is_canceled."
        )

# =========================================================
# TAB 3 - BUSINESS VALUE
# =========================================================

with tab3:

    st.subheader(
        "💰 Business Value"
    )

    if y_eval is not None:

        bv_xgb = calcular_business_value(
            y_eval,
            y_pred_xgb
        )

        bv_rf = calcular_business_value(
            y_eval,
            y_pred_rf
        )

        # ==========================
        # TABLA COMPARATIVA
        # ==========================

        comparison_bv = pd.DataFrame({

            "Concepto": [
                "True Positives",
                "False Positives",
                "False Negatives",
                "Gain Unitario",
                "Business Value"
            ],

            "XGBoost": [
                bv_xgb["TP"],
                bv_xgb["FP"],
                bv_xgb["FN"],
                bv_xgb["Gain"],
                bv_xgb["Business Value"]
            ],

            "Random Forest": [
                bv_rf["TP"],
                bv_rf["FP"],
                bv_rf["FN"],
                bv_rf["Gain"],
                bv_rf["Business Value"]
            ]
        })

        st.dataframe(
            comparison_bv,
            width="stretch"
        )

        st.divider()

        # ==========================
        # IMPACTO ECONÓMICO COMPARATIVO
        # ==========================

        business_df = pd.DataFrame({

            "Concepto": [
                "Gain Total",
                "Costo Operativo",
                "Pérdida FN",
                "Business Value",

                "Gain Total",
                "Costo Operativo",
                "Pérdida FN",
                "Business Value"
            ],

            "Valor": [

                # XGBoost
                bv_xgb["Gain"] * bv_xgb["TP"],

                COSTO_FP * (
                    bv_xgb["TP"]
                    + bv_xgb["FP"]
                ),

                PERDIDA_FN * bv_xgb["FN"],

                bv_xgb["Business Value"],

                # Random Forest
                bv_rf["Gain"] * bv_rf["TP"],

                COSTO_FP * (
                    bv_rf["TP"]
                    + bv_rf["FP"]
                ),

                PERDIDA_FN * bv_rf["FN"],

                bv_rf["Business Value"]
            ],

            "Modelo": [

                "XGBoost",
                "XGBoost",
                "XGBoost",
                "XGBoost",

                "Random Forest",
                "Random Forest",
                "Random Forest",
                "Random Forest"
            ]
        })

        fig_bv = px.bar(

            business_df,

            x="Concepto",
            y="Valor",

            color="Modelo",

            barmode="group",

            template="plotly_dark",

            text_auto=".2s"
        )

        fig_bv.update_layout(

            title="💰 Comparación de Impacto Económico",

            title_font=dict(
                color="#FFFFFF",
                size=24
            ),

            font=dict(
                color="#FFFFFF",
                size=14
            ),

            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",

            xaxis=dict(
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF")
            ),

            yaxis=dict(
                title_font=dict(color="#FFFFFF"),
                tickfont=dict(color="#FFFFFF")
            ),

            legend=dict(
                font=dict(color="#FFFFFF")
            )
        )

        fig_bv.update_traces(
            textfont_color="#FFFFFF"
        )

        st.plotly_chart(
            fig_bv,
            width="stretch"
        )

        st.divider()

        # ==========================
        # GAUGES COMPARATIVOS
        # ==========================

        st.subheader(
            "🚦 Comparación de Business Value"
        )

        col1, col2 = st.columns(2)

        max_esperado = max(
            abs(bv_xgb["Business Value"]),
            abs(bv_rf["Business Value"]),
            1
        ) * 1.2

        # =====================================================
        # XGBOOST
        # =====================================================

        with col1:

            fig_gauge_xgb = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=bv_xgb["Business Value"],

                    title={
                        "text": "🚀 XGBoost",
                        "font": {
                            "size": 24,
                            "color": "#FFFFFF"
                        }
                    },

                    number={
                        "prefix": "$",
                        "font": {
                            "size": 34,
                            "color": "#FFFFFF"
                        }
                    },

                    gauge={

                        "axis": {
                            "range": [
                                -max_esperado,
                                max_esperado
                            ],
                            "tickcolor": "#FFFFFF"
                        },

                        "bar": {
                            "color": "#00CC96"
                        },

                        "bgcolor": "#1C2333",

                        "steps": [

                            {
                                "range": [
                                    -max_esperado,
                                    0
                                ],
                                "color": "#EF553B"
                            },

                            {
                                "range": [
                                    0,
                                    max_esperado * 0.5
                                ],
                                "color": "#FECB52"
                            },

                            {
                                "range": [
                                    max_esperado * 0.5,
                                    max_esperado
                                ],
                                "color": "#00CC96"
                            }
                        ]
                    }
                )
            )

            fig_gauge_xgb.update_layout(

                template="plotly_dark",

                paper_bgcolor="#0E1117",

                font={
                    "color": "#FFFFFF"
                },

                height=400
            )

            st.plotly_chart(
                fig_gauge_xgb,
                width="stretch"
            )

        # =====================================================
        # RANDOM FOREST
        # =====================================================

        with col2:

            fig_gauge_rf = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=bv_rf["Business Value"],

                    title={
                        "text": "🌲 Random Forest",
                        "font": {
                            "size": 24,
                            "color": "#FFFFFF"
                        }
                    },

                    number={
                        "prefix": "$",
                        "font": {
                            "size": 34,
                            "color": "#FFFFFF"
                        }
                    },

                    gauge={

                        "axis": {
                            "range": [
                                -max_esperado,
                                max_esperado
                            ],
                            "tickcolor": "#FFFFFF"
                        },

                        "bar": {
                            "color": "#636EFA"
                        },

                        "bgcolor": "#1C2333",

                        "steps": [

                            {
                                "range": [
                                    -max_esperado,
                                    0
                                ],
                                "color": "#EF553B"
                            },

                            {
                                "range": [
                                    0,
                                    max_esperado * 0.5
                                ],
                                "color": "#FECB52"
                            },

                            {
                                "range": [
                                    max_esperado * 0.5,
                                    max_esperado
                                ],
                                "color": "#636EFA"
                            }
                        ]
                    }
                )
            )

            fig_gauge_rf.update_layout(

                template="plotly_dark",

                paper_bgcolor="#0E1117",

                font={
                    "color": "#FFFFFF"
                },

                height=400
            )

            st.plotly_chart(
                fig_gauge_rf,
                width="stretch"
            )

        st.divider()

    # ==========================
    # THRESHOLD OPTIMIZATION
    # ==========================

    st.subheader(
        "📈 Optimización de Threshold"
    )

    # =====================================
    # VALIDACIÓN
    # =====================================

    y_true = (
        pd.Series(y_eval)
        .fillna(0)
        .astype(int)
        .values
    )

    thresholds = np.arange(
        0.05,
        1.0,
        0.05
    )

    business_values_xgb = []
    business_values_rf = []

    # =====================================
    # XGBOOST
    # =====================================

    for t in thresholds:

        pred_xgb_t = (
            y_proba_xgb >= t
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_eval,
            pred_xgb_t
        ).ravel()

        gain_total = (
            tp * BENEFICIO_TP
        )

        costo_total = (
            tp + fp
        ) * COSTO_FP

        perdida_fn = (
            fn * PERDIDA_FN
        )

        business_value = (
            gain_total
            - costo_total
        )

        if isinstance(business_value, np.ndarray):
            business_value = business_value[0]

        business_values_xgb.append(
            float(business_value)
        )

    # =====================================
    # RANDOM FOREST
    # =====================================

    for t in thresholds:

        pred_rf_t = (
            y_proba_rf >= t
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_eval,
            pred_rf_t
        ).ravel()

        gain_total = (
            tp * BENEFICIO_TP
        )

        costo_total = (
            tp + fp
        ) * COSTO_FP

        perdida_fn = (
            fn * PERDIDA_FN
        )

        business_value = (
            gain_total
            - costo_total
        )

        if isinstance(business_value, np.ndarray):
            business_value = business_value[0]

        business_values_rf.append(
            float(business_value)
        )

    # =====================================
    # DATAFRAME FINAL
    # =====================================

    threshold_df = pd.DataFrame({

        "Threshold": thresholds,

        "Business Value XGBoost":
            business_values_xgb,

        "Business Value Random Forest":
            business_values_rf
    })

    # =====================================
    # GRÁFICO
    # =====================================

    fig_threshold = go.Figure()

    fig_threshold.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"].tolist(),
            y=threshold_df[
                "Business Value XGBoost"
            ].tolist(),
            mode="lines+markers",
            name="XGBoost"
        )
    )

    fig_threshold.add_trace(
        go.Scatter(
            x=threshold_df["Threshold"].tolist(),
            y=threshold_df[
                "Business Value Random Forest"
            ].tolist(),
            mode="lines+markers",
            name="Random Forest"
        )
    )

    fig_threshold.update_layout(
        template="plotly_dark",
        title="📈 Threshold Optimization",
        xaxis_title="Threshold",
        yaxis_title="Business Value",
        font=dict(
            color="#FFFFFF",
            size=14
        ),
        title_font=dict(
            color="#FFFFFF",
            size=22
        )
    )

    st.plotly_chart(
        fig_threshold,
        width="stretch"
    )

    # ==========================================
    # MEJORES THRESHOLDS
    # ==========================================

    best_idx_xgb = threshold_df[
        "Business Value XGBoost"
    ].idxmax()

    best_threshold_xgb = threshold_df.loc[
        best_idx_xgb,
        "Threshold"
    ]

    best_bv_xgb = threshold_df.loc[
        best_idx_xgb,
        "Business Value XGBoost"
    ]

    best_idx_rf = threshold_df[
        "Business Value Random Forest"
    ].idxmax()

    best_threshold_rf = threshold_df.loc[
        best_idx_rf,
        "Threshold"
    ]

    best_bv_rf = threshold_df.loc[
        best_idx_rf,
        "Business Value Random Forest"
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"""
            🎯 XGBoost

            Threshold óptimo: {best_threshold_xgb:.2f}

            Business Value esperado:
            ${best_bv_xgb:,.0f}
            """
        )

    with col2:

        st.success(
            f"""
            🌲 Random Forest

            Threshold óptimo: {best_threshold_rf:.2f}

            Business Value esperado:
            ${best_bv_rf:,.0f}
            """
        )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader(
        "📋 Predicciones"
    )

    st.subheader("XGBoost")

    st.dataframe(
        results_xgb.head(100),
        width="stretch"
    )

    st.divider()

    st.subheader("Random Forest")

    st.dataframe(
        results_rf.head(100),
        width="stretch"
    )

    csv = results_xgb.to_csv(   
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Descargar predicciones XGBoost",
        csv,
        "predicciones_xgb.csv",
        "text/csv"
    )

    csv_rf = results_rf.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Descargar predicciones Random Forest",
        csv_rf,
        "predicciones_rf.csv",
        "text/csv"
    )

# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.subheader(
        "🔍 Explainability Comparativo"
    )

    tabs_shap = st.tabs([
        "XGBoost",
        "Random Forest"
    ])

    # =====================================================
    # XGBOOST SHAP
    # =====================================================

    with tabs_shap[0]:

        st.markdown("## 🚀 XGBoost SHAP")

        try:

            preprocessor_xgb = xgb_model.named_steps[
                "preprocessor"
            ]

            clf_xgb = xgb_model.named_steps[
                "clf"
            ]

            X_transformed_xgb = (
                preprocessor_xgb.transform(X_eval)
            )

            feature_names_xgb = (
                preprocessor_xgb.get_feature_names_out()
            )

            feature_names_xgb = [
                col.replace("num__", "")
                   .replace("cat__", "")
                for col in feature_names_xgb
            ]

            X_transformed_df_xgb = pd.DataFrame(
                X_transformed_xgb,
                columns=feature_names_xgb
            )

            sample_shap_xgb = (
                X_transformed_df_xgb.sample(
                    min(300, len(X_transformed_df_xgb)),
                    random_state=42
                )
            )

            explainer_xgb = shap.TreeExplainer(
                clf_xgb
            )

            shap_values_xgb = (
                explainer_xgb.shap_values(
                    sample_shap_xgb
                )
            )

            fig_shap_xgb, ax = plt.subplots(
                figsize=(12,8)
            )

            shap.summary_plot(
                shap_values_xgb,
                sample_shap_xgb,
                show=False
            )

            st.pyplot(fig_shap_xgb)

        except Exception as e:

            st.warning(
                f"⚠️ Error SHAP XGBoost: {e}"
            )

    # =====================================================
    # RANDOM FOREST SHAP
    # =====================================================

    with tabs_shap[1]:

        st.markdown("## 🌲 Random Forest SHAP")

        try:

            preprocessor_rf = rf_model.named_steps[
                "preprocessor"
            ]

            clf_rf = rf_model.named_steps[
                "clf"
            ]

            X_transformed_rf = (
                preprocessor_rf.transform(X_eval)
            )

            feature_names_rf = (
                preprocessor_rf.get_feature_names_out()
            )

            feature_names_rf = [
                col.replace("num__", "")
                   .replace("cat__", "")
                for col in feature_names_rf
            ]

            X_transformed_df_rf = pd.DataFrame(
                X_transformed_rf,
                columns=feature_names_rf
            )

            sample_shap_rf = (
                X_transformed_df_rf.sample(
                    min(300, len(X_transformed_df_rf)),
                    random_state=42
                )
            )

            explainer_rf = shap.TreeExplainer(
                clf_rf
            )

            shap_values_rf = explainer_rf(
                sample_shap_rf
            )

# =====================================================
# AJUSTE FORMATO RANDOM FOREST
# =====================================================

# pyrefly: ignore [missing-import]
            import numpy as np

# Caso lista
            if isinstance(shap_values_rf, list):

                shap_values_rf = shap_values_rf[1]

# Caso array 3D
            elif len(np.array(shap_values_rf).shape) == 3:

                shap_values_rf = shap_values_rf[:, :, 1]

            fig_shap_rf = plt.figure(
                figsize=(12,8)
            )

            shap.plots.beeswarm(
                shap_values_rf,
                show=False
            )

            st.pyplot(fig_shap_rf)
            plt.close()

        except Exception as e:

            st.warning(
                f"⚠️ Error SHAP Random Forest: {e}"
            )