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
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #1E1E2F;
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
    color: white;
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
    color: white;
}
.card-text {
    font-size: 16px;
    margin-bottom: 10px;
    color: white;
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
        color: white;
    }

    .card-text {
        font-size: 16px;
        margin-bottom: 10px;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_pipeline.pkl"
)

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)

model = load_model()

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

st.sidebar.write(
    f"Threshold actual: {threshold:.2f}"
)

st.sidebar.info("""
Threshold bajos:
↑ Recall
↓ Precision

Threshold altos:
↓ Recall
↑ Precision
""")

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
                "⏳ Lead Time",
                0,
                700,
                120
            )

            adr = st.number_input(
                "💰 ADR",
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

            adults = st.slider(
                "🧑 Adultos",
                1,
                5,
                2
            )

            children = st.slider(
                "🧒 Niños",
                0,
                4,
                0
            )

            babies = st.slider(
                "👶 Bebés",
                0,
                2,
                0
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

API_URL = os.getenv("API_URL", "http://34.204.98.17:8000")

def api_disponible():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def predecir_con_api(df_input):
    probabilidades = []
    progress = st.progress(0)
    total = len(df_input)
    mes_map = {
        "January":1,"February":2,"March":3,"April":4,
        "May":5,"June":6,"July":7,"August":8,
        "September":9,"October":10,"November":11,"December":12
    }
    for i, (_, row) in enumerate(df_input.iterrows()):
        try:
            mes_num = mes_map.get(str(row.get("arrival_date_month","January")), 1)
            arrival_date = (
                f"{int(row.get('arrival_date_year', 2017))}-"
                f"{mes_num:02d}-"
                f"{int(row.get('arrival_date_day_of_month', 1)):02d}"
            )
            payload = {
                "hotel": str(row.get("hotel", "City Hotel")),
                "lead_time": int(row.get("lead_time", 0)),
                "arrival_date": arrival_date,
                "stays_in_weekend_nights": int(row.get("stays_in_weekend_nights", 0)),
                "stays_in_week_nights": int(row.get("stays_in_week_nights", 1)),
                "adults": int(row.get("adults", 2)),
                "children": int(row.get("children", 0) or 0),
                "babies": int(row.get("babies", 0)),
                "meal": str(row.get("meal", "BB")),
                "market_segment": str(row.get("market_segment", "Online TA")),
                "distribution_channel": str(row.get("distribution_channel", "TA/TO")),
                "is_repeated_guest": int(row.get("is_repeated_guest", 0)),
                "previous_cancellations": int(row.get("previous_cancellations", 0)),
                "previous_bookings_not_canceled": int(row.get("previous_bookings_not_canceled", 0)),
                "reserved_room_type": str(row.get("reserved_room_type", "A")),
                "assigned_room_type": str(row.get("assigned_room_type", "A")),
                "booking_changes": int(row.get("booking_changes", 0)),
                "deposit_type": str(row.get("deposit_type", "No Deposit")),
                "days_in_waiting_list": int(row.get("days_in_waiting_list", 0)),
                "customer_type": str(row.get("customer_type", "Transient")),
                "adr": float(row.get("adr", 100.0)),
                "required_car_parking_spaces": int(row.get("required_car_parking_spaces", 0)),
                "total_of_special_requests": int(row.get("total_of_special_requests", 0)),
            }
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            prob = resp.json().get("probability", 0.0)
        except Exception:
            prob = 0.0
        probabilidades.append(prob)
        progress.progress((i + 1) / total)
    progress.empty()
    return np.array(probabilidades)

usar_api = api_disponible()

if usar_api:
    st.sidebar.success("✅ API activa — prediciendo con AWS")
    y_proba = predecir_con_api(df_original)
else:
    st.sidebar.warning("⚠️ API no disponible — usando modelo local")
    y_proba = model.predict_proba(X_eval)[:,1]

y_pred = (
    y_proba >= threshold
).astype(int)

# =========================================================
# RESULTS
# =========================================================

results = X_eval.copy()

results["Probabilidad"] = np.round(
    y_proba,
    3
)

results["Predicción"] = np.where(
    y_pred == 1,
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

results["Nivel de Riesgo"] = results[
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

results["Acción Recomendada"] = results[
    "Nivel de Riesgo"
].apply(accion_recomendada)

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

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Resumen Ejecutivo",
    "🤖 Performance ML",
    "📋 Predicciones",
    "🔍 Explainability"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    total = len(results)

    cancelaciones = (
        results["Predicción"]
        ==
        "❌ Cancelará"
    ).sum()

    riesgo_prom = results[
        "Probabilidad"
    ].mean()

    pct_alto = (
        results["Nivel de Riesgo"]
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
        results["Nivel de Riesgo"]
        == "🔴 Riesgo Alto"
    ).sum()

    medio = (
        results["Nivel de Riesgo"]
        == "🟡 Riesgo Medio"
    ).sum()

    bajo = (
        results["Nivel de Riesgo"]
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
            results,
            x="Probabilidad",
            color="Nivel de Riesgo",
            template="plotly_dark",
            nbins=30,
            color_discrete_map=risk_colors
        )

        fig_hist.update_layout(
            title="📈 Distribución de Probabilidades",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )

        st.plotly_chart(
            fig_hist,
            width="stretch"
        )

    with col2:

        riesgo_counts = (
            results["Nivel de Riesgo"]
            .value_counts()
            .reset_index()
        )

        riesgo_counts.columns = [
            "Nivel de Riesgo",
            "Cantidad"
        ]

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
            paper_bgcolor="#0E1117"
        )

        st.plotly_chart(
            fig_pie,
            width="stretch"
        )
# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader(
        "🤖 Performance del Modelo"
    )

    if y_eval is not None:

        accuracy = accuracy_score(
            y_eval,
            y_pred
        )

        recall = recall_score(
            y_eval,
            y_pred
        )

        f1 = f1_score(
            y_eval,
            y_pred
        )

        auc = roc_auc_score(
            y_eval,
            y_proba
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "🎯 Accuracy",
            f"{accuracy:.2%}"
        )

        col2.metric(
            "📡 Recall",
            f"{recall:.2%}"
        )

        col3.metric(
            "⚖️ F1 Score",
            f"{f1:.2%}"
        )

        col4.metric(
            "📈 AUC ROC",
            f"{auc:.2%}"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            cm = confusion_matrix(
                y_eval,
                y_pred
            )

            fig_cm = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="RdYlGn",
                title="🧩 Confusion Matrix"
            )

            st.plotly_chart(
                fig_cm,
                width="stretch"
            )

        with col2:

            fpr, tpr, _ = roc_curve(
                y_eval,
                y_proba
            )

            fig_roc = go.Figure()

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"AUC = {auc:.3f}",
                    line=dict(
                        color="#00CC96",
                        width=4
                    )
                )
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0,1],
                    y=[0,1],
                    mode="lines",
                    line=dict(
                        dash="dash",
                        color="#EF553B"
                    ),
                    showlegend=False
                )
            )

            fig_roc.update_layout(
                template="plotly_dark",
                title="📈 ROC Curve"
            )

            st.plotly_chart(
                fig_roc,
                width="stretch"
            )

    else:

        st.info(
            "📂 Suba dataset con is_canceled para evaluar métricas."
        )

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader(
        "📋 Predicciones"
    )

    st.dataframe(
        results.head(100),
        width="stretch"
    )

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Descargar predicciones",
        csv,
        "predicciones.csv",
        "text/csv"
    )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader(
        "🔍 Explainability con SHAP"
    )

    try:

        preprocessor = model.named_steps[
            "preprocessor"
        ]

        clf = model.named_steps[
            "clf"
        ]

        X_transformed = preprocessor.transform(
            X_eval
        )

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        feature_names = [
            col.replace("num__", "")
               .replace("cat__", "")
            for col in feature_names
        ]

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        sample_shap = X_transformed_df.sample(
            min(300, len(X_transformed_df)),
            random_state=42
        )

        explainer = shap.TreeExplainer(
            clf
        )

        shap_values = explainer.shap_values(
            sample_shap
        )

        fig_shap, ax = plt.subplots(
            figsize=(12,8)
        )

        shap.summary_plot(
            shap_values,
            sample_shap,
            show=False
        )

        st.pyplot(fig_shap)

    except Exception as e:

        st.warning(
            f"⚠️ Error SHAP: {e}"
        )