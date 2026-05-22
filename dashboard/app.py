# =========================================================
# LIBRERÍAS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import joblib
import requests
import os
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
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Dashboard de cancelaciones hoteleras",
    layout="wide"
)

# =========================================================
# ESTILOS
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

</style>
""", unsafe_allow_html=True)

# =========================================================
# RUTA DEL MODELO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "final_pipeline.pkl"
)

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

        df["adr"] = df[
            "adr"
        ].clip(upper=p99)

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

    df["adr_log"] = np.log1p(
        df["adr"]
    )

    df["lead_time_log"] = np.log1p(
        df["lead_time"]
    )

    return df

# =========================================================
# CONFIGURACIÓN API
# =========================================================

API_URL = os.getenv(
    "API_URL",
    "http://3.81.51.149:8000"
)

@st.cache_resource
def load_model():

    try:

        return joblib.load(MODEL_PATH)

    except Exception:

        return None

model = load_model()

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
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Panel de Control"
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

st.title(
    "Dashboard de cancelaciones hoteleras"
)

st.write("""
Sistema interactivo de predicción
de cancelaciones hoteleras basado
en Machine Learning.
""")

st.divider()

# =========================================================
# MODO DE USO
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
# PREDICCIÓN MASIVA
# =========================================================

if modo == "📂 Predicción Masiva":

    uploaded = st.file_uploader(
        "📂 Suba un archivo CSV",
        type="csv"
    )

    if uploaded is None:

        st.warning(
            "Suba un dataset para comenzar."
        )

        st.stop()

    df_eval = pd.read_csv(uploaded)

    st.success(
        "Dataset cargado correctamente"
    )

    with st.expander(
        "Vista previa del dataset"
    ):

        st.dataframe(
            df_eval.head(),
            width="stretch"
        )

# =========================================================
# SIMULACIÓN MANUAL
# =========================================================

else:

    st.subheader(
        "Simulación manual de reserva"
    )

    with st.form("manual_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            hotel = st.selectbox(
                "Hotel",
                [
                    "Resort Hotel",
                    "City Hotel"
                ]
            )

            lead_time = st.slider(
                "Lead Time",
                0,
                700,
                120
            )

            adr = st.number_input(
                "ADR",
                0.0,
                1000.0,
                120.0
            )

            month = st.selectbox(
                "Mes llegada",
                [
                    "January","February","March",
                    "April","May","June",
                    "July","August","September",
                    "October","November","December"
                ]
            )

        with col2:

            adults = st.slider(
                "Adultos",
                1,
                5,
                2
            )

            children = st.slider(
                "Niños",
                0,
                4,
                0
            )

            babies = st.slider(
                "Bebés",
                0,
                2,
                0
            )

            previous_cancellations = st.slider(
                "Cancelaciones previas",
                0,
                10,
                0
            )

        with col3:

            market_segment = st.selectbox(
                "Segmento",
                [
                    "Online TA",
                    "Offline TA/TO",
                    "Direct",
                    "Groups",
                    "Corporate"
                ]
            )

            deposit_type = st.selectbox(
                "Depósito",
                [
                    "No Deposit",
                    "Refundable",
                    "Non Refund"
                ]
            )

            special_requests = st.slider(
                "Solicitudes especiales",
                0,
                5,
                1
            )

            booking_changes = st.slider(
                "Cambios de reserva",
                0,
                10,
                0
            )

        submitted = st.form_submit_button(
            "Predecir cancelación"
        )

    if not submitted:

        st.stop()

    df_eval = pd.DataFrame([{

        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": 2017,
        "arrival_date_month": month,
        "arrival_date_week_number": 30,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": "BB",
        "market_segment": market_segment,
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": adr,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": special_requests

    }])

    st.success(
        "Reserva generada correctamente"
    )

    st.dataframe(df_eval)

# =========================================================
# GUARDAR ORIGINAL
# =========================================================

df_original = df_eval.copy()

# =========================================================
# CLEANING
# =========================================================

df_eval = clean_data(df_eval)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df_eval = feature_engineering(df_eval)

# =========================================================
# DEFINIR X / Y
# =========================================================

if "is_canceled" in df_eval.columns:

    X_eval = df_eval.drop(
        "is_canceled",
        axis=1
    )

    y_eval = df_eval[
        "is_canceled"
    ]

else:

    X_eval = df_eval.copy()

    y_eval = None

# =========================================================
# PREDICCIÓN API
# =========================================================

def predecir_con_api(df_input):

    probabilidades = []

    for _, row in df_input.iterrows():

        try:

            mes_map = {
                "January":1,"February":2,"March":3,
                "April":4,"May":5,"June":6,
                "July":7,"August":8,"September":9,
                "October":10,"November":11,"December":12
            }

            mes_num = mes_map.get(
                str(row.get("arrival_date_month","January")),
                1
            )

            arrival_date = (
                f"{int(row.get('arrival_date_year',2017))}-"
                f"{mes_num:02d}-"
                f"{int(row.get('arrival_date_day_of_month',1)):02d}"
            )

            payload = {

                "hotel": str(row.get("hotel","City Hotel")),
                "lead_time": int(row.get("lead_time",0)),
                "arrival_date": arrival_date,
                "stays_in_weekend_nights": int(row.get("stays_in_weekend_nights",0)),
                "stays_in_week_nights": int(row.get("stays_in_week_nights",1)),
                "adults": int(row.get("adults",2)),
                "children": int(row.get("children",0)),
                "babies": int(row.get("babies",0)),
                "meal": str(row.get("meal","BB")),
                "market_segment": str(row.get("market_segment","Online TA")),
                "distribution_channel": str(row.get("distribution_channel","TA/TO")),
                "is_repeated_guest": int(row.get("is_repeated_guest",0)),
                "previous_cancellations": int(row.get("previous_cancellations",0)),
                "previous_bookings_not_canceled": int(row.get("previous_bookings_not_canceled",0)),
                "reserved_room_type": str(row.get("reserved_room_type","A")),
                "assigned_room_type": str(row.get("assigned_room_type","A")),
                "booking_changes": int(row.get("booking_changes",0)),
                "deposit_type": str(row.get("deposit_type","No Deposit")),
                "days_in_waiting_list": int(row.get("days_in_waiting_list",0)),
                "customer_type": str(row.get("customer_type","Transient")),
                "adr": float(row.get("adr",100.0)),
                "required_car_parking_spaces": int(row.get("required_car_parking_spaces",0)),
                "total_of_special_requests": int(row.get("total_of_special_requests",0))

            }

            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10
            )

            prob = resp.json().get(
                "probability",
                0.0
            )

        except Exception:

            prob = 0.0

        probabilidades.append(prob)

    return np.array(probabilidades)

# =========================================================
# PREDECIR
# =========================================================

usar_api = api_disponible()

if usar_api:

    st.sidebar.success(
        "API AWS conectada"
    )

    y_proba = predecir_con_api(
        df_original
    )

else:

    st.sidebar.warning(
        "Usando modelo local"
    )

    y_proba = model.predict_proba(
        X_eval
    )[:,1]

y_pred = (
    y_proba >= threshold
).astype(int)

# =========================================================
# RESULTADOS
# =========================================================

results = X_eval.copy()

results["Probabilidad"] = np.round(
    y_proba,
    3
)

results["Predicción"] = np.where(
    y_pred == 1,
    "Cancelará",
    "No cancelará"
)

# =========================================================
# NIVEL DE RIESGO
# =========================================================

def riesgo(p):

    if p > 0.7:
        return "🔴 Alto"

    elif p > 0.4:
        return "🟡 Medio"

    else:
        return "🟢 Bajo"

results["Nivel de Riesgo"] = results[
    "Probabilidad"
].apply(riesgo)

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

    st.subheader(
        "Resumen Ejecutivo"
    )

    total = len(results)

    cancelaciones = (
        results["Predicción"]
        ==
        "Cancelará"
    ).sum()

    riesgo_prom = results[
        "Probabilidad"
    ].mean()

    pct_alto = (
        results["Nivel de Riesgo"]
        ==
        "🔴 Alto"
    ).mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Reservas Totales",
        total
    )

    col2.metric(
        "Cancelaciones Predichas",
        cancelaciones
    )

    col3.metric(
        "Riesgo Promedio",
        f"{riesgo_prom:.2%}"
    )

    col4.metric(
        "% Riesgo Alto",
        f"{pct_alto:.1f}%"
    )

    st.divider()

    fig_dist = px.histogram(
        results,
        x="Probabilidad",
        nbins=30,
        title="Distribución de probabilidades"
    )

    st.plotly_chart(
        fig_dist,
        width="stretch"
    )

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader(
        "Performance del Modelo"
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
            "Accuracy",
            f"{accuracy:.2%}"
        )

        col2.metric(
            "Recall",
            f"{recall:.2%}"
        )

        col3.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

        col4.metric(
            "AUC ROC",
            f"{auc:.2%}"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            cm = confusion_matrix(
                y_eval,
                y_pred
            )

            fig, ax = plt.subplots(
                figsize=(6,5)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax
            )

            st.pyplot(fig)

        with col2:

            fpr, tpr, _ = roc_curve(
                y_eval,
                y_proba
            )

            fig2, ax2 = plt.subplots(
                figsize=(6,5)
            )

            ax2.plot(
                fpr,
                tpr,
                linewidth=3,
                label=f"AUC = {auc:.3f}"
            )

            ax2.plot(
                [0,1],
                [0,1],
                "--"
            )

            ax2.legend()

            st.pyplot(fig2)

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader(
        "Predicciones"
    )

    st.dataframe(
        results.head(100),
        width="stretch"
    )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.subheader(
        "Explainability con SHAP"
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
            f"Error SHAP: {e}"
        )
