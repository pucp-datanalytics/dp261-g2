import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

    mask_people = (
        (df["adults"] == 0)
        &
        (df["children"].fillna(0) == 0)
        &
        (df["babies"] == 0)
    )

    df = df[~mask_people]

    df.dropna(
        subset=["children"],
        inplace=True
    )

    df["children"] = df[
        "children"
    ].astype(int)

    df = df[
        df["children"] != 10
    ]

    df = df[
        df["adr"] >= 0
    ]

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

    # =====================================================
    # TOTAL NIGHTS
    # =====================================================

    df["total_nights"] = (
        df["stays_in_week_nights"]
        +
        df["stays_in_weekend_nights"]
    )

    # =====================================================
    # TOTAL GUESTS
    # =====================================================

    df["total_guests"] = (
        df["adults"]
        +
        df["children"]
        +
        df["babies"]
    )

    # =====================================================
    # HIGH SEASON
    # =====================================================

    high_season = [
        "July",
        "August"
    ]

    df["is_high_season"] = (
        df["arrival_date_month"]
        .isin(high_season)
        .astype(int)
    )

    # =====================================================
    # LOG TRANSFORMS
    # =====================================================

    df["adr_log"] = np.log1p(
        df["adr"]
    )

    df["lead_time_log"] = np.log1p(
        df["lead_time"]
    )

    return df

# =========================================================
# CARGAR MODELO
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    return model

model = load_model()

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
# CARGAR DATASET
# =========================================================

uploaded = st.file_uploader(
    "📂 Suba un archivo CSV",
    type="csv"
)

if uploaded is None:

    st.warning(
        "Suba un dataset para comenzar."
    )

    st.stop()

# =========================================================
# LEER DATASET
# =========================================================

df_eval = pd.read_csv(
    uploaded
)

st.success(
    "Dataset cargado correctamente"
)

# =========================================================
# PREVIEW
# =========================================================

with st.expander(
    "Vista previa del dataset"
):

    st.dataframe(
        df_eval.head(),
        width="stretch"
    )

# =========================================================
# GUARDAR ORIGINAL
# =========================================================

df_original = df_eval.copy()

# =========================================================
# CLEANING
# =========================================================

df_eval = clean_data(
    df_eval
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df_eval = feature_engineering(
    df_eval
)

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
# PREDICCIONES
# =========================================================

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
# TAB 1 — RESUMEN EJECUTIVO
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

    # =====================================================
    # KPIS
    # =====================================================

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

    st.success(f"""
    El modelo detectó que el
    {pct_alto:.1f}% de las reservas
    presenta alto riesgo de cancelación.
    """)

    # =====================================================
    # RIESGO POR HOTEL
    # =====================================================

    if "hotel" in results.columns:

        riesgo_hotel = (
            results.groupby("hotel")[
                "Probabilidad"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig_hotel = px.bar(
            riesgo_hotel,
            x="hotel",
            y="Probabilidad",
            title="Riesgo promedio por hotel"
        )

        st.plotly_chart(
            fig_hotel,
            width="stretch"
        )

    # =====================================================
    # RIESGO POR SEGMENTO
    # =====================================================

    if "market_segment" in results.columns:

        riesgo_segmento = (
            results.groupby(
                "market_segment"
            )["Probabilidad"]
            .mean()
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        fig_seg = px.bar(
            riesgo_segmento,
            x="market_segment",
            y="Probabilidad",
            title="Riesgo promedio por segmento"
        )

        st.plotly_chart(
            fig_seg,
            width="stretch"
        )

    # =====================================================
    # DISTRIBUCIÓN
    # =====================================================

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
# TAB 2 — PERFORMANCE ML
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

        # =================================================
        # KPIS
        # =================================================

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

        # =================================================
        # MATRIZ + ROC
        # =================================================

        col1, col2 = st.columns(2)

        # =================================================
        # MATRIZ
        # =================================================

        with col1:

            st.write(
                "### Matriz de Confusión"
            )

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

            ax.set_xlabel(
                "Predicción"
            )

            ax.set_ylabel(
                "Real"
            )

            st.pyplot(fig)

        # =================================================
        # ROC
        # =================================================

        with col2:

            st.write(
                "### Curva ROC"
            )

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

    else:

        st.info("""
        El dataset no contiene la
        variable objetivo
        'is_canceled'.
        """)

# =========================================================
# TAB 3 — PREDICCIONES
# =========================================================

with tab3:

    st.subheader(
        "Simulador de Predicciones"
    )

    filtro = st.selectbox(
        "Filtrar resultados",
        [
            "Todos",
            "Cancelará",
            "No cancelará"
        ]
    )

    results_filtered = results.copy()

    if filtro != "Todos":

        results_filtered = results_filtered[
            results_filtered[
                "Predicción"
            ] == filtro
        ]

    # =====================================================
    # KPIS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Reservas",
        len(results_filtered)
    )

    col2.metric(
        "Cancelaciones Predichas",
        int(
            (
                results_filtered[
                    "Predicción"
                ]
                ==
                "Cancelará"
            ).sum()
        )
    )

    col3.metric(
        "Riesgo Promedio",
        f"{results_filtered['Probabilidad'].mean():.2%}"
    )

    st.divider()

    # =====================================================
    # TOP RIESGO
    # =====================================================

    st.write(
        "### Reservas críticas"
    )

    top_riesgo = results_filtered.sort_values(
        "Probabilidad",
        ascending=False
    ).head(10)

    st.dataframe(
        top_riesgo,
        width="stretch"
    )

    st.divider()

    # =====================================================
    # RESULTADOS
    # =====================================================

    st.write(
        "### Resultados completos"
    )

    st.dataframe(
        results_filtered.head(100),
        width="stretch"
    )

    # =====================================================
    # DESCARGA
    # =====================================================

    csv = results_filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Descargar resultados",
        data=csv,
        file_name="predicciones.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 4 — SHAP
# =========================================================

with tab4:

    st.subheader(
        "Explainability con SHAP"
    )

    st.write("""
    SHAP permite interpretar cómo
    cada variable impacta en las
    predicciones del modelo.
    """)

    # =====================================================
    # COMPONENTES PIPELINE
    # =====================================================

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    clf = model.named_steps[
        "clf"
    ]

    # =====================================================
    # TRANSFORMAR
    # =====================================================

    X_transformed = preprocessor.transform(
        X_eval
    )

    # =====================================================
    # FEATURE NAMES
    # =====================================================

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

    # =====================================================
    # SAMPLE
    # =====================================================

    sample_shap = X_transformed_df.sample(
        min(300, len(X_transformed_df)),
        random_state=42
    )

    # =====================================================
    # SHAP VALUES
    # =====================================================

    explainer = shap.TreeExplainer(
        clf
    )

    shap_values = explainer.shap_values(
        sample_shap
    )

    # =====================================================
    # SUMMARY PLOT
    # =====================================================

    fig_shap, ax = plt.subplots(
        figsize=(12,8)
    )

    shap.summary_plot(
        shap_values,
        sample_shap,
        show=False
    )

    st.pyplot(fig_shap)

    st.info("""
    Las variables ubicadas arriba
    son las más influyentes para
    el modelo.
    """)