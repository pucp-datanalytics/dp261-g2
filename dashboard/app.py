import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ========================
# CONFIGURACIÓN
# ========================

st.set_page_config(
    page_title="Dashboard de cancelaciones hoteleras",
    layout="wide"
)

# ========================
# DARK MODE CUSTOM
# ========================

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

# ========================
# CARGAR MODELO
# ========================

model = joblib.load(
    "models/final_model.pkl"
)

# ========================
# SIDEBAR
# ========================

st.sidebar.title("Panel de Control")

st.sidebar.caption(
    "Predicción inteligente de cancelaciones hoteleras"
)

st.sidebar.subheader("Métricas del Modelo")

st.sidebar.metric(
    "Recall",
    "77.5%"
)

st.sidebar.metric(
    "AUC-ROC",
    "93.6%"
)

st.sidebar.metric(
    "F1 Score",
    "81.2%"
)

st.sidebar.divider()

threshold = st.sidebar.slider(
    "Threshold de decisión",
    0.0,
    1.0,
    0.35
)

st.sidebar.write(
    f"Threshold actual: {threshold:.2f}"
)

# ========================
# HEADER
# ========================

st.title("Dashboard de cancelaciones hoteleras")

st.write("""
Dashboard interactivo para predicción
de cancelaciones hoteleras.
""")

st.markdown("""
### Objetivo del Dashboard

Este dashboard permite:

- Evaluar el performance del modelo
- Simular predicciones de cancelación
- Explicar decisiones usando SHAP
""")

st.divider()

# ========================
# TABS
# ========================

tab1, tab2, tab3 = st.tabs([
    "📊 Performance",
    "🤖 Predicciones",
    "🔍 SHAP Explainability"
])

# =========================================================
# TAB 1 — PERFORMANCE
# =========================================================

with tab1:

    st.subheader("Performance del Modelo")

    col1, col2 = st.columns(2)

    # =====================================================
    # MATRIZ DE CONFUSIÓN
    # =====================================================

    with col1:

        st.write("### Matriz de Confusión")

        cm = np.array([
            [13826, 1176],
            [1986, 6853]
        ])

        fig, ax = plt.subplots(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )

        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

        ax.set_xticklabels([
            "No Cancela",
            "Cancela"
        ])

        ax.set_yticklabels([
            "No Cancela",
            "Cancela"
        ])

        st.pyplot(fig)

    # =====================================================
    # CURVA ROC
    # =====================================================

    with col2:

        st.write("### Curva ROC")

        fpr = [0, 0.05, 0.1, 0.2, 1]
        tpr = [0, 0.75, 0.84, 0.92, 1]

        fig2, ax2 = plt.subplots(figsize=(6,5))

        ax2.plot(
            fpr,
            tpr,
            linewidth=3,
            label="AUC = 0.936"
        )

        ax2.plot(
            [0,1],
            [0,1],
            "--"
        )

        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")

        ax2.legend()

        st.pyplot(fig2)

    # =====================================================
    # EXPLICACIÓN
    # =====================================================

    st.info("""
    El modelo tiene alta capacidad de discriminación
    entre reservas que cancelan y no cancelan.
    """)

# =========================================================
# TAB 2 — PREDICCIONES
# =========================================================

with tab2:

    st.subheader("Simulador de Predicciones")

    uploaded = st.file_uploader(
        "Sube un archivo CSV",
        type="csv"
    )

    if uploaded:

        try:

            # =================================================
            # LEER CSV
            # =================================================

            df = pd.read_csv(uploaded)

            st.success(
                "Dataset cargado correctamente"
            )

            # =================================================
            # PREVIEW
            # =================================================

            with st.expander(
                "Vista previa del dataset"
            ):

                st.dataframe(
                    df.head()
                )

            # =================================================
            # PREDICCIONES
            # =================================================

            proba = model.predict_proba(df)[:,1]

            preds = (
                proba >= threshold
            ).astype(int)

            # =================================================
            # RESULTS
            # =================================================

            results = pd.DataFrame()

            results["Probabilidad"] = np.round(
                proba,
                3
            )

            results["Predicción"] = np.where(
                preds == 1,
                "Cancelará",
                "No cancelará"
            )

            # =================================================
            # RIESGO
            # =================================================

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

            # =================================================
            # KPIS
            # =================================================

            st.subheader(
                "Resumen de Predicciones"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Reservas",
                len(results)
            )

            col2.metric(
                "Cancelaciones Predichas",
                int((preds == 1).sum())
            )

            col3.metric(
                "Riesgo Promedio",
                f"{proba.mean():.2%}"
            )

            # =================================================
            # GRÁFICO
            # =================================================

            st.write(
                "### Distribución del Riesgo"
            )

            fig3, ax3 = plt.subplots(
                figsize=(8,4)
            )

            ax3.hist(
                proba,
                bins=20
            )

            ax3.set_xlabel(
                "Probabilidad de Cancelación"
            )

            ax3.set_ylabel(
                "Cantidad"
            )

            st.pyplot(fig3)

            # =================================================
            # FILTRO
            # =================================================

            st.subheader(
                "Resultados de Predicción"
            )

            filtro = st.selectbox(
                "Filtrar resultados",
                [
                    "Todas",
                    "Cancelará",
                    "No cancelará"
                ]
            )

            if filtro == "Cancelará":

                filtered_results = results[
                    results["Predicción"] == "Cancelará"
                ]

            elif filtro == "No cancelará":

                filtered_results = results[
                    results["Predicción"] == "No cancelará"
                ]

            else:

                filtered_results = results

            # =================================================
            # TABLA
            # =================================================

            st.dataframe(
                filtered_results,
                use_container_width=True
            )

            # =================================================
            # DESCARGA
            # =================================================

            csv = filtered_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Descargar Resultados",
                data=csv,
                file_name="predicciones_filtradas.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(
                f"Error procesando archivo: {e}"
            )

# =========================================================
# TAB 3 — SHAP
# =========================================================

with tab3:

    st.subheader(
        "Explainability con SHAP"
    )

    st.write("""
    SHAP permite entender por qué el modelo
    predice cancelación para una reserva específica.
    """)

    st.image(
        "dashboard/assets/shap_summary.png",
        caption="SHAP Summary Plot",
        use_container_width=True
    )

    st.info("""
    Variables como depósito no reembolsable,
    lead time y cancelaciones previas tienen
    fuerte impacto en la predicción.
    """)