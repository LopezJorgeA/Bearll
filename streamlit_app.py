import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import base64

from data_loader import DataLoader
from processor import DataProcessor
from model import StockModel


st.set_page_config(page_title="NVDA Stock Predictor", layout="wide")


def _load_header_image_base64(path: str) -> str:
    """Devuelve la imagen codificada en base64 para usarla como background en CSS."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


_header_img_b64 = _load_header_image_base64("assets/TheWallStreet.jpg")

# Header tipo banner con imagen de fondo y título encima
st.markdown(
    f"""
    <style>
    .nvda-header {{
        position: relative;
        width: 100%;
        height: 180px;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }}
    .nvda-header-bg {{
        position: absolute;
        inset: 0;
        background-image: url('data:image/jpg;base64,{_header_img_b64}');
        background-size: cover;
        background-position: center;
        filter: brightness(0.45);
    }}
    .nvda-header-content {{
        position: relative;
        z-index: 1;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0 2rem;
        color: #f9fafb;
    }}
    .nvda-header-title {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }}
    .nvda-header-subtitle {{
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }}
    </style>
    <div class="nvda-header">
        <div class="nvda-header-bg"></div>
        <div class="nvda-header-content">
            <h1 class="nvda-header-title">📈 NVDA Stock Predictor</h1>
            <div class="nvda-header-subtitle">Interfaz web para ver las predicciones del modelo y su histórico de aciertos.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=True)
def load_data(days_back: int = 730):
    loader = DataLoader()
    start = datetime.now() - timedelta(days=days_back)
    df_raw = loader.get_stock_data(["NVDA", "SPY"], start)
    processor = DataProcessor(df_raw)
    df_enriched = processor.add_technical_indicators()
    return df_raw, df_enriched


@st.cache_resource(show_spinner=True)
def load_model(df_enriched: pd.DataFrame) -> StockModel:
    model = StockModel(df_enriched)
    model.prepare_data()
    model.train()
    return model


st.markdown("---")
st.subheader("Parámetros del modelo")

params_col1, params_col2 = st.columns([3, 1])
with params_col1:
    days_back = st.slider(
        "Días pasados para cargar datos históricos",
        min_value=180,
        max_value=1095,
        value=730,
        step=30,
    )
with params_col2:
    run_button = st.button("🔄 Actualizar datos y predicción", use_container_width=True)

if run_button:
    st.session_state["run_clicked"] = True

if "run_clicked" not in st.session_state:
    st.session_state["run_clicked"] = True

if st.session_state["run_clicked"]:
    with st.spinner("Descargando datos y entrenando modelo..."):
        df_raw, df_enriched = load_data(days_back)
        model = load_model(df_enriched)

    last_row = df_enriched.iloc[-1]
    last_ts = pd.to_datetime(last_row["timestamp"])
    last_close = float(last_row["close"])

    prediction = model.predict_tomorrow(last_close)

    predicted_price = prediction["predicted_price"]
    predicted_return = prediction["predicted_return"] * 100
    predicted_open = prediction["predicted_open_price"]
    predicted_open_return = prediction["predicted_open_return"] * 100
    prob_up = prediction["prob_up"] * 100
    direction = prediction["direction"]
    action = prediction["action"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Último cierre conocido")
        st.metric(
            label=str(last_ts.date()),
            value=f"${last_close:,.2f}",
        )

        # Botón para refrescar exclusivamente el último cierre y las predicciones
        if st.button("Actualizar último cierre", key="refresh_last_close"):
            # Forzamos una nueva ejecución completa del flujo
            st.session_state["run_clicked"] = True
            # Vuelve a ejecutar el script para tomar datos más recientes
            st.rerun()

    with col2:
        st.subheader("Cierre predicho próximo día hábil")
        st.metric(
            label=f"Cambio esperado ({direction})",
            value=f"${predicted_price:,.2f}",
            delta=f"{predicted_return:.2f}%",
        )

    with col3:
        st.subheader("Apertura predicha próximo día hábil")
        st.metric(
            label="vs. último cierre",
            value=f"${predicted_open:,.2f}",
            delta=f"{predicted_open_return:.2f}%",
        )

    st.markdown("---")
    st.subheader("Señal y probabilidad")
    st.write(f"**Probabilidad de movimiento al alza:** {prob_up:.2f}%")
    st.write(f"**Sugerencia de posición (modelo):** {action}")
    st.caption("Modelo cuantitativo, no es consejo financiero.")

    # Gráfico de precios recientes
    st.markdown("---")
    st.subheader("Histórico reciente de precios (NVDA)")
    plot_df = df_enriched[["timestamp", "close"]].tail(200).copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])
    plot_df.set_index("timestamp", inplace=True)
    st.line_chart(plot_df["close"], height=300)

    # Historial de predicciones si existe el CSV que genera main.py
    st.markdown("---")
    st.subheader("Histórico de predicciones")
    try:
        log_df = pd.read_csv("predictions_log.csv")
        st.dataframe(log_df.tail(20))

        # Cálculo de error usando los precios reales disponibles en df_enriched
        hist_df = log_df.copy()
        hist_df["target_date"] = pd.to_datetime(hist_df["target_date"]).dt.date

        prices_df = df_enriched.copy()
        prices_df["date"] = pd.to_datetime(prices_df["timestamp"]).dt.date

        merged = hist_df.merge(
            prices_df[["date", "close"]],
            left_on="target_date",
            right_on="date",
            how="left",
        )

        valid = merged.dropna(subset=["close"])

        if len(valid) > 0:
            actual_close = valid["close"]
            pred_close = valid["predicted_close"]

            # Error porcentual absoluto por predicción
            ape = (pred_close - actual_close).abs() / actual_close * 100.0

            mape = ape.mean()
            last_error = ape.iloc[-1]

            st.subheader("Métricas de error en cierre")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric(
                    label="Error porcentual medio (MAPE)",
                    value=f"{mape:.2f}%",
                )
            with mcol2:
                st.metric(
                    label="Error porcentual de la última predicción con dato real",
                    value=f"{last_error:.2f}%",
                )
        else:
            st.info(
                "Todavía no hay días en los que se conozca el precio real de cierre para las fechas predichas."
            )
    except FileNotFoundError:
        st.info("Aún no existe predictions_log.csv. Ejecuta main.py varias veces para ir guardando predicciones.")
