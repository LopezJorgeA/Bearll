from data_loader import DataLoader
from processor import DataProcessor
from model import StockModel
from datetime import datetime, timedelta
import pandas as pd
import os


def log_and_evaluate_predictions(df_enriched, prediction):
    """Guarda la predicción del día actual y calcula métricas históricas de acierto."""
    from pandas.tseries.offsets import BDay

    last_row = df_enriched.iloc[-1]
    last_ts = pd.to_datetime(last_row['timestamp'])

    # Próximo día hábil aproximado para NASDAQ
    target_date = (last_ts + BDay(1)).date()

    log_path = "predictions_log.csv"

    row = {
        "prediction_datetime": pd.Timestamp.utcnow().isoformat(),
        "last_timestamp": str(last_ts),
        "target_date": str(target_date),
        "last_close": float(last_row['close']),
        "predicted_close": prediction["predicted_price"],
        "predicted_return": prediction["predicted_return"],
        "predicted_open": prediction.get("predicted_open_price"),
        "predicted_open_return": prediction.get("predicted_open_return"),
        "direction": prediction["direction"],
        "action": prediction["action"],
    }

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)
    else:
        log_df = pd.DataFrame([row])

    log_df.to_csv(log_path, index=False)

    # ---- Cálculo de métricas históricas ----
    # Vinculamos target_date con los precios reales descargados
    hist_df = log_df.copy()
    hist_df['target_date'] = pd.to_datetime(hist_df['target_date']).dt.date

    prices_df = df_enriched.copy()
    prices_df['date'] = pd.to_datetime(prices_df['timestamp']).dt.date

    merged = hist_df.merge(
        prices_df[['date', 'close', 'open']],
        left_on='target_date',
        right_on='date',
        how='left'
    )

    # Filas donde ya conocemos el precio real de cierre del día predicho
    valid = merged.dropna(subset=['close'])

    if len(valid) == 0:
        print("\n📊 Aún no hay suficientes datos reales para evaluar el histórico de predicciones.")
        return

    actual_close = valid['close']
    pred_close = valid['predicted_close']

    mae_close = (actual_close - pred_close).abs().mean()

    # Porcentaje de acierto en dirección del CIERRE
    actual_return = (actual_close / valid['last_close']) - 1
    pred_return = valid['predicted_return']

    dir_hit_rate = ((actual_return > 0) == (pred_return > 0)).mean()

    print("\n📊 HISTÓRICO DE PREDICCIONES (CIERRE):")
    print(f"Registros con precio real disponible: {len(valid)}")
    print(f"MAE histórico en cierre: ${mae_close:.2f}")
    print(f"Porcentaje de acierto en dirección: {dir_hit_rate*100:.2f}%")

def run_project():
    # 1. Configuración: Ahora incluimos SPY como referencia de mercado
    SYMBOLS = ["NVDA", "SPY"]
    START_DATE = datetime.now() - timedelta(days=730) 

    # 2. Carga de datos (ahora recibe una lista)
    loader = DataLoader()
    df_raw = loader.get_stock_data(SYMBOLS, START_DATE)

    # 3. Procesamiento (incluye el refuerzo de SPY y Bollinger)
    processor = DataProcessor(df_raw)
    df_enriched = processor.add_technical_indicators()

    # 4. Modelado
    ai_model = StockModel(df_enriched)
    ai_model.prepare_data()
    ai_model.train()
    
    # --- NUEVO: Ver qué variables son las más importantes ---
    print("\n🔍 ANALIZANDO IMPORTANCIA DE LAS VARIABLES:")
    ai_model.show_importance()
    # --------------------------------------------------------

    # 5. Evaluación
    ai_model.evaluate(df_enriched['close'].iloc[-1])

    # 6. Predicción para mañana
    last_close = df_enriched['close'].iloc[-1]
    prediction = ai_model.predict_tomorrow(last_close)

    predicted_price = prediction["predicted_price"]
    predicted_return = prediction["predicted_return"] * 100
    predicted_open = prediction["predicted_open_price"]
    predicted_open_return = prediction["predicted_open_return"] * 100
    prob_up = prediction["prob_up"] * 100
    direction = prediction["direction"]
    action = prediction["action"]

    last_ts = pd.to_datetime(df_enriched['timestamp'].iloc[-1])

    print("\n📌 PREDICCIÓN PARA EL PRÓXIMO CIERRE DE NVDA")
    print(f"Fecha último dato (cierre conocido): {last_ts.date()}")
    print(f"Precio actual (cierre conocido): ${last_close:.2f}")
    print(f"Precio de cierre predicho: ${predicted_price:.2f}")
    print(f"Cambio esperado al cierre: {predicted_return:.2f}% ({direction})")

    print("\n📌 PREDICCIÓN PARA LA APERTURA DEL SIGUIENTE DÍA HÁBIL")
    print(f"Precio de apertura predicho: ${predicted_open:.2f}")
    print(f"Cambio esperado en apertura vs. último cierre: {predicted_open_return:.2f}%")

    print(f"Probabilidad de movimiento al alza (modelo de clasificación): {prob_up:.2f}%")
    print(f"Sugerencia de posición: {action} (modelo cuantitativo, no consejo financiero)")

    # Guardar predicción y evaluar histórico
    log_and_evaluate_predictions(df_enriched, prediction)

if __name__ == "__main__":
    run_project()