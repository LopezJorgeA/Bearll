from data_loader import DataLoader
from processor import DataProcessor
from model import StockModel
from datetime import datetime, timedelta

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
    prob_up = prediction["prob_up"] * 100
    direction = prediction["direction"]
    action = prediction["action"]

    print("\n📌 PREDICCIÓN PARA EL PRÓXIMO CIERRE DE NVDA")
    print(f"Precio actual: ${last_close:.2f}")
    print(f"Precio predicho: ${predicted_price:.2f}")
    print(f"Cambio esperado: {predicted_return:.2f}% ({direction})")
    print(f"Probabilidad de movimiento al alza (modelo de clasificación): {prob_up:.2f}%")
    print(f"Sugerencia de posición: {action} (modelo cuantitativo, no consejo financiero)")

if __name__ == "__main__":
    run_project()