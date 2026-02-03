import os
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# Cargar las variables del archivo .env
load_dotenv()

class DataLoader:
    def __init__(self):
        # Obtenemos las credenciales del archivo .env
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Inicializamos el cliente de Alpaca
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def get_stock_data(self, symbols, start_date):
        # symbols ahora puede ser una lista como ["NVDA", "QQQ", "SOXX"]
        print(f"Descargando datos para {symbols}...")
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start_date
        )

        bars = self.client.get_stock_bars(request_params)
        df = bars.df.reset_index()

        # Pivotamos para tener cierres de todos los símbolos como columnas
        # Por ejemplo: close_NVDA, close_QQQ, close_SOXX
        df_pivot = df.pivot(index='timestamp', columns='symbol', values='close')
        df_pivot.columns = [f'close_{col}' for col in df_pivot.columns]

        # También necesitamos los otros datos de NVDA (Open, High, Low, Volume, VWAP...)
        nvda_raw = df[df['symbol'] == 'NVDA'].set_index('timestamp')

        # Eliminamos la columna de cierre NVDA del pivot (ya la tenemos en nvda_raw)
        for col in list(df_pivot.columns):
            if col.lower() in ("close_nvda",):
                df_pivot.drop(columns=[col], inplace=True)

        # Combinamos todo: NVDA + referencias de mercado (QQQ, SOXX, etc.)
        final_df = nvda_raw.join(df_pivot)
        return final_df.reset_index()

# --- Bloque de prueba ---
if __name__ == "__main__":
    # Esto es solo para probar que este archivo funciona solo
    loader = DataLoader()
    # Probamos con NVIDIA y referencias de mercado desde hace 2 años (puedes cambiar la fecha)
    test_date = datetime(2024, 1, 1)
    data = loader.get_stock_data(["NVDA", "QQQ", "SOXX"], test_date)
    
    print("\nDatos descargados con éxito:")
    print(data.head()) # Muestra las primeras 5 filas
    print(f"\nTotal de días recolectados: {len(data)}")