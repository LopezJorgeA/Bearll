import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df):
        # Al pivotar en el loader, 'close' es el precio de NVDA
        # y 'close_SPY' es el del índice.
        self.df = df.copy()

    def add_technical_indicators(self):
        print("🛠 Reforzando el set de datos con memoria técnica...")

        # --- EL ARREGLO ESTÁ AQUÍ ---
        # Primero calculamos el retorno de la acción principal (NVDA)
        self.df['returns'] = self.df['close'].pct_change()
        # ----------------------------

        # 1. Lags (¿Qué pasó hace 1, 2 y 3 días?)
        for i in range(1, 4):
            self.df[f'lag_{i}'] = self.df['returns'].shift(i)

        # 2. Indicadores Clásicos
        self.df['sma_10'] = self.df['close'].rolling(window=10).mean()
        self.df['rsi'] = self._calculate_rsi(self.df['close'])
        
        # 3. Volatilidad Relativa
        self.df['volatility'] = self.df['returns'].rolling(window=10).std()

        # 4. Correlación con el Mercado (SPY)
        # Calculamos el retorno del SPY para que el modelo vea el contexto
        self.df['spy_returns'] = self.df['close_SPY'].pct_change()
        
        # 5. Bandas de Bollinger
        sma_20 = self.df['close'].rolling(window=20).mean()
        std_20 = self.df['close'].rolling(window=20).std()
        self.df['bollinger_upper'] = sma_20 + (std_20 * 2)
        self.df['bollinger_lower'] = sma_20 - (std_20 * 2)
        
        # Posición relativa dentro de las bandas (0 abajo, 1 arriba)
        self.df['b_position'] = (self.df['close'] - self.df['bollinger_lower']) / \
                                (self.df['bollinger_upper'] - self.df['bollinger_lower'])

        # 6. Definir el Target (Lo que queremos predecir mañana)
        self.df['target_return'] = self.df['returns'].shift(-1)

        # Limpieza de valores nulos SOLO en las features, no en el target
        feature_cols = ['returns']
        feature_cols += [f'lag_{i}' for i in range(1, 4)]
        feature_cols += [
            'sma_10', 'rsi', 'volatility', 'spy_returns',
            'bollinger_upper', 'bollinger_lower', 'b_position'
        ]

        # Eliminamos filas donde falte alguna feature técnica
        self.df.dropna(subset=feature_cols, inplace=True)
        
        return self.df

    def _calculate_rsi(self, series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))