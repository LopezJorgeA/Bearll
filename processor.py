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

        # 4. Correlación con el Mercado (QQQ / SOXX)
        # Calculamos los retornos de índices más relacionados con NVDA
        if 'close_QQQ' in self.df.columns:
            self.df['qqq_returns'] = self.df['close_QQQ'].pct_change()
        else:
            self.df['qqq_returns'] = np.nan

        if 'close_SOXX' in self.df.columns:
            self.df['soxx_returns'] = self.df['close_SOXX'].pct_change()
        else:
            self.df['soxx_returns'] = np.nan
        
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

        # Target de apertura del día siguiente (open_{t+1} respecto al close_t)
        self.df['target_open_return'] = (self.df['open'].shift(-1) / self.df['close']) - 1

        # 7. ATR (Average True Range) - Volatilidad real
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['atr_14'] = true_range.rolling(14).mean()

        # 8. VWAP relativo (precio vs precio medio ponderado por volumen)
        if 'vwap' in self.df.columns:
            self.df['dist_vwap'] = (self.df['close'] - self.df['vwap']) / self.df['vwap']
        else:
            self.df['dist_vwap'] = np.nan

        # 9. RSI de Volumen simple (cambio relativo de volumen)
        self.df['vol_change'] = self.df['volume'].pct_change()

        # 10. Distancia a SMA20 (breakouts)
        sma20 = self.df['close'].rolling(20).mean()
        self.df['dist_sma20'] = (self.df['close'] - sma20) / sma20

        # 11. Momentum (Rate of Change a 5 días)
        self.df['roc_5'] = self.df['close'].pct_change(5)

        # 12. ADX (Average Directional Index) - fuerza de la tendencia
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=self.df.index)
        minus_dm = pd.Series(minus_dm, index=self.df.index)

        tr14 = true_range.rolling(14).mean()
        plus_di14 = 100 * plus_dm.rolling(14).mean() / tr14
        minus_di14 = 100 * minus_dm.rolling(14).mean() / tr14

        dx = 100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14)
        self.df['adx_14'] = dx.rolling(14).mean()

        # 13. Régimen de volatilidad (bajo/medio/alto) basado en 'volatility'
        vol = self.df['volatility']
        q_low = vol.quantile(0.33)
        q_high = vol.quantile(0.66)
        self.df['regime_vol'] = np.select(
            [vol <= q_low, (vol > q_low) & (vol <= q_high), vol > q_high],
            [0, 1, 2],
            default=1,
        )

        # Limpieza de valores nulos SOLO en las features, no en los targets
        feature_cols = ['returns']
        feature_cols += [f'lag_{i}' for i in range(1, 4)]
        feature_cols += ['sma_10', 'rsi', 'volatility']

        # Solo usamos estos retornos de índices si realmente hay datos
        if self.df['qqq_returns'].notna().any():
            feature_cols.append('qqq_returns')
        if self.df['soxx_returns'].notna().any():
            feature_cols.append('soxx_returns')

        feature_cols += [
            'bollinger_upper', 'bollinger_lower', 'b_position',
            'atr_14', 'dist_vwap', 'vol_change',
            'dist_sma20', 'roc_5', 'adx_14', 'regime_vol',
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