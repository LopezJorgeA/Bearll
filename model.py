import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier

class StockModel:
    def __init__(self, df):
        self.df = df
        # Modelo regresor para el retorno porcentual del próximo día (cierre)
        self.reg_model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

        # Modelo regresor para la apertura del día siguiente
        self.open_reg_model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )

        # Modelo clasificador para dirección (sube/baja)
        self.dir_model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    def prepare_data(self):
        # Añadimos los nuevos refuerzos a la lista de pistas
        features = [
            'lag_1', 'lag_2', 'rsi', 'volatility',
            'qqq_returns', 'soxx_returns',
            'atr_14', 'dist_vwap', 'vol_change',
            'dist_sma20', 'roc_5', 'adx_14',
            'regime_vol', 'b_position',
        ]
        self.features = features

        # Usamos solo filas donde los targets están definidos para entrenar
        df_model = self.df.dropna(subset=['target_return', 'target_open_return']).copy()
        self.df_model = df_model

        self.X = df_model[features]

        # Variable objetivo para regresión: retorno porcentual siguiente día (cierre)
        self.y = df_model['target_return']

        # Variable objetivo para regresión: retorno de apertura del día siguiente
        self.y_open = df_model['target_open_return']

        # Variable objetivo para clasificación: 1 si sube el cierre, 0 si baja o se mantiene
        self.y_dir = (df_model['target_return'] > 0).astype(int)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_dir_train,
            self.y_dir_test,
            self.y_open_train,
            self.y_open_test,
        ) = train_test_split(
            self.X,
            self.y,
            self.y_dir,
            self.y_open,
            test_size=0.2,
            shuffle=False,
        )

    def show_importance(self):
        """Muestra qué datos están ayudando más a la predicción."""
        importances = self.reg_model.feature_importances_
        for i, feat in enumerate(self.X.columns):
            print(f"Feature: {feat}, Score: {importances[i]:.4f}")

    def train(self):
        print("Entrenando modelo de regresión (retornos porcentuales)...")
        self.reg_model.fit(self.X_train, self.y_train)

        print("Entrenando modelo de clasificación (dirección del movimiento)...")
        self.dir_model.fit(self.X_train, self.y_dir_train)

        print("Entrenando modelo de regresión (apertura del día siguiente)...")
        self.open_reg_model.fit(self.X_train, self.y_open_train)

    def evaluate(self, current_price):
        # 1. Predecir el % de cambio
        pred_returns = self.reg_model.predict(self.X_test)
        
        # 2. Convertir esos % de vuelta a dólares para que tenga sentido
        # Usamos los precios reales del set de prueba para la conversión
        df_eval = self.df_model.iloc[-len(self.y_test):]
        test_prices = df_eval['close'].values
        real_future_prices = test_prices * (1 + self.y_test.values)
        pred_future_prices = test_prices * (1 + pred_returns)

        mae = mean_absolute_error(real_future_prices, pred_future_prices)
        print(f"\nNUEVO MAE REFINADO (precio de cierre): ${mae:.2f}")

        # Métrica para la apertura del día siguiente
        pred_open_returns = self.open_reg_model.predict(self.X_test)
        real_open_prices = test_prices * (1 + self.y_open_test.values)
        pred_open_prices = test_prices * (1 + pred_open_returns)

        mae_open = mean_absolute_error(real_open_prices, pred_open_prices)
        print(f"MAE en precio de apertura: ${mae_open:.2f}")

        # Métrica de dirección: ¿acierta si sube o baja?
        dir_pred_test = self.dir_model.predict(self.X_test)
        dir_accuracy = accuracy_score(self.y_dir_test, dir_pred_test)
        print(f"Precisión en dirección (sube/baja): {dir_accuracy*100:.2f}%")

        self._plot_results(real_future_prices, pred_future_prices)
        return mae, dir_accuracy

    def _plot_results(self, real, pred):
        plt.figure(figsize=(10, 5))
        plt.plot(real, label="Precio Real", color="#1f77b4")
        plt.plot(pred, label="Predicción (Basada en Retornos)", color="#ff7f0e", linestyle="--")
        plt.title("NVIDIA: Refinamiento por Variación Porcentual")
        plt.legend()
        plt.show()

    def predict_tomorrow(self, last_close):
        # Usamos las features de la ÚLTIMA fila disponible, aunque su target sea NaN
        latest_features = self.df[self.features].tail(1)

        # Predicción de retorno para el cierre del próximo día
        pred_return = self.reg_model.predict(latest_features)[0]
        predicted_price = last_close * (1 + pred_return)

        # Predicción de retorno para la apertura del próximo día
        pred_open_return = self.open_reg_model.predict(latest_features)[0]
        predicted_open_price = last_close * (1 + pred_open_return)

        # Probabilidad de que el movimiento sea al alza según el clasificador
        prob_up = self.dir_model.predict_proba(latest_features)[0][1]
        direction = "al alza" if pred_return > 0 else "a la baja"

        # Umbrales dinámicos según el régimen de volatilidad y ATR
        latest_row = self.df.tail(1).iloc[0]
        atr_val = latest_row.get('atr_14', 0)
        vol_val = latest_row.get('volatility', 0)

        atr_med = self.df['atr_14'].median()
        vol_med = self.df['volatility'].median()

        high_vol = (atr_val >= atr_med) or (vol_val >= vol_med)

        if high_vol:
            # Mercado más "caliente": umbrales más agresivos
            up_threshold = 0.55
            down_threshold = 0.45
        else:
            # Mercado tranquilo: pedimos más confianza
            up_threshold = 0.65
            down_threshold = 0.35

        # Recomendación basada en señal, confianza y volatilidad
        if prob_up >= up_threshold and pred_return > 0:
            action = "LONG"
        elif prob_up <= down_threshold and pred_return < 0:
            action = "SHORT"
        else:
            action = "NEUTRAL"

        return {
            "predicted_price": float(predicted_price),
            "predicted_return": float(pred_return),
            "predicted_open_price": float(predicted_open_price),
            "predicted_open_return": float(pred_open_return),
            "prob_up": float(prob_up),
            "direction": direction,
            "action": action,
        }