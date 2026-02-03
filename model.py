import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

class StockModel:
    def __init__(self, df):
        self.df = df
        # Modelo regresor para el retorno porcentual del próximo día
        self.reg_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        # Modelo clasificador para dirección (sube/baja)
        self.dir_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

    def prepare_data(self):
        # Añadimos los nuevos refuerzos a la lista de pistas
        features = [
            'lag_1', 'lag_2', 'rsi', 'volatility', 
            'spy_returns', 'b_position'
        ]
        self.features = features

        # Usamos solo filas donde el target está definido para entrenar
        df_model = self.df.dropna(subset=['target_return']).copy()

        self.X = df_model[features]

        # Variable objetivo para regresión: retorno porcentual siguiente día
        self.y = df_model['target_return']

        # Variable objetivo para clasificación: 1 si sube, 0 si baja o se mantiene
        self.y_dir = (df_model['target_return'] > 0).astype(int)

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_dir_train,
            self.y_dir_test,
        ) = train_test_split(
            self.X,
            self.y,
            self.y_dir,
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

    def evaluate(self, current_price):
        # 1. Predecir el % de cambio
        pred_returns = self.reg_model.predict(self.X_test)
        
        # 2. Convertir esos % de vuelta a dólares para que tenga sentido
        # Usamos los precios reales del set de prueba para la conversión
        test_prices = self.df['close'].iloc[-len(self.y_test):].values
        real_future_prices = test_prices * (1 + self.y_test.values)
        pred_future_prices = test_prices * (1 + pred_returns)

        mae = mean_absolute_error(real_future_prices, pred_future_prices)
        print(f"\nNUEVO MAE REFINADO (precio): ${mae:.2f}")

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
        pred_return = self.reg_model.predict(latest_features)[0]

        # Probabilidad de que el movimiento sea al alza según el clasificador
        prob_up = self.dir_model.predict_proba(latest_features)[0][1]

        predicted_price = last_close * (1 + pred_return)
        direction = "al alza" if pred_return > 0 else "a la baja"

        # Recomendación simple basada en señal y confianza
        if prob_up >= 0.6 and pred_return > 0:
            action = "LONG"
        elif prob_up <= 0.4 and pred_return < 0:
            action = "SHORT"
        else:
            action = "NEUTRAL"

        return {
            "predicted_price": float(predicted_price),
            "predicted_return": float(pred_return),
            "prob_up": float(prob_up),
            "direction": direction,
            "action": action,
        }