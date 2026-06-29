
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

modelo = load_model("modelo.keras")

scaler = joblib.load("scaler.pkl")

df = pd.read_csv("data_inference.csv")

X = df.drop(columns=["SalePrice","Precio_Alto"])

X = pd.get_dummies(X, drop_first=True)

X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

X = scaler.transform(X)

probabilidades = modelo.predict(X)

predicciones = pd.DataFrame({

    "ID": df.index,

    "Probabilidad": probabilidades.flatten(),

    "Prediccion": (probabilidades > 0.5).astype(int).flatten()

})

predicciones.to_csv("predictions.csv", index=False)

print("Predicciones generadas correctamente.")
