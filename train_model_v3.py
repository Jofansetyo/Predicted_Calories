"""
Script training model prediksi kalori terbakar menggunakan dataset baru
(Kaggle Playground Series S5E5 - 750.000 baris).

Melatih 2 model untuk dibandingkan:
- Random Forest (Machine Learning klasik, berbasis pohon keputusan)
- Neural Network / MLP Regressor (Deep Learning, berbasis jaringan saraf tiruan)

Perbandingan dilakukan berdasarkan MAE dan RMSE.

Cara pakai:
1. Pastikan dataset_baru/train.csv sudah ada
2. Jalankan: python train_model_v3.py
3. Model terbaik otomatis disimpan ke model/calorie_model.pkl
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 1. Load dataset
data = pd.read_csv("dataset_baru/train.csv")

# 2. Ubah kolom Sex jadi angka (0 = female, 1 = male)
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

# 3. Pilih fitur dan target
fitur = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
X = data[fitur]
y = data["Calories"]

# 4. Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

hasil_perbandingan = {}

# 5. Latih Random Forest (Machine Learning klasik)
print("Melatih Random Forest...")
t0 = time.time()
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
rf_time = time.time() - t0

rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
hasil_perbandingan["Random Forest"] = {"mae": rf_mae, "rmse": rf_rmse, "waktu": rf_time}
print(f"Random Forest -> MAE: {rf_mae:.3f} | RMSE: {rf_rmse:.3f} | waktu: {rf_time:.1f}s")

# 6. Latih Neural Network / MLP (Deep Learning)
# Neural network butuh fitur di-scaling dulu (StandardScaler) biar training stabil,
# makanya dibungkus dalam Pipeline supaya scaler ikut tersimpan bersama modelnya.
print("Melatih Neural Network (MLP)...")
t0 = time.time()
nn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
    )),
])
nn_model.fit(X_train, y_train)
nn_time = time.time() - t0

nn_pred = nn_model.predict(X_test)
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
hasil_perbandingan["Neural Network (MLP)"] = {"mae": nn_mae, "rmse": nn_rmse, "waktu": nn_time}
print(f"Neural Network (MLP) -> MAE: {nn_mae:.3f} | RMSE: {nn_rmse:.3f} | waktu: {nn_time:.1f}s")

# 7. Ringkasan dan pemilihan model terbaik
print("\n=== Ringkasan perbandingan ===")
for nama, hasil in hasil_perbandingan.items():
    print(f"{nama}: MAE={hasil['mae']:.3f}, RMSE={hasil['rmse']:.3f}, waktu latih={hasil['waktu']:.1f}s")

model_terbaik_nama = min(hasil_perbandingan, key=lambda k: hasil_perbandingan[k]["mae"])
model_terbaik = rf_model if model_terbaik_nama == "Random Forest" else nn_model

print(f"\nModel terbaik: {model_terbaik_nama}")

# 8. Simpan model terbaik
joblib.dump(model_terbaik, "model/calorie_model.pkl")
print("Model tersimpan di model/calorie_model.pkl")
