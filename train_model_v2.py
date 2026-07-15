"""
Script training model prediksi kalori terbakar menggunakan dataset baru
(Kaggle Playground Series S5E5 - 750.000 baris).

Melatih 2 model (Random Forest dan XGBoost) lalu membandingkan performanya
berdasarkan Mean Absolute Error (MAE) dan RMSE.

Cara pakai:
1. Taruh train.csv dari Playground Series S5E5 di folder yang sama
2. Jalankan: python train_model_v2.py
3. Model terbaik otomatis disimpan ke model/calorie_model.pkl
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
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

# 5. Latih Random Forest
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

# 6. Latih XGBoost
print("Melatih XGBoost...")
t0 = time.time()
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - t0

xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
hasil_perbandingan["XGBoost"] = {"mae": xgb_mae, "rmse": xgb_rmse, "waktu": xgb_time}
print(f"XGBoost -> MAE: {xgb_mae:.3f} | RMSE: {xgb_rmse:.3f} | waktu: {xgb_time:.1f}s")

# 7. Pilih model terbaik berdasarkan MAE terkecil
print("\n=== Ringkasan perbandingan ===")
for nama, hasil in hasil_perbandingan.items():
    print(f"{nama}: MAE={hasil['mae']:.3f}, RMSE={hasil['rmse']:.3f}, waktu latih={hasil['waktu']:.1f}s")

model_terbaik_nama = min(hasil_perbandingan, key=lambda k: hasil_perbandingan[k]["mae"])
model_terbaik = rf_model if model_terbaik_nama == "Random Forest" else xgb_model

print(f"\nModel terbaik: {model_terbaik_nama}")

# 8. Simpan model terbaik
joblib.dump(model_terbaik, "model/calorie_model.pkl")
print("Model tersimpan di model/calorie_model.pkl")
