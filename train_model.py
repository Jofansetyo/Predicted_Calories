"""
Script untuk melatih model prediksi kalori terbakar.

Cara pakai:
1. Download dataset dari Kaggle: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos
   -> ada 2 file: exercise.csv dan calories.csv
2. Taruh kedua file itu di folder yang sama dengan script ini
3. Jalankan: python train_model.py
4. Model akan disimpan di model/calorie_model.pkl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load dataset
exercise = pd.read_csv("exercise.csv")
calories = pd.read_csv("calories.csv")

# gabungkan berdasarkan User_ID
data = exercise.merge(calories, on="User_ID")

# 2. Ubah kolom Gender jadi angka (0 = female, 1 = male)
data["Gender"] = data["Gender"].map({"male": 1, "female": 0})

# 3. Pilih fitur (input) dan target (output)
fitur = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
X = data[fitur]
y = data["Calories"]

# 4. Split data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Latih model Random Forest
# n_estimators & max_depth dibatasi biar ukuran file model gak terlalu besar
model = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluasi model
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print(f"Mean Absolute Error: {mae:.2f} kalori")

# 7. Simpan model ke file .pkl
joblib.dump(model, "model/calorie_model.pkl")
print("Model berhasil disimpan di model/calorie_model.pkl")
