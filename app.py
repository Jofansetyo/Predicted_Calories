from flask import Flask, request, jsonify, render_template
import joblib
import math
import os
import pandas as pd

app = Flask(__name__)

# Load model sekali saat server nyala
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "calorie_model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

FITUR_MODEL = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]


def parse_angka(data, fields):
    if not isinstance(data, dict):
        return None, "Body request harus berupa JSON object."

    try:
        values = {field: float(data[field]) for field in fields}
    except (KeyError, TypeError, ValueError):
        return None, "Semua input wajib diisi dengan angka yang valid."

    if not all(math.isfinite(value) for value in values.values()):
        return None, "Input harus berupa angka yang valid."
    if any(value <= 0 for value in values.values()):
        return None, "Semua nilai harus lebih besar dari 0."

    return values, None


def hitung_bmi(berat, tinggi_cm):
    tinggi_m = tinggi_cm / 100
    bmi = berat / (tinggi_m ** 2)
    return round(bmi, 1)


def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Kurus", "Berat badan kamu tergolong kurang. Perbanyak asupan nutrisi seimbang dan latihan kekuatan untuk membangun massa otot."
    elif bmi < 25:
        return "Normal", "Berat badan kamu tergolong ideal. Pertahankan pola makan seimbang dan rutin berolahraga."
    elif bmi < 30:
        return "Gemuk", "Berat badan kamu tergolong berlebih. Kombinasi olahraga kardio rutin dan pola makan seimbang bisa membantu menurunkannya secara bertahap."
    else:
        return "Obesitas", "Berat badan kamu tergolong obesitas. Disarankan berkonsultasi dengan tenaga medis atau ahli gizi untuk program yang aman dan sesuai kondisi kamu."


# Estimasi kalori terbakar per menit untuk berbagai jenis olahraga
# (perkiraan umum untuk orang dewasa dengan berat rata-rata ~70kg)
DAFTAR_OLAHRAGA = [
    {"nama": "Jalan santai", "kalori_per_menit": 4, "intensitas": "Ringan"},
    {"nama": "Bersepeda santai", "kalori_per_menit": 5, "intensitas": "Ringan"},
    {"nama": "Yoga", "kalori_per_menit": 3, "intensitas": "Ringan"},
    {"nama": "Jalan cepat", "kalori_per_menit": 6, "intensitas": "Sedang"},
    {"nama": "Bersepeda santai-cepat", "kalori_per_menit": 8, "intensitas": "Sedang"},
    {"nama": "Berenang santai", "kalori_per_menit": 8, "intensitas": "Sedang"},
    {"nama": "Badminton", "kalori_per_menit": 7, "intensitas": "Sedang"},
    {"nama": "Jogging", "kalori_per_menit": 10, "intensitas": "Berat"},
    {"nama": "Lari", "kalori_per_menit": 13, "intensitas": "Berat"},
    {"nama": "HIIT / Circuit training", "kalori_per_menit": 14, "intensitas": "Berat"},
    {"nama": "Berenang cepat", "kalori_per_menit": 12, "intensitas": "Berat"},
    {"nama": "Basket / Futsal", "kalori_per_menit": 11, "intensitas": "Berat"},
]


@app.route("/")
def beranda():
    return render_template("index.html", active="beranda")


@app.route("/prediksi")
def prediksi():
    return render_template("predict.html", active="prediksi")


@app.route("/tentang")
def tentang():
    return render_template("tentang.html", active="tentang")


@app.route("/rekomendasi")
def rekomendasi():
    return render_template("rekomendasi.html", active="rekomendasi")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "File model tidak ditemukan. Jalankan train_model_v3.py untuk membuat model."}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Body request harus berupa JSON object."}), 400

    gender = data.get("gender")
    if gender not in ("male", "female"):
        return jsonify({"error": "Jenis kelamin harus male atau female."}), 400

    values, error = parse_angka(
        data, ["age", "height", "weight", "duration", "heart_rate", "body_temp"]
    )
    if error:
        return jsonify({"error": error}), 400

    fitur = pd.DataFrame(
        [[
            1 if gender == "male" else 0,
            values["age"],
            values["height"],
            values["weight"],
            values["duration"],
            values["heart_rate"],
            values["body_temp"],
        ]],
        columns=FITUR_MODEL,
    )
    hasil = round(float(model.predict(fitur)[0]), 1)

    bmi = hitung_bmi(values["weight"], values["height"])
    label_bmi, saran = kategori_bmi(bmi)

    return jsonify({
        "calories": hasil,
        "bmi": bmi,
        "label_bmi": label_bmi,
        "saran": saran,
    })


@app.route("/rekomendasi-olahraga", methods=["POST"])
def rekomendasi_olahraga():
    data = request.get_json(silent=True)
    values, error = parse_angka(data, ["target_kalori"])
    if error:
        return jsonify({"error": error}), 400

    target_kalori = values["target_kalori"]

    hasil = []
    for olahraga in DAFTAR_OLAHRAGA:
        durasi_menit = max(1, round(target_kalori / olahraga["kalori_per_menit"]))
        hasil.append({
            "nama": olahraga["nama"],
            "intensitas": olahraga["intensitas"],
            "durasi_menit": durasi_menit,
        })

    return jsonify({"rekomendasi": hasil})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
