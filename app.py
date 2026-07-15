from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model sekali saat server nyala
MODEL_PATH = "model/calorie_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


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
        return jsonify({"error": "Model belum ditraining. Jalankan train_model.py dulu."}), 400

    data = request.get_json()

    gender = 1 if data.get("gender") == "male" else 0
    age = float(data.get("age"))
    height = float(data.get("height"))
    weight = float(data.get("weight"))
    duration = float(data.get("duration"))
    heart_rate = float(data.get("heart_rate"))
    body_temp = float(data.get("body_temp"))

    fitur = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
    hasil = round(float(model.predict(fitur)[0]), 1)

    bmi = hitung_bmi(weight, height)
    label_bmi, saran = kategori_bmi(bmi)

    return jsonify({
        "calories": hasil,
        "bmi": bmi,
        "label_bmi": label_bmi,
        "saran": saran,
    })


@app.route("/rekomendasi-olahraga", methods=["POST"])
def rekomendasi_olahraga():
    data = request.get_json()
    target_kalori = float(data.get("target_kalori"))

    hasil = []
    for olahraga in DAFTAR_OLAHRAGA:
        durasi_menit = round(target_kalori / olahraga["kalori_per_menit"])
        hasil.append({
            "nama": olahraga["nama"],
            "intensitas": olahraga["intensitas"],
            "durasi_menit": durasi_menit,
        })

    return jsonify({"rekomendasi": hasil})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
