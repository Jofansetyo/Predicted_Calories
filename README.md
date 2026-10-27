# KaloriKu — Sistem Prediksi Kalori Terbakar Berbasis Machine Learning

KaloriKu adalah aplikasi web yang memprediksi jumlah kalori yang terbakar saat
olahraga berdasarkan biodata dan data aktivitas fisik pengguna, menggunakan
model machine learning yang dilatih dari 750.000 data aktivitas nyata.

---

## 1. Bahasa Pemrograman

- **Python 3** — backend, machine learning, dan logika aplikasi
- **HTML5** — struktur halaman (template Jinja2)
- **CSS3** — styling dan tema (dark/light mode)
- **JavaScript (Vanilla)** — interaktivitas di sisi klien (tanpa framework seperti React/Vue)

Aplikasi ini **tidak menggunakan PHP/Laravel** atau framework backend lain di
luar Python — murni native Flask + HTML/CSS/JS.

---

## 2. Framework, Library, API, dan Tools

| Kategori | Nama | Kegunaan |
|---|---|---|
| Web Framework | [Flask](https://flask.palletsprojects.com/) | Routing, backend server, render halaman |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) | Random Forest Regressor, MLP Regressor, StandardScaler, train_test_split |
| Data Processing | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | Manipulasi dan preprocessing data |
| Serialisasi Model | [joblib](https://joblib.readthedocs.io/) | Menyimpan dan memuat model `.pkl` |
| Visualisasi (notebook) | [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) | Exploratory Data Analysis |
| Grafik (opsional) | [Chart.js](https://www.chartjs.org/) (CDN) | Visualisasi data di sisi klien |
| Template Engine | Jinja2 (bawaan Flask) | Templating halaman HTML |
| Tipografi | Google Fonts (Big Shoulders Display, Inter, JetBrains Mono) | Desain antarmuka |
| API Eksternal | [Kaggle API](https://www.kaggle.com/docs/api) | Mengunduh dataset training |

---

## 3. Fungsi dan Fitur Proyek

1. **Prediksi Kalori Terbakar** — pengguna mengisi biodata (jenis kelamin, usia,
   tinggi, berat) dan data aktivitas (durasi olahraga, detak jantung, suhu tubuh),
   sistem memprediksi estimasi kalori yang terbakar.
2. **Analisis BMI otomatis** — setiap hasil prediksi disertai perhitungan Body
   Mass Index dan kategori status gizi (Kurus/Normal/Gemuk/Obesitas).
3. **Saran kesehatan** — rekomendasi otomatis berdasarkan kategori BMI pengguna.
4. **Rekomendasi jenis olahraga** — pengguna memasukkan target kalori yang ingin
   dibakar, sistem menyarankan jenis olahraga dan estimasi durasinya.
5. **Dark/Light mode** — tema tampilan bisa diganti dan tersimpan di browser.
6. **Halaman "Tentang"** — penjelasan pipeline model, metrik performa, dan
   variabel input yang digunakan.

---

## 4. Kelebihan Proyek

- Model dilatih dengan dataset besar (750.000 baris), bukan dataset kecil,
  sehingga generalisasi model lebih baik.
- Dua algoritma dibandingkan secara objektif (Random Forest vs Neural Network)
  dengan metrik MAE, RMSE, dan waktu training, bukan asal pilih satu model.
- Antarmuka didesain khusus (custom UI/UX), bukan template generik — termasuk
  ilustrasi SVG orisinal, bukan gambar hasil unduhan/stok.
- Kode terstruktur rapi: pemisahan jelas antara `templates/`, `static/`, dan
  logika backend di `app.py`.
- Tidak bergantung pada framework frontend berat (React/Vue), sehingga ringan
  dan mudah dipahami untuk keperluan pembelajaran.
- Notebook training disertakan lengkap dengan tahapan EDA, sehingga proses
  pelatihan model transparan dan dapat direproduksi ulang.

---

## 5. Batasan Proyek

- **Tidak ada sistem autentikasi/login** — aplikasi ini tidak menyimpan riwayat
  prediksi per pengguna. Setiap prediksi bersifat sesi-tunggal (tidak
  persisten setelah halaman ditutup).
- **Model belum divalidasi lintas populasi Indonesia** — dataset training
  berasal dari data internasional (Kaggle), sehingga akurasi untuk karakteristik
  fisik populasi Indonesia belum diuji secara khusus.
- **Validasi input membatasi nilai ke angka positif**, tetapi belum membatasi
  rentang fisiologis yang wajar. Input di luar rentang data training dapat
  menghasilkan estimasi yang tidak andal.
- **Model disimpan sebagai file `.pkl`** yang bergantung pada versi scikit-learn
  saat training; model kemungkinan gagal dimuat jika dijalankan dengan versi
  scikit-learn yang jauh berbeda dari saat training.

---

## 6. Dataset

**Sumber:** Kaggle Playground Series — Season 5, Episode 5: "Predict Calorie Expenditure"
**Link:** https://www.kaggle.com/competitions/playground-series-s5e5/data

---

## 7. Penjelasan Dataset

Dataset ini berisi data sintetis yang dihasilkan dari model yang dilatih pada
dataset asli **Calorie Burn Prediction** (fmendes-DAT263x-demos), dengan
distribusi statistik yang serupa namun jumlah data yang jauh lebih besar.

| Kolom | Tipe | Keterangan |
|---|---|---|
| `id` | integer | ID unik baris data |
| `Sex` | kategorikal | Jenis kelamin (`male` / `female`) |
| `Age` | numerik | Usia (tahun) |
| `Height` | numerik | Tinggi badan (cm) |
| `Weight` | numerik | Berat badan (kg) |
| `Duration` | numerik | Durasi olahraga (menit) |
| `Heart_Rate` | numerik | Detak jantung selama beraktivitas (bpm) |
| `Body_Temp` | numerik | Suhu tubuh selama beraktivitas (°C) |
| `Calories` | numerik (target) | Jumlah kalori yang terbakar (kkal) |

**Statistik dataset:**
- Jumlah data latih (`train.csv`): 750.000 baris
- Tidak ada missing value maupun kolom kosong
- Distribusi gender seimbang (±50/50)
- Split yang digunakan pada training: 80% data latih (600.000 baris), 20% data uji (150.000 baris)

---

## Arsitektur & Pipeline Model

```
Dataset (750.000 baris)
        │
        ▼
Preprocessing (encoding Sex, split data 80/20)
        │
        ▼
┌───────────────────┬──────────────────────┐
│  Random Forest     │  Neural Network (MLP) │
│  n_estimators=100  │  hidden_layer=(64,32)  │
│  max_depth=12       │  activation=relu       │
└───────────────────┴──────────────────────┘
        │
        ▼
   Evaluasi (MAE, RMSE)
        │
        ▼
  Model terbaik disimpan (calorie_model.pkl)
        │
        ▼
   Aplikasi Web Flask (prediksi real-time)
```

### Hasil Perbandingan Model

| Model | MAE | RMSE | Waktu Training |
|---|---|---|---|
| Random Forest | 2.592 | 4.122 | 97.1 detik |
| **Neural Network (MLP)** | **2.186** | **3.644** | **12.0 detik** |

Model **Neural Network (MLP)** dipilih sebagai model final karena memberikan
akurasi lebih tinggi dan waktu training lebih efisien. Detail lengkap ada di
[`HASIL_PERBANDINGAN_MODEL.md`](HASIL_PERBANDINGAN_MODEL.md).

Catatan: CNN dan Transfer Learning tidak digunakan karena kedua pendekatan
tersebut dirancang untuk data bertipe gambar (memiliki struktur spasial),
sedangkan dataset ini berupa data tabular/numerik biasa, sehingga MLP adalah
pendekatan deep learning yang lebih sesuai secara konsep.

---

## Struktur Proyek

```
kalori-prediksi/
├── app.py                          # Aplikasi Flask (routing & logika backend)
├── requirements.txt                # Daftar dependency Python
├── model/
│   └── calorie_model.pkl           # Model terlatih (Neural Network/MLP)
├── notebook/
│   └── Training_Model_Kalori.ipynb # Notebook pelatihan model (EDA + training + evaluasi)
├── templates/                      # Template HTML (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── rekomendasi.html
│   └── tentang.html
├── static/
│   ├── style.css
│   └── script.js
├── dataset_baru/                   # Dataset training (lihat catatan di bawah)
├── train_model_v3.py               # Script training (versi command-line dari notebook)
└── HASIL_PERBANDINGAN_MODEL.md     # Dokumentasi hasil perbandingan model
```

> **Catatan:** Folder `dataset_baru/` berisi file CSV berukuran besar (puluhan
> MB). Jika mengunggah ke GitHub, disarankan menambahkan folder ini ke
> `.gitignore` dan cukup mencantumkan link dataset di README ini, karena
> GitHub tidak ideal untuk menyimpan file data besar.

---

## Cara Menjalankan

### Windows (PowerShell)

Buka terminal di folder proyek, lalu jalankan:

```bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

Jika folder `venv` belum ada, buat virtual environment terlebih dahulu:

```bash
py -3 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

Buka `http://127.0.0.1:5000` di browser. Model terlatih sudah disediakan di
`model/calorie_model.pkl`, jadi training tidak diperlukan untuk mencoba aplikasi.
Di VS Code, pilih konfigurasi **Python: Flask** pada panel Run and Debug lalu
tekan **F5**.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Untuk melatih ulang model, pastikan `dataset_baru/train.csv` tersedia, lalu
jalankan `python train_model_v3.py`. Proses ini melatih ulang model dan
menimpa `model/calorie_model.pkl`.

---

## Lisensi

Proyek ini dibuat untuk keperluan tugas akademik/edukasi.
