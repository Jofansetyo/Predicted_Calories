# Hasil Perbandingan Model - Prediksi Kalori Terbakar

## Dataset
- Sumber: Kaggle Playground Series S5E5 - "Predict Calorie Expenditure"
- Jumlah data: 750.000 baris
- Fitur: Sex, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
- Target: Calories
- Tidak ada missing value
- Split data: 80% latih (600.000 baris) / 20% uji (150.000 baris)

## Model yang Dibandingkan

1. **Random Forest** — model machine learning klasik berbasis ensemble pohon
   keputusan (bagging).
2. **Neural Network / Multi-Layer Perceptron (MLP)** — model deep learning
   berbasis jaringan saraf tiruan, cocok untuk data tabular/numerik seperti
   dataset ini. Fitur discaling terlebih dahulu menggunakan StandardScaler
   sebelum masuk ke jaringan saraf.

Catatan: CNN (Convolutional Neural Network) dan Transfer Learning tidak
digunakan karena kedua metode tersebut dirancang untuk data yang memiliki
struktur spasial seperti gambar, sedangkan dataset ini berupa data
tabular/numerik biasa sehingga MLP adalah pendekatan deep learning yang
lebih sesuai.

## Perbandingan Hasil

| Model                  | MAE   | RMSE  | Waktu Training |
|-------------------------|-------|-------|----------------|
| Random Forest           | 2.592 | 4.122 | 97.1 detik     |
| Neural Network (MLP)     | 2.186 | 3.644 | 12.0 detik     |

## Arsitektur Neural Network

- Hidden layer: 2 layer (64 neuron, 32 neuron)
- Fungsi aktivasi: ReLU
- Optimizer: Adam (default scikit-learn)
- Early stopping: aktif, untuk mencegah overfitting dan mempercepat training
- Preprocessing: StandardScaler pada seluruh fitur numerik

## Analisis

- **Neural Network (MLP) lebih unggul** di kedua metrik error (MAE dan RMSE)
  dibanding Random Forest, dan juga lebih cepat dilatih karena early stopping
  menghentikan training begitu performa validasi tidak lagi meningkat.
- Random Forest tetap kompetitif dan lebih mudah diinterpretasi (feature
  importance), namun performa prediksinya sedikit di bawah MLP pada dataset
  berukuran besar ini.
- Parameter Random Forest: n_estimators=100, max_depth=12
- Parameter Neural Network: hidden_layer_sizes=(64, 32), activation=relu,
  early_stopping=True

## Kesimpulan

Model yang dipilih untuk digunakan dalam sistem adalah **Neural Network
(MLP)**, karena memberikan akurasi prediksi terbaik dengan waktu komputasi
yang efisien, sehingga sesuai untuk implementasi pada sistem berbasis web
yang membutuhkan respons cepat.
