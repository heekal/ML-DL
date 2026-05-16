# 🎓 UTS Machine Learning — Haikal Ali

> Repository tugas tengah semester mata kuliah Machine Learning.
> Berisi tiga notebook independen yang mencakup task Clustering, Regresi, dan Klasifikasi.

---

## 👤 Identitas

| Field  | Detail               |
|--------|----------------------|
| Nama   | Haikal Ali           |
| Kelas  | TK-46-GAB            |
| NIM    | 1103223071           |

---

## 📁 Struktur Repository
UTS-ML/
├── UTS_ML_Clustering.ipynb          # Segmentasi pelanggan dengan clustering </br>
├── UTS_ML_REGRESI.ipynb             # Prediksi tahun rilis lagu dengan regresi </br>
├── UTS_ML_TransactionDataset.ipynb  # Deteksi fraud dengan klasifikasi </br>
└── README.md </br>

---

## 📌 Gambaran Umum

Repository ini berisi tiga proyek machine learning yang diselesaikan sebagai bagian dari penilaian UTS. Setiap notebook bersifat mandiri dengan dataset, pipeline preprocessing, pelatihan model, dan evaluasi masing-masing.

---

## 🧪 Detail Proyek

### 1. 📊 Segmentasi Pelanggan — Clustering

**Tujuan:** Mengelompokkan pelanggan kartu kredit ke dalam segmen perilaku yang bermakna menggunakan unsupervised learning.

**Model yang Digunakan:**
| Model                    | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|--------------------------|-------------|-----------------|---------------------|
| **KMeans** ✅ Terbaik    | **0.2586**  | 1.4980          | **1795.89**         |
| Agglomerative Clustering | 0.2512      | 1.5663          | 1463.99             |
| DBSCAN                   | 0.2338      | **0.6567**      | 18.44               |

**Kesimpulan:** KMeans dipilih sebagai model terbaik berdasarkan Silhouette Score tertinggi dan Calinski-Harabasz tertinggi, menunjukkan cluster yang paling padat dan terpisah dengan baik.

---

### 2. 📈 Prediksi Tahun Rilis Lagu — Regresi

**Tujuan:** Memprediksi tahun rilis lagu berdasarkan 90 fitur audio numerik menggunakan dataset UCI Million Song Dataset (MSD).

**Model yang Digunakan:** LightGBM Regressor (LGBMRegressor) + 5-Fold Cross Validation

**Hasil Evaluasi:**
| Metrik | Nilai  |
|--------|--------|
| MSE    | 82.0993 |
| RMSE   | 9.0609  |
| MAE    | 6.3393  |
| R²     | 0.3029  |

**Kesimpulan:** Model mampu menjelaskan ~30.3% variansi tahun rilis. Rata-rata prediksi meleset sekitar 9.1 tahun, yang wajar mengingat fitur audio saja tidak sepenuhnya merepresentasikan era rilis lagu.

---

### 3. 🔍 Deteksi Fraud Transaksi — Klasifikasi

**Tujuan:** Mendeteksi transaksi kecurangan (fraud) dengan memprediksi label `isFraud` pada dataset transaksi yang sangat tidak seimbang (imbalanced).

**Model yang Digunakan:** LightGBM Classifier (LGBMClassifier) + `scale_pos_weight` untuk penanganan imbalanced data

**Hasil Evaluasi:**
| Metrik                   | Nilai  |
|--------------------------|--------|
| ROC-AUC (OOF)            | 0.9717 |
| Average Precision (PR-AUC) | 0.8597 |
| Accuracy                 | 0.99   |
| Precision (Fraud)        | 0.86   |
| Recall (Fraud)           | 0.77   |
| F1-Score (Fraud)         | 0.81   |

**Kesimpulan:** Model menunjukkan performa sangat kuat dengan ROC-AUC 97.17%, mampu mendeteksi 77% kasus fraud dengan presisi 86%.

---

## 🗺️ Cara Navigasi

1. Buka notebook sesuai topik yang ingin dilihat
2. Jalankan sel secara berurutan dari atas ke bawah
3. Setiap notebook sudah berisi penjelasan per langkah: *data preprocessing → eksplorasi → pelatihan model → evaluasi → kesimpulan*

> **Catatan:** Pastikan semua dependensi sudah terinstall. Cek bagian `import` di awal setiap notebook.

---

## 🔗 Link Repository

[https://github.com/heekal/ML-DL/tree/main/UTS-ML](https://github.com/heekal/ML-DL/tree/main/UTS-ML)
