# 🎓 UTS Deep Learning — Haikal Ali

> Repository tugas tengah semester mata kuliah Deep Learning.  
> Berisi tiga notebook independen yang mencakup task Clustering, Regresi, dan Klasifikasi berbasis Deep Learning.

---

## 👤 Identitas

| Field | Detail        |
|-------|---------------|
| Nama  | Haikal Ali    |
| Kelas | TK-46-GAB     |
| NIM   | 1103223071    |

---

## 📁 Struktur Repository

UTS-DL/  </br>
├── UTS_DL_Clustering.ipynb          # Credit card customer clustering dengan Autoencoder + KMeans  </br>
├── UTS_DL_REGRESI.ipynb             # Prediksi tahun rilis lagu dengan MLP PyTorch  </br>
├── UTS_DL_TransactionDataset.ipynb  # Deteksi fraud dengan TabNet PyTorch  </br
└── README.md  </br>

---

## 📌 Gambaran Umum

Repository ini berisi tiga proyek deep learning yang diselesaikan sebagai bagian dari penilaian UTS. Setiap notebook bersifat mandiri dengan dataset, pipeline preprocessing, arsitektur model neural network, pelatihan, dan evaluasi masing-masing. Seluruh model dibangun menggunakan **PyTorch** sebagai framework utama.

---

## 🧪 Detail Proyek

### 1. 📊 Credit Card Customer Clustering — Deep Clustering

**Tujuan:** Mengelompokkan perilaku pelanggan kartu kredit berdasarkan fitur pengeluaran dan pembayaran menggunakan kombinasi Deep Learning dan unsupervised learning.

**Pipeline:**
```
Raw Features (22 dim)
↓
Deep Autoencoder (Encoder)
↓
Latent Space (8 dim)
↓
K-Means Clustering (K=3)
↓
Visualisasi UMAP & PCA (2D)
```

**Model yang Digunakan:**
| Komponen              | Detail                                              |
|-----------------------|-----------------------------------------------------|
| Deep Autoencoder      | Dimensionality reduction: 22 fitur → 8 latent space |
| K-Means Clustering    | K optimal = 3 cluster                               |
| UMAP & PCA            | Visualisasi proyeksi 2D dari latent space           |

**Hasil Evaluasi:**
| Metrik                    | Nilai     |
|---------------------------|-----------|
| Mean Reconstruction Error | 0.0795    |
| Silhouette Score ↑        | **0.3604** |
| Davies-Bouldin ↓          | 1.5445    |
| Calinski-Harabasz ↑       | 1903.2939 |

**Kesimpulan:** Autoencoder berhasil mengekstrak representasi latent yang lebih baik dibanding raw features, terbukti dari Silhouette Score 0.3604 yang lebih tinggi dibanding pendekatan KMeans murni (0.2586 di UTS-ML).

---

### 2. 📈 Prediksi Tahun Rilis Lagu — Regresi MLP

**Tujuan:** Memprediksi tahun rilis lagu berdasarkan 99 fitur audio numerik menggunakan neural network ringan berbasis PyTorch.

**Arsitektur MLP:**

Input (99) → Dense(256) → Dense(128) → Dense(64) → Output(1)

**Model yang Digunakan:** Multilayer Perceptron (MLP) — PyTorch + K-Fold Cross Validation

**Hasil Evaluasi:**
| Metrik       | Nilai   |
|--------------|---------|
| OOF RMSE     | 8.7086  |
| Test MSE     | 76.7140 |
| Test RMSE    | 8.7587  |
| Test MAE     | 6.0204  |
| Test R²      | 0.3486  |

**Kesimpulan:** Model MLP mampu menjelaskan ~34.9% variansi tahun rilis dengan rata-rata prediksi meleset 8.8 tahun. Performa sedikit lebih baik dari LightGBM (R² 0.3029) di UTS-ML, menunjukkan neural network menangkap pola non-linear yang lebih kompleks dari fitur audio.

---

### 3. 🔍 Deteksi Fraud Transaksi — TabNet Classifier

**Tujuan:** Mendeteksi transaksi penipuan (fraud) pada dataset transaksi yang sangat tidak seimbang menggunakan model berbasis attention mechanism untuk data tabular.

**Model yang Digunakan:** PyTorch TabNetClassifier — dirancang khusus untuk data tabular dengan mekanisme attention yang menyeleksi fitur secara adaptif per-sampel.

**Hasil Evaluasi:**
| Metrik                    | Nilai   |
|---------------------------|---------|
| OOF ROC-AUC               | 0.9206  |
| PR-AUC (Average Precision) | 0.5608 |
| Accuracy                  | 0.89    |
| Precision (Fraud)         | 0.21    |
| Recall (Fraud)            | **0.80** |
| F1-Score (Fraud)          | 0.34    |

**Confusion Matrix:**
| | Prediksi: Non-Fraud | Prediksi: Fraud |
|---|---|---|
| **Aktual: Non-Fraud** | 508.841 (TN) | 61.036 (FP) |
| **Aktual: Fraud**     | 4.134 (FN)   | 16.529 (TP) |

**Kesimpulan:** Model memprioritaskan Recall tinggi (80%) untuk meminimalkan kasus fraud yang tidak terdeteksi (FN), trade-off wajar untuk use case fraud detection. Precision rendah (21%) mengindikasikan banyak false alarm, namun konsekuensi miss fraud jauh lebih berat daripada false alarm di dunia nyata.

---

## 🗺️ Cara Navigasi

1. Masuk ke folder `UTS-DL/`
2. Buka notebook sesuai topik yang ingin dilihat
3. Jalankan sel secara berurutan dari atas ke bawah
4. Setiap notebook sudah berisi penjelasan per langkah: *data preprocessing → eksplorasi → arsitektur model → pelatihan → evaluasi → kesimpulan*

> **Catatan:** Semua model menggunakan PyTorch. Pastikan environment lo sudah terinstall `torch`, `pytorch-tabnet`, `umap-learn`, dan dependensi lainnya yang tercantum di bagian `import` awal setiap notebook.

---

## 🔗 Link Repository

[https://github.com/heekal/ML-DL](https://github.com/heekal/ML-DL)
