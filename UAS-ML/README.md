# 🎓 UAS Machine Learning & Deep Learning — Haikal Ali

> Repository tugas akhir semester mata kuliah Machine Learning dan Deep Learning.  
> Berisi dua notebook end-to-end yang mencakup task Regresi dan Klasifikasi Fraud Transaction.

---

## 👤 Identitas

| Field  | Detail               |
|--------|----------------------|
| Nama   | Haikal Ali           |
| Kelas  | TK-46-GAB            |
| NIM    | 1103223071           |

---

## 📁 Struktur Repository

```text
UAS-ML/
├── README.md
│
├── UAS-ML-FRAUDTRANSACTION/
│   ├── mlflow_tracking.db
│   ├── UAS_ML_FRAUD_TRANSACTION.ipynb
│   │
│   ├── artifacts/
│   │   ├── config.json
│   │   ├── fraud_end_to_end_dl_optuna_mlflow_colab.ipynb
│   │   ├── fraud_transaction_only_dl_optuna_mlflow_colab.ipynb
│   │   ├── optuna_trials.csv
│   │   ├── transaction_only_fraud_mlp.keras
│   │   └── transaction_only_preprocessor.joblib
│   │
│   ├── final/
│   │   └── artifacts/
│   │       ├── config.json
│   │       ├── optuna_trials.csv
│   │       ├── transaction_only_fraud_mlp.keras
│   │       ├── transaction_only_metrics.json
│   │       ├── transaction_only_preprocessor.joblib
│   │       └── transaction_only_test_predictions.csv
│   │
│   └── mlflow_artifacts/
│       └── 18db948135a8464abf0bef352172a7ed/
│           └── artifacts/
│               ├── config.json
│               ├── optuna_trials.csv
│               ├── transaction_only_fraud_mlp.keras
│               ├── transaction_only_metrics.json
│               ├── transaction_only_preprocessor.joblib
│               └── transaction_only_test_predictions.csv
│
└── UAS-ML-REGRESSION/
    ├── mlflow_song_year.db
    ├── song_year_artifacts.zip
    ├── UAS_ML_REGRESSION.ipynb
    │
    ├── mlflow_song_year_artifacts/
    │   └── models/
    │       ├── m-167211879cd547c98c825b9712a8b27b/
    │       ├── m-2814f36da57b4dd6bdf382e7044cc5a4/
    │       └── m-c7c63b21ecd74d74b1eb78c90b802254/
    │
    └── song_year_artifacts/
        ├── deep_learning_mlp.keras
        ├── median_imputer.joblib
        ├── model_comparison_metrics.csv
        └── preprocessing_metadata.json
```

---

## 📌 Gambaran Umum

Repository ini berisi dua proyek end-to-end untuk memenuhi tugas UAS Machine Learning dan Deep Learning. Kedua proyek dibuat dalam notebook terpisah dan mencakup alur kerja lengkap mulai dari data loading, data cleaning, preprocessing, feature engineering, model training, hyperparameter tuning, evaluasi, interpretasi, hingga experiment tracking.

Task yang dikerjakan:

1. **Regresi** — memprediksi tahun rilis lagu berdasarkan fitur numerik audio.
2. **Klasifikasi Fraud** — memprediksi probabilitas transaksi online merupakan transaksi fraud.

---

## 🧪 Detail Proyek

### 1. 📈 Prediksi Tahun Rilis Lagu — Regresi

**Folder:** `UAS-ML-REGRESSION/`  
**Notebook:** `UAS_ML_REGRESSION.ipynb`  
**Dataset:** `midterm-regresi-dataset.csv`  
**Target:** `year`

**Tujuan:**  
Membangun pipeline regresi end-to-end untuk memprediksi tahun rilis lagu berdasarkan fitur numerik audio. Dataset tidak memiliki header, sehingga kolom pertama dibaca sebagai target `year`, sedangkan kolom berikutnya diberi nama `feature_1` sampai `feature_90`.

**Masalah Awal yang Ditangani:**

File CSV tidak memiliki nama kolom. Jika dibaca langsung menggunakan `pd.read_csv()` default, baris pertama akan salah terbaca sebagai header. Karena itu, dataset dibaca menggunakan:

```python
df = pd.read_csv("midterm-regresi-dataset.csv", header=None)
df.columns = ["year"] + [f"feature_{i}" for i in range(1, df.shape[1])]
```

**Tahapan Pipeline:**

- Membaca dataset menggunakan `header=None`
- Rename kolom menjadi `year`, `feature_1`, `feature_2`, ..., `feature_90`
- Validasi struktur dataset dan tipe data
- Membersihkan target `year`
- Analisis missing value
- Menghapus fitur dengan missing value tinggi
- Menghapus fitur konstan atau hampir konstan
- Menangani missing value dengan median imputation
- Menangani outlier menggunakan quantile clipping
- Melakukan train/validation/test split
- Melatih baseline model menggunakan `DummyRegressor`
- Melatih model machine learning menggunakan `HistGradientBoostingRegressor`
- Melakukan hyperparameter tuning menggunakan Optuna
- Melatih model deep learning menggunakan MLP TensorFlow/Keras
- Mengevaluasi model menggunakan MSE, RMSE, MAE, dan R²
- Melakukan interpretasi lokal menggunakan LIME
- Melacak eksperimen menggunakan MLflow SQLite backend
- Menyimpan model dan artifact preprocessing

**Model yang Digunakan:**

| Model | MSE ↓ | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|------:|-------:|------:|-----:|
| Baseline Median | 130.9645 | 11.4440 | 7.6058 | -0.1087 |
| Tuned HistGradientBoosting | 79.4003 | 8.9107 | 6.2333 | 0.3278 |
| **Deep Learning MLP** ✅ Terbaik | **73.5594** | **8.5767** | **5.8563** | **0.3773** |

**Interpretasi Hasil:**  
Model terbaik adalah **Deep Learning MLP** karena memiliki MAE dan RMSE paling rendah serta R² paling tinggi. Nilai MAE sebesar **5.8563** berarti rata-rata prediksi model meleset sekitar **5.9 tahun** dari tahun rilis asli. Nilai R² sebesar **0.3773** menunjukkan bahwa model mampu menjelaskan sekitar **37.7% variasi tahun rilis lagu** berdasarkan fitur audio numerik yang tersedia.

Hasil ini masih realistis karena tahun rilis lagu tidak hanya dipengaruhi oleh fitur audio, tetapi juga faktor lain yang tidak tersedia di dataset, seperti genre, artis, tren musik, kualitas produksi, dan metadata lagu.

**Interpretasi LIME:**  
LIME digunakan untuk menjelaskan satu prediksi individual. Karena fitur tidak memiliki nama domain yang human-friendly, interpretasi dilakukan pada level `feature_1`, `feature_2`, dan seterusnya. Kontribusi positif dari LIME berarti fitur tersebut mendorong prediksi ke tahun yang lebih baru, sedangkan kontribusi negatif mendorong prediksi ke tahun yang lebih lama.

---

### 2. 🔍 Deteksi Fraud Transaksi Online — Klasifikasi

**Folder:** `UAS-ML-FRAUDTRANSACTION/`  
**Notebook:** `UAS_ML_FRAUD_TRANSACTION.ipynb`  
**Dataset:** `train_transaction.csv`  
**Target:** `isFraud`

**Tujuan:**  
Membangun pipeline deep learning end-to-end untuk memprediksi probabilitas suatu transaksi online merupakan transaksi fraud.

**Catatan Dataset:**  
Requirement awal menyebutkan penggunaan transaction table dan identity table. Namun, pada implementasi ini file `train_identity.csv` tidak tersedia, sehingga pipeline dibuat menggunakan **transaction-only dataset** agar notebook tetap runnable dan tidak error.

**Tahapan Pipeline:**

- Membaca dataset transaksi dengan pertimbangan limit RAM Google Colab Free
- Mengecek header, ukuran dataset, tipe data, dan distribusi target
- Melakukan profiling missing value secara chunk
- Memilih kolom berdasarkan missing ratio agar dataset lebih ringan
- Melakukan feature engineering dari `TransactionDT` dan `TransactionAmt`
- Membagi data menggunakan time-based split
- Melakukan frequency encoding untuk fitur kategorikal
- Melakukan imputation dan scaling
- Menangani class imbalance menggunakan `class_weight`
- Melatih model deep learning MLP berbasis TensorFlow/Keras
- Melakukan hyperparameter tuning menggunakan Optuna
- Menggunakan PR-AUC sebagai metric utama tuning karena target sangat imbalance
- Menentukan threshold terbaik dari validation set berdasarkan F1-score
- Mengevaluasi model pada validation dan test set
- Menyimpan model, preprocessor, metrics, trial Optuna, dan hasil prediksi
- Melacak eksperimen menggunakan MLflow SQLite backend

**Model yang Digunakan:**  
Deep Learning MLP dengan class weighting dan hyperparameter tuning menggunakan Optuna.

**Best Hyperparameter dari Optuna:**

| Parameter | Nilai |
|----------|------:|
| `units_1` | 256 |
| `units_2` | 64 |
| `use_third_layer` | False |
| `dropout_1` | 0.1228 |
| `dropout_2` | 0.4321 |
| `learning_rate` | 0.00156 |
| `activation` | swish |
| `batch_size` | 4096 |

**Hasil Evaluasi:**

| Dataset | ROC-AUC ↑ | PR-AUC ↑ | Log Loss ↓ | Precision ↑ | Recall ↑ | F1-Score ↑ |
|---------|----------:|---------:|-----------:|------------:|---------:|-----------:|
| Validation | 0.8570 | 0.4067 | 0.2687 | 0.4468 | 0.4007 | 0.4225 |
| Test | 0.8387 | 0.3092 | 0.2717 | 0.4015 | 0.3253 | 0.3594 |

**Best Threshold:** `0.8273`

**Classification Report Test Set:**

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 / Non-Fraud | 0.9758 | 0.9825 | 0.9792 | 85498 |
| 1 / Fraud | 0.4015 | 0.3253 | 0.3594 | 3083 |

**Confusion Matrix Test Set:**

| Actual \ Predicted | Non-Fraud | Fraud |
|--------------------|----------:|------:|
| Non-Fraud | 84003 | 1495 |
| Fraud | 2080 | 1003 |

**Interpretasi Hasil:**  
Model memperoleh ROC-AUC test sebesar **0.8387**, yang berarti model cukup mampu membedakan transaksi fraud dan non-fraud. Namun, PR-AUC dan F1-score masih lebih rendah karena dataset fraud sangat tidak seimbang. Dalam kasus fraud detection, accuracy tidak cukup untuk menilai performa model karena kelas non-fraud jauh lebih dominan.

Model berhasil mendeteksi **1003 transaksi fraud** pada test set, tetapi masih melewatkan **2080 transaksi fraud**. Ini menunjukkan bahwa model sudah memiliki kemampuan deteksi, tetapi recall fraud masih dapat ditingkatkan jika tujuan sistem lebih memprioritaskan menangkap sebanyak mungkin transaksi fraud.

---

## 🧰 Teknologi yang Digunakan

| Kategori | Tools / Library |
|---------|------------------|
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Interpretability | LIME |
| Model Persistence | joblib, Keras model format |
| Environment | Google Colab |

---

## 📊 Ringkasan Model Terbaik

| Task | Model Terbaik | Metrik Utama | Nilai |
|------|---------------|--------------|------:|
| Regresi Tahun Lagu | Deep Learning MLP | Test MAE | 5.8563 |
| Regresi Tahun Lagu | Deep Learning MLP | Test RMSE | 8.5767 |
| Regresi Tahun Lagu | Deep Learning MLP | Test R² | 0.3773 |
| Fraud Transaction | Deep Learning MLP | Test ROC-AUC | 0.8387 |
| Fraud Transaction | Deep Learning MLP | Test PR-AUC | 0.3092 |
| Fraud Transaction | Deep Learning MLP | Test F1-Score | 0.3594 |

---

## 🗺️ Cara Navigasi

1. Buka folder sesuai task:
   - `UAS-ML-REGRESSION/` untuk prediksi tahun rilis lagu.
   - `UAS-ML-FRAUDTRANSACTION/` untuk deteksi fraud transaksi.
2. Buka notebook utama pada masing-masing folder:
   - `UAS_ML_REGRESSION.ipynb`
   - `UAS_ML_FRAUD_TRANSACTION.ipynb`
3. Jalankan notebook dari atas ke bawah.
4. Lihat folder artifact untuk output model, metrik, konfigurasi, dan hasil prediksi.
5. Database MLflow tersedia dalam file:
   - `mlflow_song_year.db`
   - `mlflow_tracking.db`

---

## ⚠️ Catatan Penting

- Folder `artifacts/` pada fraud transaction berisi artifact dan notebook draft. Output final utama tersedia di `final/artifacts/`.
- Proyek fraud transaction menggunakan **transaction-only dataset** karena file identity tidak tersedia.
- Evaluasi fraud tidak berfokus pada accuracy saja karena dataset sangat imbalanced.
- Hasil training dapat sedikit berubah jika notebook dijalankan ulang karena deep learning dan Optuna memiliki unsur stochastic.
- Model regresi memiliki keterbatasan karena hanya memakai fitur audio numerik tanpa metadata tambahan seperti genre, artis, dan popularitas lagu.
- MLflow menggunakan SQLite backend agar kompatibel dengan environment Google Colab dan versi MLflow terbaru.

---

## 🔗 Link Repository

[https://github.com/heekal/ML-DL/tree/main/UAS-ML](https://github.com/heekal/ML-DL/tree/main/UAS-ML)
