# ML-DL

> **Repo Tugas Machine Learning dan Deep Learning**  
> Kumpulan notebook pembelajaran, recreate buku, enrichment task, dan UTS untuk topik Machine Learning, Deep Learning, Statistik, dan Aljabar Linear.

---

## 📌 Overview

Repository ini berisi kumpulan notebook berbasis **Jupyter Notebook / Google Colab** untuk mendukung pembelajaran **Machine Learning** dan **Deep Learning** dari dasar sampai praktik model yang lebih kompleks.

Isi repository tidak hanya berfokus pada model, tetapi juga mencakup fondasi penting seperti:

- statistik praktis untuk data science,
- aljabar linear untuk machine learning,
- supervised dan unsupervised learning,
- preprocessing dan feature engineering,
- evaluasi model,
- deep learning dari nol,
- CNN, RNN, LSTM, NLP,
- federated learning,
- proyek UTS ML,
- proyek UTS DL.

---

## 🛠️ Tech Stack

Tools dan library utama yang digunakan di repository ini:

- **Python**
- **Jupyter Notebook / Google Colab**
- **NumPy**
- **pandas**
- **SciPy**
- **matplotlib**
- **scikit-learn**
- **PyTorch**
- **TensorFlow / Keras**
- **LightGBM**
- **PyTorch TabNet**
- **UMAP**
- **phe** untuk eksperimen homomorphic encryption

---

## 📁 Global File Structure

```text
ML-DL/
├── GrokkingDeepLearning/
│   ├── 16 notebook Deep Learning dari dasar sampai federated learning
│   └── README.md
├── IntroductionMachineLearningwithpython/
│   ├── 8 notebook Introduction to Machine Learning with Python
│   └── README.md
├── Practical-Statistics-for-Data-Scientist-Books/
│   ├── Classification.ipynb
│   ├── Data_and_Sampling_Distributions_Colab.ipynb
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── Regression_and_Prediction.ipynb
│   ├── Statistical_Experiments_and_Significance_Testing.ipynb
│   ├── Unsupervised_Learning.ipynb
│   └── README.md
├── PracticalLinearAlgebra/
│   ├── 16 notebook Practical Linear Algebra for Data Science
│   └── README.md
├── scikit-learn-cookbook/
│   ├── 13 notebook scikit-learn Cookbook
│   └── README.md
├── UTS-ML/
│   ├── UTS_ML_Clustering.ipynb
│   ├── UTS_ML_REGRESI.ipynb
│   ├── UTS_ML_TransactionDataset.ipynb
│   └── README.md
├── UTS-DL/
│   ├── UTS_DL_Clustering.ipynb
│   ├── UTS_DL_REGRESI.ipynb
│   ├── UTS_DL_TransactionDataset.ipynb
│   └── README.md
└── README.md
```

---

## 🧭 Repository Breakdown

| Folder | Fokus | Isi Utama |
| --- | --- | --- |
| `IntroductionMachineLearningwithpython` | Dasar Machine Learning | Recreate buku *Introduction to Machine Learning with Python* berisi supervised learning, unsupervised learning, feature engineering, model evaluation, pipelines, dan text data. |
| `scikit-learn-cookbook` | Enrichment Machine Learning | Reproduksi dan theoretical deep-dive dari *scikit-learn Cookbook*, mencakup API scikit-learn, preprocessing, dimensionality reduction, KNN, linear models, logistic regression, SVM, tree ensemble, NLP, clustering, outlier detection, cross-validation, dan deployment. |
| `Practical-Statistics-for-Data-Scientist-Books` | Statistik Data Science | Notebook statistik praktis seperti sampling distribution, EDA, regression, significance testing, classification, dan unsupervised learning. |
| `PracticalLinearAlgebra` | Fondasi Matematika | Recreate *Practical Linear Algebra for Data Science* dengan Python, mencakup vectors, matrices, inverse, QR, LU, least squares, eigendecomposition, SVD, PCA, dan aplikasi data science. |
| `GrokkingDeepLearning` | Deep Learning dari Nol | Recreate *Grokking Deep Learning* dalam Colab, mulai dari neural prediction, gradient descent, backpropagation, regularization, CNN, NLP, RNN, LSTM, mini framework, dan federated learning. |
| `UTS-ML` | Proyek UTS Machine Learning | Tiga notebook: clustering pelanggan kartu kredit, regresi tahun rilis lagu, dan klasifikasi fraud transaction. |
| `UTS-DL` | Proyek UTS Deep Learning | Tiga notebook: deep clustering dengan autoencoder, regresi MLP PyTorch, dan fraud detection dengan TabNet. |

---

## 📚 Learning Path yang Disarankan

Urutan belajar yang paling masuk akal dari repository ini:

1. **Fondasi matematika**
   - `PracticalLinearAlgebra`
   - `Practical-Statistics-for-Data-Scientist-Books`

2. **Dasar Machine Learning**
   - `IntroductionMachineLearningwithpython`

3. **Machine Learning lebih luas dan praktis**
   - `scikit-learn-cookbook`

4. **Proyek Machine Learning**
   - `UTS-ML`

5. **Deep Learning dari nol**
   - `GrokkingDeepLearning`

6. **Proyek Deep Learning**
   - `UTS-DL`

---

## 🔥 Highlight Project

### UTS Machine Learning

Folder `UTS-ML` berisi tiga proyek utama:

| Project | Task | Model / Approach |
| --- | --- | --- |
| Customer Segmentation | Clustering | KMeans, Agglomerative Clustering, DBSCAN |
| Song Release Year Prediction | Regression | LightGBM Regressor + 5-Fold Cross Validation |
| Fraud Transaction Detection | Classification | LightGBM Classifier + `scale_pos_weight` |

Ringkasan hasil:
- Clustering terbaik menggunakan **KMeans** dengan Silhouette Score **0.2586**.
- Regresi tahun rilis lagu menghasilkan RMSE sekitar **9.0609** dan R² **0.3029**.
- Fraud detection mencapai ROC-AUC **0.9717**, Precision Fraud **0.86**, Recall Fraud **0.77**, dan F1-Score Fraud **0.81**.

---

### UTS Deep Learning

Folder `UTS-DL` berisi tiga proyek utama:

| Project | Task | Model / Approach |
| --- | --- | --- |
| Credit Card Customer Clustering | Deep Clustering | Autoencoder + KMeans |
| Song Release Year Prediction | Regression | MLP PyTorch + K-Fold Cross Validation |
| Fraud Transaction Detection | Classification | PyTorch TabNetClassifier |

Ringkasan hasil:
- Deep clustering menggunakan Autoencoder menghasilkan Silhouette Score **0.3604**.
- Regresi MLP menghasilkan Test RMSE **8.7587** dan Test R² **0.3486**.
- Fraud detection dengan TabNet menghasilkan OOF ROC-AUC **0.9206**, Recall Fraud **0.80**, dan F1-Score Fraud **0.34**.

---

## 🚀 How to Use

Clone repository:

```bash
git clone https://github.com/heekal/ML-DL.git
cd ML-DL
```

Buka folder yang ingin dipelajari:

```bash
cd GrokkingDeepLearning
```

Jalankan notebook dengan Jupyter:

```bash
jupyter notebook
```

Atau upload notebook ke **Google Colab** dan jalankan cell dari atas ke bawah.

---

## 📦 Recommended Environment

Minimal environment:

```bash
pip install numpy pandas scipy matplotlib scikit-learn jupyter
```

Untuk notebook tertentu, dependency tambahan mungkin dibutuhkan:

```bash
pip install torch tensorflow lightgbm pytorch-tabnet umap-learn phe
```

Cek bagian `import` di awal setiap notebook untuk dependency spesifik.

---

## 🎯 Who Is This For?

Repository ini cocok untuk:

- mahasiswa yang sedang mengambil mata kuliah Machine Learning dan Deep Learning,
- pemula yang ingin belajar ML/DL lewat notebook langsung jalan,
- pembelajar yang ingin memahami teori sambil praktik,
- calon AI Engineer, ML Engineer, Data Scientist, atau Research Engineer,
- siapa pun yang ingin membangun portofolio ML/DL berbasis project dan code reproduction.

---

## 🧠 Notes

Repository ini berisi banyak notebook hasil recreate dan pembelajaran dari berbagai buku serta tugas kuliah. Beberapa folder memiliki gaya README berbeda karena dibuat untuk konteks tugas yang berbeda.

README global ini berfungsi sebagai pintu masuk utama agar struktur repository lebih mudah dipahami.

---

## ⚖️ Copyright & Acknowledgments

Repository ini berisi notebook pembelajaran dan recreate dari beberapa sumber buku serta tugas akademik, termasuk:

- *Introduction to Machine Learning with Python* — Andreas C. Müller & Sarah Guido
- *Grokking Deep Learning* — Andrew W. Trask
- *Practical Linear Algebra for Data Science* — Mike X Cohen
- *scikit-learn Cookbook* — John Sukup
- materi statistik praktis untuk data science
- tugas UTS Machine Learning dan Deep Learning

Semua hak cipta atas buku, struktur materi asli, dan konten sumber tetap milik penulis serta penerbit masing-masing. Repository ini digunakan untuk tujuan edukasi, pembelajaran pribadi, dan dokumentasi tugas.

---

## 👤 Author

**Haikal Ali**  
Computer Engineering Student  
Machine Learning & Deep Learning Coursework Repository
