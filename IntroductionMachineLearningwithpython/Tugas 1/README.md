# 📚 Introduction to Machine Learning with scikit-learn

> Repositori ini berisi catatan belajar, kode, dan penjelasan teori dari buku **"Introduction to Machine Learning with Python"** oleh Andreas C. Müller & Sarah Guido (O'Reilly).

### Chapter 1 — Introduction

**Apa itu Machine Learning?**

Machine learning adalah pendekatan untuk mengekstrak pengetahuan dari data, tanpa harus menulis aturan logika secara manual. Dua kelemahan utama sistem berbasis aturan (rule-based):
- Logika terikat satu domain dan task tertentu — mengubah task sedikit saja butuh rewrite keseluruhan sistem.
- Membutuhkan pemahaman mendalam dari human expert.

ML menjadi solusi ketika aturan terlalu kompleks untuk ditulis secara manual (contoh: deteksi wajah, filter spam, deteksi fraud).

---

**Jenis-Jenis Machine Learning**

| Tipe | Deskripsi | Contoh |
|------|-----------|--------|
| **Supervised Learning** | Data input + output (label) tersedia. Algoritma belajar memetakan input → output. | Klasifikasi tumor, deteksi fraud, zip code recognition |
| **Unsupervised Learning** | Hanya input yang tersedia, tidak ada label. Algoritma menemukan struktur sendiri. | Clustering pelanggan, topic modeling, anomaly detection |

---

**Konsep Kunci**

| Istilah | Penjelasan |
|---------|-----------|
| **Sample / Data Point** | Satu baris data (misal: satu bunga iris) |
| **Feature** | Kolom / atribut yang mendeskripsikan sample (misal: panjang sepal) |
| **Label** | Output yang ingin diprediksi (misal: spesies bunga) |
| **Training Set** | Data yang dipakai untuk melatih model |
| **Test Set** | Data yang dipakai untuk mengevaluasi performa model (tidak dilihat saat training) |
| **Accuracy** | Proporsi prediksi benar terhadap total prediksi |

---

**Studi Kasus: Klasifikasi Iris**

Dataset Iris adalah dataset klasik berisi 150 data bunga iris dari 3 spesies: *setosa*, *versicolor*, dan *virginica*. Setiap bunga memiliki 4 fitur: panjang/lebar sepal dan panjang/lebar petal.

Workflow yang digunakan:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load data
iris_dataset = load_iris()

# Split data 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(
  iris_dataset['data'], iris_dataset['target'], random_state=0
)

# Buat model k-NN dengan k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Evaluasi
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
# Output: Test set score: 0.97
```

**Mengapa k-Nearest Neighbors?**

k-NN adalah algoritma yang intuitif: untuk memprediksi label sebuah data baru, cari `k` data training yang paling dekat (secara jarak), lalu ambil label mayoritas dari mereka. Dengan `k=1`, cukup gunakan tetangga terdekat saja.

---

**Library yang Digunakan**

| Library | Fungsi |
|---------|--------|
| `numpy` | Array multidimensi, operasi matematika |
| `scipy` | Operasi ilmiah lanjutan, sparse matrix |
| `pandas` | Manipulasi data tabular (DataFrame) |
| `matplotlib` | Visualisasi data (plot, histogram, scatter) |
| `scikit-learn` | Algoritma ML, preprocessing, evaluasi |

---

**Kesimpulan Chapter 1**

- ML lebih efektif dari rule-based system ketika aturan terlalu kompleks untuk ditulis manual.
- Supervised learning butuh pasangan input-output; unsupervised hanya input.
- Selalu pisahkan data menjadi training set dan test set — jangan evaluasi model menggunakan data yang sama yang dipakai untuk training.
- Interface standar scikit-learn: `.fit()`, `.predict()`, `.score()` — berlaku untuk hampir semua model.
- Model k-NN pada Iris dataset mencapai akurasi **97%** pada test set.

---

## 🛠️ Setup Environment

```bash
pip install numpy scipy matplotlib ipython scikit-learn pandas
```

Atau gunakan Anaconda (recommended):

```bash
conda install numpy scipy matplotlib ipython scikit-learn pandas jupyter
```

Versi yang digunakan dalam buku ini:

```
Python       : 3.5.2
NumPy        : 1.11.1
SciPy        : 0.17.1
matplotlib   : 1.5.1
pandas       : 0.18.1
IPython      : 5.1.0
scikit-learn : 0.18
```

> scikit-learn versi **≥ 0.18** wajib — modul `model_selection` baru tersedia di versi ini.

---

## 📌 Catatan

- Semua notebook diasumsikan dijalankan di **Jupyter Notebook** dengan `%matplotlib inline` aktif.
- Penjelasan teori di setiap notebook dibantu dengan referensi dari buku dan LLM.
- File `mglearn` (utility plotting dari buku) bisa diinstall via: `pip install mglearn`
