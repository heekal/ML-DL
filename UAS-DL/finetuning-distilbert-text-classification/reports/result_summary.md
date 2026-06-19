# Ringkasan Hasil

## Proyek

Fine-Tuning DistilBERT untuk Klasifikasi Teks AG News

## Identitas

| Field | Nilai |
|---|---|
| Nama | Haikal Ali |
| Kelas | TK-46-GAB |
| NIM | 1103223071 |

## Dataset

Dataset yang digunakan pada proyek ini adalah **AG News** dari HuggingFace Datasets. Dataset ini berisi artikel berita pendek yang diklasifikasikan ke dalam empat kategori utama:

- World
- Sports
- Business
- Sci/Tech

AG News digunakan karena cocok untuk tugas **multi-class text classification**, yaitu klasifikasi teks dengan lebih dari dua kelas. Setiap data berisi teks berita dan label kategori yang menjadi target prediksi model.

Untuk menyesuaikan dengan keterbatasan waktu dan resource Google Colab, eksperimen ini tidak menggunakan seluruh dataset. Sebagai gantinya, digunakan subset terkontrol agar proses training tetap praktis, reproducible, dan dapat dijalankan dalam lingkungan komputasi terbatas.

## Model

Model yang digunakan adalah `distilbert-base-uncased`.

DistilBERT merupakan model encoder-only Transformer dari keluarga BERT yang dibuat lebih ringan dibandingkan BERT asli. Model ini dipilih karena lebih cepat dan lebih hemat memori, tetapi tetap memiliki kemampuan representasi bahasa yang baik untuk tugas klasifikasi teks.

Pada proyek ini, DistilBERT digunakan sebagai model sequence classification. Teks berita dimasukkan ke tokenizer, kemudian model memprediksi salah satu dari empat label kategori AG News.

## Konfigurasi Training

| Konfigurasi | Nilai |
|---|---:|
| Model | distilbert-base-uncased |
| Dataset | AG News |
| Task | Multi-class Text Classification |
| Training Samples | 8.000 |
| Validation Samples | 1.000 |
| Test Samples | 1.000 |
| Epochs | 2 |
| Batch Size | 16 |
| Evaluation Batch Size | 32 |
| Learning Rate | 2e-5 |
| Max Token Length | 192 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Main Metric | Macro F1 |

## Hasil Evaluasi Akhir

| Metric | Score |
|---|---:|
| Accuracy | 0.9090 |
| Macro Precision | 0.9115 |
| Macro Recall | 0.9100 |
| Macro F1 | 0.9089 |
| Weighted F1 | 0.9090 |

## Hasil Evaluasi Per Kelas

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| World | 0.9623 | 0.8647 | 0.9109 | 266 |
| Sports | 0.9683 | 0.9919 | 0.9799 | 246 |
| Business | 0.8922 | 0.8415 | 0.8661 | 246 |
| Sci/Tech | 0.8231 | 0.9421 | 0.8786 | 242 |

Dari total 1.000 data test, model menghasilkan 909 prediksi benar dan 91 prediksi salah.

## Analisis

Model DistilBERT yang telah di-fine-tune memperoleh accuracy sebesar 0.9090 dan macro F1 sebesar 0.9089 pada test set. Hasil ini menunjukkan bahwa model mampu melakukan klasifikasi berita AG News dengan performa yang cukup kuat.

Macro F1 digunakan sebagai metrik utama karena dataset memiliki empat kelas yang perlu diperlakukan secara seimbang. Berbeda dengan accuracy yang hanya menghitung proporsi prediksi benar secara keseluruhan, macro F1 menghitung rata-rata performa setiap kelas tanpa terlalu dipengaruhi oleh jumlah data pada masing-masing kelas. Hal ini penting untuk memastikan bahwa model tidak hanya bagus pada kelas yang dominan, tetapi juga tetap cukup baik pada seluruh kategori.

Berdasarkan hasil per kelas, kategori **Sports** memperoleh performa terbaik dengan F1-score sebesar 0.9799. Hal ini menunjukkan bahwa berita olahraga cenderung memiliki pola bahasa dan kata kunci yang lebih mudah dibedakan, seperti nama tim, pertandingan, liga, atau atlet.

Kategori **World** juga memperoleh precision yang tinggi sebesar 0.9623, tetapi recall-nya lebih rendah yaitu 0.8647. Artinya, ketika model memprediksi suatu teks sebagai World, prediksi tersebut biasanya benar. Namun, masih ada sebagian berita World yang diklasifikasikan ke kategori lain.

Kategori **Business** memiliki F1-score sebesar 0.8661, sedangkan **Sci/Tech** memperoleh F1-score sebesar 0.8786. Kedua kelas ini menjadi area yang lebih menantang karena topik bisnis dan teknologi sering saling tumpang tindih. Contohnya, berita tentang Google, pasar teknologi, perusahaan perangkat lunak, atau produk digital dapat memiliki unsur Business sekaligus Sci/Tech.

Nilai precision Sci/Tech sebesar 0.8231 menunjukkan bahwa sebagian prediksi Sci/Tech masih berasal dari kelas lain. Namun, recall Sci/Tech sebesar 0.9421 menunjukkan bahwa sebagian besar data yang benar-benar termasuk Sci/Tech berhasil dikenali oleh model. Dengan kata lain, model cukup sensitif dalam mendeteksi berita teknologi, tetapi kadang terlalu mudah mengklasifikasikan berita lain sebagai Sci/Tech.

## Error Analysis

Confusion matrix dan tabel wrong predictions digunakan untuk melihat kesalahan model secara lebih detail. Dari 1.000 data test, terdapat 91 kesalahan prediksi. Kesalahan paling umum terjadi pada berita yang memiliki topik campuran atau konteks yang ambigu.

Beberapa pola kesalahan yang dapat diamati:

1. Berita Business dapat diprediksi sebagai Sci/Tech ketika teks membahas perusahaan teknologi, produk digital, mesin pencari, perangkat lunak, atau isu keamanan teknologi.
2. Berita World dapat diprediksi sebagai Sports ketika berita membahas negara, kompetisi internasional, atau tokoh olahraga dalam konteks global.
3. Berita Sci/Tech dapat diprediksi sebagai World apabila teks membahas kejadian ilmiah atau teknologi yang memiliki konteks global.
4. Berita Business dapat diprediksi sebagai Sports ketika teks membahas perusahaan atau tokoh bisnis yang berkaitan dengan industri olahraga.

Kesalahan seperti ini wajar pada klasifikasi berita karena kategori berita tidak selalu terpisah secara mutlak. Satu artikel dapat memiliki unsur ekonomi, teknologi, politik, dan olahraga secara bersamaan.

## Kesimpulan

Proyek ini berhasil mengimplementasikan pipeline end-to-end untuk fine-tuning model Transformer pada tugas klasifikasi teks. Pipeline yang dibuat mencakup proses load dataset, eksplorasi data, sampling, tokenisasi, fine-tuning model, evaluasi, visualisasi confusion matrix, error analysis, inference, serta penyimpanan model dan hasil eksperimen.

Model `distilbert-base-uncased` memperoleh accuracy sebesar 0.9090 dan macro F1 sebesar 0.9089 pada test set. Hasil ini menunjukkan bahwa DistilBERT mampu melakukan klasifikasi berita AG News dengan baik meskipun menggunakan subset data yang lebih kecil untuk menyesuaikan keterbatasan resource.

Secara keseluruhan, DistilBERT terbukti menjadi pilihan model yang efektif untuk multi-class text classification karena memiliki keseimbangan antara performa, kecepatan training, dan efisiensi memori. Performa model masih dapat ditingkatkan dengan menggunakan jumlah data training yang lebih besar, menambah epoch, melakukan hyperparameter tuning, atau mencoba model encoder yang lebih besar seperti BERT-base atau RoBERTa.
