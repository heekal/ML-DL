# Result Summary

## Proyek

Fine-tuning T5 untuk Question Answering menggunakan SQuAD

## Identitas Mahasiswa

| Field | Nilai      |
| ----- | ---------- |
| Nama  | Haikal Ali |
| Kelas | TK-46-GAB  |
| NIM   | 1103223071 |

## Dataset

Dataset yang digunakan pada proyek ini adalah SQuAD dari HuggingFace Datasets. Dataset SQuAD berisi paragraf konteks, pertanyaan, dan jawaban referensi.

Karena ukuran dataset asli cukup besar, eksperimen ini menggunakan subset data agar proses training tetap praktis dijalankan di Google Colab. Data training diambil dari split `train`, sedangkan data validasi dan test diambil dari bagian berbeda pada split `validation`.

## Model

Model yang digunakan adalah `t5-base`, yaitu model Transformer encoder-decoder yang dirancang untuk tugas text-to-text generation.

Pada proyek ini, tugas Question Answering diubah menjadi format text-to-text. Input model berisi pertanyaan dan konteks, sedangkan output model adalah jawaban yang dihasilkan.

Format input:

```text
question: <question> context: <context>
```

Format target:

```text
<answer>
```

## Konfigurasi Training

| Konfigurasi                 | Nilai                         |
| --------------------------- | ----------------------------- |
| Model                       | t5-base                       |
| Dataset                     | SQuAD                         |
| Task                        | Generative Question Answering |
| Train Samples               | 2.000                         |
| Validation Samples          | 400                           |
| Test Samples                | 200                           |
| Epochs                      | 1                             |
| Batch Size                  | 4                             |
| Gradient Accumulation Steps | 4                             |
| Effective Batch Size        | 16                            |
| Learning Rate               | 3e-4                          |
| Max Source Length           | 384                           |
| Max Target Length           | 48                            |
| Device                      | GPU Tesla T4                  |

## Hasil Evaluasi

### Hasil Validasi

| Metric                   |   Score |
| ------------------------ | ------: |
| Validation Loss          |  0.5289 |
| Exact Match              |   64.75 |
| F1 Score                 | 81.4747 |
| Average Generated Length |   3.255 |

### Hasil Test

| Metric                   |   Score |
| ------------------------ | ------: |
| Test Loss                |  0.5111 |
| Exact Match              |   61.50 |
| F1 Score                 | 79.0026 |
| Average Generated Length |    3.58 |

## Analisis

Model `t5-base` yang telah di-fine-tune berhasil memperoleh Exact Match sebesar 61.50 dan F1 Score sebesar 79.0026 pada test set. Hasil ini menunjukkan bahwa model mampu menghasilkan jawaban yang cukup relevan terhadap pertanyaan dan konteks yang diberikan.

Nilai F1 Score lebih tinggi dibandingkan Exact Match karena Exact Match hanya menghitung jawaban sebagai benar apabila hasil prediksi sama persis dengan jawaban referensi setelah normalisasi teks. Sementara itu, F1 Score menghitung overlap token antara jawaban prediksi dan jawaban referensi. Oleh karena itu, jawaban yang sebagian benar atau memiliki kemiripan token masih dapat memperoleh nilai F1 yang baik walaupun tidak lolos Exact Match.

Model bekerja cukup baik pada jawaban faktual yang pendek, seperti nama, tanggal, lokasi, dan frasa singkat. Namun, masih terdapat beberapa kesalahan ketika model memilih entitas yang salah dari konteks atau menghasilkan jawaban yang tidak lengkap. Hal ini wajar pada pendekatan generative Question Answering karena model tidak hanya mengambil span jawaban secara langsung, tetapi menghasilkan jawaban token demi token.

Rata-rata panjang jawaban yang dihasilkan pada test set adalah 3.58 kata. Nilai ini masih masuk akal karena mayoritas jawaban pada SQuAD berbentuk jawaban pendek.

## Kesimpulan

Proyek ini berhasil mengimplementasikan pipeline fine-tuning Transformer untuk Question Answering menggunakan model `t5-base` dan dataset SQuAD. Pipeline yang dibuat mencakup proses load dataset, eksplorasi data, preprocessing, format text-to-text, tokenisasi, fine-tuning model, generate jawaban, evaluasi, dan analisis error.

Berdasarkan hasil evaluasi pada test set, model memperoleh Exact Match sebesar 61.50 dan F1 Score sebesar 79.0026. Hasil ini menunjukkan bahwa model mampu mempelajari pola Question Answering dengan cukup baik meskipun hanya menggunakan subset kecil dari dataset SQuAD.

Performa model masih dapat ditingkatkan dengan menambah jumlah data training, menambah jumlah epoch, melakukan hyperparameter tuning, menggunakan strategi decoding seperti beam search, atau mencoba varian model T5 yang lebih besar.
