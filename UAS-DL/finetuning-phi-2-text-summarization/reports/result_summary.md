# Ringkasan Hasil

## Proyek

Fine-Tuning Phi-2 untuk Abstractive Text Summarization pada Dataset XSum

## Identitas

| Field | Nilai |
| ----- | ----- |
| Nama | Haikal Ali |
| Kelas | TK-46-GAB |
| NIM | 1103223071 |

## Dataset

Dataset yang digunakan pada proyek ini adalah **XSum** dari `EdinburghNLP/xsum`. Dataset ini berisi artikel BBC beserta ringkasan satu kalimat. Dataset XSum cocok untuk tugas **abstractive text summarization**, yaitu tugas membuat ringkasan baru berdasarkan isi artikel, bukan hanya menyalin kalimat penting dari teks asli.

Dataset XSum memiliki tiga bagian utama:

| Split | Jumlah Data |
| ----- | ----------: |
| Train | 204.045 |
| Validation | 11.332 |
| Test | 11.334 |

Setiap data memiliki tiga fitur utama, yaitu:

- `document`: isi artikel berita
- `summary`: ringkasan referensi
- `id`: identitas data

Berdasarkan eksplorasi awal pada 1.000 data training, rata-rata panjang artikel adalah sekitar **362 kata**, sedangkan rata-rata panjang ringkasan adalah sekitar **21 kata**. Hal ini menunjukkan bahwa XSum merupakan dataset yang cukup menantang karena model harus mampu memahami artikel panjang dan mengubahnya menjadi ringkasan singkat dalam satu kalimat.

Untuk menyesuaikan dengan keterbatasan waktu dan resource komputasi, eksperimen ini tidak menggunakan seluruh dataset. Subset yang digunakan adalah:

| Split | Jumlah Sampel yang Digunakan |
| ----- | ---------------------------: |
| Training | 3.000 |
| Validation | 500 |
| Test | 200 |

Penggunaan subset ini bertujuan agar proses fine-tuning tetap praktis, dapat dijalankan pada lingkungan Google Colab/Kaggle, dan tidak membutuhkan waktu training yang terlalu besar.

## Model

Model yang digunakan adalah `microsoft/phi-2`.

Phi-2 merupakan model bahasa decoder-only dengan sekitar **2,7 miliar parameter**. Berbeda dengan model encoder-decoder seperti T5 atau BART yang memang dirancang khusus untuk tugas sequence-to-sequence, Phi-2 bekerja sebagai causal language model yang memprediksi token berikutnya berdasarkan token sebelumnya.

Pada proyek ini, Phi-2 digunakan untuk text summarization melalui format prompt instruksional. Artikel dimasukkan ke dalam prompt, lalu model dilatih untuk menghasilkan bagian `Output` berupa ringkasan satu kalimat.

Format prompt yang digunakan adalah:

```text
Instruct: Summarize the following BBC article in one sentence.
Article: [isi artikel]
Output: [ringkasan]
```

Untuk inference, bagian ringkasan tidak diberikan sehingga model diminta menghasilkan ringkasan sendiri berdasarkan artikel yang diberikan.

## Metode Fine-Tuning

Fine-tuning dilakukan menggunakan metode **QLoRA**, yaitu kombinasi antara 4-bit quantization dan LoRA adapter.

QLoRA digunakan karena Phi-2 memiliki ukuran parameter yang besar. Jika model dimuat secara penuh dalam FP16, kebutuhan VRAM bisa menjadi cukup tinggi. Dengan quantization 4-bit NF4, model dasar dapat dimuat dengan penggunaan memori yang lebih hemat. Kemudian, hanya parameter kecil tambahan dari LoRA adapter yang dilatih, sedangkan sebagian besar parameter model dasar tetap dibekukan.

Konfigurasi quantization yang digunakan adalah:

| Komponen | Nilai |
| -------- | ----- |
| Quantization | 4-bit |
| Quantization Type | NF4 |
| Double Quantization | Ya |
| Compute dtype | bfloat16 |
| Library | BitsAndBytes |

LoRA diterapkan pada beberapa modul linear Phi-2, yaitu:

- `q_proj`
- `k_proj`
- `v_proj`
- `dense`
- `fc1`
- `fc2`

Konfigurasi LoRA yang digunakan adalah:

| Konfigurasi | Nilai |
| ----------- | ----: |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Bias | none |
| Task Type | CAUSAL_LM |

Setelah LoRA diterapkan, jumlah parameter yang dilatih adalah **23.592.960 parameter** dari total **2.803.276.800 parameter**. Dengan demikian, hanya sekitar **0,8416%** parameter yang benar-benar dilatih. Ini menunjukkan bahwa pendekatan LoRA sangat efisien karena tidak perlu melatih seluruh parameter model.

## Konfigurasi Training

| Konfigurasi | Nilai |
| ----------- | ----- |
| Model | `microsoft/phi-2` |
| Dataset | XSum |
| Task | Abstractive Text Summarization |
| Method | QLoRA |
| Training Samples | 3.000 |
| Validation Samples | 500 |
| Test Samples | 200 |
| Epochs | 2 |
| Per-device Train Batch Size | 1 |
| Gradient Accumulation Steps | 8 |
| Effective Batch Size | 8 |
| Per-device Eval Batch Size | 1 |
| Learning Rate | 2e-4 |
| Weight Decay | 0.001 |
| Warmup Ratio | 0.03 |
| LR Scheduler | Cosine |
| Optimizer | paged_adamw_8bit |
| Max Input Length | 512 |
| Max Sequence Length | 640 |
| Evaluation Strategy | Steps |
| Evaluation Steps | 200 |
| Save Steps | 400 |
| Main Metric | eval_loss |
| Output Directory | `./phi2-xsum-qlora` |

Training dijalankan pada GPU **Tesla T4** dengan VRAM sekitar **15,64 GB**. Sebelum training, model menggunakan sekitar **3,75 GB VRAM**. Selama proses training, peak VRAM yang tercatat tetap sekitar **3,75 GB**, sehingga metode QLoRA berhasil membuat proses fine-tuning jauh lebih hemat memori.

## Hasil Training

Hasil training akhir adalah sebagai berikut:

| Metric | Nilai |
| ------ | ----: |
| Training Loss | 2.2259 |
| Training Time | 31.857 detik |
| Peak VRAM Used | 3.75 GB |

Training dilakukan selama 2 epoch dengan total sekitar 750 training steps. Nilai training loss sebesar **2.2259** menunjukkan bahwa model berhasil melakukan proses pembelajaran pada subset XSum, meskipun ukuran data training yang digunakan relatif kecil dibandingkan total dataset asli.

## Hasil Evaluasi Akhir

Evaluasi dilakukan menggunakan metrik ROUGE, yaitu metrik umum untuk tugas summarization. ROUGE mengukur tingkat kemiripan antara ringkasan yang dihasilkan model dengan ringkasan referensi.

| Metric | Score | Score (%) |
| ------ | ----: | --------: |
| ROUGE-1 | 0.3275 | 32.75% |
| ROUGE-2 | 0.1130 | 11.30% |
| ROUGE-L | 0.2602 | 26.02% |

ROUGE-1 mengukur overlap unigram antara ringkasan prediksi dan ringkasan referensi. ROUGE-2 mengukur overlap bigram, sedangkan ROUGE-L mengukur kemiripan berdasarkan longest common subsequence.

Nilai ROUGE-1 sebesar **32,75%** menunjukkan bahwa model cukup mampu menangkap kata-kata penting dari ringkasan referensi. Nilai ROUGE-2 sebesar **11,30%** lebih rendah karena bigram overlap lebih sulit dicapai, terutama pada tugas abstractive summarization. Nilai ROUGE-L sebesar **26,02%** menunjukkan bahwa sebagian struktur atau urutan informasi dalam ringkasan model masih memiliki kemiripan dengan referensi.

## Analisis

Hasil evaluasi menunjukkan bahwa fine-tuning Phi-2 dengan QLoRA mampu menghasilkan ringkasan yang relevan pada dataset XSum, meskipun performanya masih terbatas. Hal ini wajar karena eksperimen hanya menggunakan **3.000 data training** dari total lebih dari **204.000 data training** yang tersedia.

Dataset XSum juga termasuk sulit karena ringkasannya sangat abstraktif. Artinya, ringkasan referensi tidak selalu mengambil kalimat langsung dari artikel, tetapi sering menulis ulang inti berita dalam bentuk kalimat baru. Model tidak cukup hanya mencari kalimat penting, tetapi harus memahami konteks artikel dan menyusun ulang informasi secara ringkas.

Dari hasil qualitative inference, model dapat menghasilkan ringkasan yang secara umum masih berkaitan dengan artikel. Misalnya, pada contoh artikel tentang kebutuhan tempat tinggal bagi mantan narapidana di Wales, model menghasilkan ringkasan yang masih membahas investasi perumahan dan pengurangan reoffending. Walaupun tidak sama persis dengan referensi, ringkasan tersebut masih menangkap inti umum artikel.

Namun, pada beberapa contoh lain, model masih melakukan kesalahan faktual. Contohnya pada artikel tentang penunjukan Nicky Hammond sebagai technical director West Brom, model menghasilkan ringkasan yang menyebut Steve McClaren dan informasi yang tidak sesuai. Ini menunjukkan bahwa model kadang menghasilkan informasi yang terdengar masuk akal tetapi tidak akurat terhadap artikel sumber.

Secara keseluruhan, hasil ini menunjukkan bahwa pendekatan QLoRA efektif untuk menjalankan fine-tuning model besar dalam resource terbatas. Namun, kualitas summarization masih dapat ditingkatkan dengan jumlah data yang lebih besar, durasi training lebih panjang, dan tuning hyperparameter yang lebih baik.

## Error Analysis

Berdasarkan contoh hasil inference dan distribusi skor ROUGE, beberapa pola kesalahan yang dapat diamati adalah sebagai berikut.

### 1. Ringkasan terlalu bebas dari isi artikel

Pada beberapa kasus, model menghasilkan ringkasan yang masih berada dalam topik umum yang sama, tetapi detailnya tidak sepenuhnya sesuai dengan isi artikel. Ini umum terjadi pada abstractive summarization karena model tidak hanya menyalin teks, melainkan mencoba membentuk kalimat baru.

### 2. Kesalahan detail entitas

Model dapat salah menyebut nama orang, organisasi, atau klub. Contohnya, pada salah satu artikel olahraga, model menghasilkan nama dan konteks yang tidak sesuai dengan referensi. Kesalahan seperti ini berbahaya pada summarization karena dapat menghasilkan ringkasan yang terlihat valid tetapi sebenarnya salah.

### 3. Hallucination

Beberapa ringkasan mengandung informasi tambahan yang tidak jelas ada pada artikel sumber. Ini menunjukkan adanya hallucination, yaitu kondisi ketika model menghasilkan informasi yang tidak didukung oleh input.

### 4. ROUGE-2 rendah

Nilai ROUGE-2 sebesar **11,30%** menunjukkan bahwa susunan dua kata berurutan antara prediksi dan referensi masih cukup berbeda. Hal ini menunjukkan bahwa model belum sepenuhnya mampu meniru struktur ringkasan referensi secara konsisten.

### 5. Keterbatasan data training

Hanya **3.000 sampel training** yang digunakan dari total **204.045 data training**. Jumlah ini cukup untuk eksperimen terbatas, tetapi masih kecil untuk memperoleh performa summarization yang kuat pada dataset XSum.

## Penyimpanan Model

Model hasil fine-tuning disimpan dalam bentuk LoRA adapter, bukan full model. Direktori penyimpanan yang digunakan adalah:

```text
./phi2-xsum-qlora
./phi2-xsum-final
```

Adapter ini berukuran jauh lebih kecil dibandingkan model penuh. Untuk menggunakan model kembali, base model Phi-2 perlu dimuat terlebih dahulu, kemudian adapter LoRA digabungkan menggunakan PEFT.

Hal ini membuat proses penyimpanan lebih efisien karena tidak perlu menyimpan ulang seluruh parameter Phi-2 yang berukuran besar.

## Kesimpulan

Proyek ini berhasil mengimplementasikan pipeline fine-tuning model bahasa besar untuk tugas abstractive text summarization. Pipeline yang dibuat mencakup proses instalasi library, load dataset XSum, eksplorasi data, preprocessing prompt, load model Phi-2 dengan 4-bit quantization, konfigurasi LoRA, training menggunakan SFTTrainer, evaluasi dengan ROUGE, qualitative inference, visualisasi distribusi skor, serta penyimpanan model adapter.

Model `microsoft/phi-2` berhasil di-fine-tune menggunakan metode QLoRA dengan hanya sekitar **0,8416% parameter** yang dilatih. Pendekatan ini membuat proses training jauh lebih hemat memori dan dapat berjalan pada GPU Tesla T4. Peak VRAM yang digunakan tercatat sekitar **3,75 GB**, sehingga metode ini cocok untuk eksperimen pada lingkungan resource terbatas.

Hasil evaluasi memperoleh **ROUGE-1 sebesar 32,75%**, **ROUGE-2 sebesar 11,30%**, dan **ROUGE-L sebesar 26,02%** pada 200 data test. Hasil ini menunjukkan bahwa model mampu menghasilkan ringkasan yang cukup relevan, tetapi masih memiliki kelemahan pada detail faktual, konsistensi informasi, dan kemungkinan hallucination.

Secara keseluruhan, eksperimen ini menunjukkan bahwa QLoRA merupakan pendekatan yang efektif untuk fine-tuning model besar seperti Phi-2 dalam keterbatasan komputasi. Performa model masih dapat ditingkatkan dengan menggunakan lebih banyak data training, menambah epoch, melakukan hyperparameter tuning, mencoba decoding strategy lain, atau menggunakan model yang lebih spesifik untuk summarization seperti T5, BART, atau PEGASUS.
