# UAS Deep Learning Projects

> Repository ini berisi kumpulan proyek **UAS Deep Learning** yang berfokus pada implementasi dan fine-tuning model NLP berbasis Transformer untuk tiga task utama: **text classification**, **text summarization**, dan **question answering**.

---

## Identitas

| Field | Nilai |
| ----- | ----- |
| Nama | Haikal Ali |
| Kelas | TK-46-GAB |
| NIM | 1103223071 |

## Group Identification

| Field | Nilai |
| ----- | ----- |
| Group Name / Number | Kelompok 12 |
| Member 1 | Haikal Ali - 1103223071 |

---

## Purpose of the Repository

Repository ini dibuat untuk memenuhi tugas **UAS Deep Learning** dengan menerapkan model deep learning modern pada task Natural Language Processing (NLP).

Tujuan utama repository ini adalah:

- menerapkan fine-tuning model Transformer pada dataset NLP;
- membandingkan penggunaan model berbeda untuk task berbeda;
- mendokumentasikan proses eksperimen dari notebook sampai laporan;
- menyimpan hasil evaluasi, visualisasi, dan ringkasan eksperimen;
- menyediakan struktur repository yang mudah dinavigasi melalui GitHub;
- menunjukkan pipeline end-to-end mulai dari dataset loading, preprocessing, training, evaluation, inference, sampai penyimpanan output.

---

## Project Overview

Repository ini terdiri dari tiga proyek utama:

| Folder | Task | Model | Dataset |
| ------ | ---- | ----- | ------- |
| `finetuning-distilbert-text-classification` | Text Classification | DistilBERT | AG News |
| `finetuning-phi-2-text-summarization` | Text Summarization | Phi-2 + QLoRA | XSum |
| `finetuning-t5-question-answering` | Question Answering | T5 | SQuAD |

Setiap project memiliki struktur yang mirip:

- `README.md` untuk dokumentasi project;
- `requirements.txt` untuk dependency;
- `notebooks/` untuk notebook utama;
- `outputs/` untuk hasil evaluasi, metrik, visualisasi, atau prediksi;
- `reports/` untuk ringkasan hasil eksperimen.

---

## Repository Structure

```text
UAS-DL/
├── finetuning-distilbert-text-classification/
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   │   └── DISTILBERT_AG_NEWS.ipynb
│   ├── outputs/
│   │   ├── .gitkeep
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   └── reports/
│       └── result_summary.md
│
├── finetuning-phi-2-text-summarization/
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/
│   │   └── PHI2_XSUM_LORA_SUMMARIZATION.ipynb
│   ├── outputs/
│   │   └── rouge_distribution.png
│   └── reports/
│       └── result_summary.md
│
└── finetuning-t5-question-answering/
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    │   └── T5_SQUAD_QUESTION_ANSWERING.ipynb
    ├── outputs/
    │   ├── .gitkeep
    │   ├── metrics.json
    │   └── sample_predictions.csv
    └── reports/
        └── result_summary.md
```

---

## Project 1: Fine-Tuning DistilBERT for Text Classification

### Project Location

```text
finetuning-distilbert-text-classification/
```

### Brief Overview

Project ini menerapkan fine-tuning model `distilbert-base-uncased` untuk melakukan klasifikasi teks berita pada dataset **AG News**.

AG News merupakan dataset text classification dengan empat kelas:

- World
- Sports
- Business
- Sci/Tech

Task pada project ini adalah **multi-class text classification**, yaitu model menerima input berupa teks berita dan memprediksi salah satu dari empat kategori tersebut.

### Model Description

Model yang digunakan adalah **DistilBERT**, yaitu versi ringan dari BERT. DistilBERT dipilih karena lebih cepat dan lebih hemat memori dibandingkan BERT-base, tetapi tetap memiliki kemampuan representasi bahasa yang baik untuk tugas klasifikasi teks.

Pipeline utama project ini meliputi:

1. load dataset AG News;
2. sampling data agar training tetap praktis;
3. tokenisasi teks menggunakan tokenizer DistilBERT;
4. fine-tuning model sequence classification;
5. evaluasi menggunakan accuracy, precision, recall, dan F1-score;
6. visualisasi confusion matrix;
7. penyimpanan metrics dan report.

### Metrics Results

| Metric | Score |
| ------ | ----: |
| Accuracy | 0.9090 |
| Macro Precision | 0.9115 |
| Macro Recall | 0.9100 |
| Macro F1 | 0.9089 |
| Weighted F1 | 0.9090 |

### Per-Class Results

| Class | Precision | Recall | F1-score | Support |
| ----- | --------: | -----: | -------: | ------: |
| World | 0.9623 | 0.8647 | 0.9109 | 266 |
| Sports | 0.9683 | 0.9919 | 0.9799 | 246 |
| Business | 0.8922 | 0.8415 | 0.8661 | 246 |
| Sci/Tech | 0.8231 | 0.9421 | 0.8786 | 242 |

### Output Files

| File | Description |
| ---- | ----------- |
| `outputs/confusion_matrix.png` | Confusion matrix hasil evaluasi model |
| `outputs/metrics.json` | File metrik evaluasi akhir |
| `reports/result_summary.md` | Ringkasan hasil eksperimen |
| `notebooks/DISTILBERT_AG_NEWS.ipynb` | Notebook utama fine-tuning |

---

## Project 2: Fine-Tuning Phi-2 for Text Summarization

### Project Location

```text
finetuning-phi-2-text-summarization/
```

### Brief Overview

Project ini menerapkan fine-tuning model `microsoft/phi-2` untuk tugas **abstractive text summarization** pada dataset **XSum**.

XSum berisi artikel berita BBC dan ringkasan satu kalimat. Task summarization pada dataset ini cukup menantang karena ringkasan bersifat abstraktif, sehingga model tidak hanya menyalin kalimat dari artikel, tetapi harus memahami isi artikel dan menghasilkan ringkasan baru.

### Model Description

Model yang digunakan adalah **Phi-2**, yaitu model bahasa decoder-only dengan sekitar 2,7 miliar parameter. Karena ukuran model cukup besar, fine-tuning dilakukan menggunakan pendekatan **QLoRA**.

QLoRA menggabungkan:

- 4-bit quantization;
- LoRA adapter;
- parameter-efficient fine-tuning.

Dengan metode ini, model dapat dilatih pada resource terbatas karena hanya sebagian kecil parameter adapter yang dilatih, sedangkan model utama tetap dibekukan.

Pipeline utama project ini meliputi:

1. load dataset XSum;
2. eksplorasi panjang artikel dan ringkasan;
3. formatting prompt instruction;
4. load Phi-2 dengan 4-bit quantization;
5. konfigurasi LoRA adapter;
6. training menggunakan SFTTrainer;
7. evaluasi menggunakan ROUGE;
8. qualitative inference;
9. visualisasi distribusi ROUGE;
10. penyimpanan result summary.

### Metrics Results

| Metric | Score | Score (%) |
| ------ | ----: | --------: |
| ROUGE-1 | 0.3275 | 32.75% |
| ROUGE-2 | 0.1130 | 11.30% |
| ROUGE-L | 0.2602 | 26.02% |

### Training Configuration Summary

| Configuration | Value |
| ------------- | ----- |
| Model | `microsoft/phi-2` |
| Dataset | XSum |
| Method | QLoRA |
| Training Samples | 3,000 |
| Validation Samples | 500 |
| Test Samples | 200 |
| Epochs | 2 |
| Learning Rate | 2e-4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Quantization | 4-bit NF4 |
| Output Directory | `./phi2-xsum-qlora` |

### Output Files

| File | Description |
| ---- | ----------- |
| `outputs/rouge_distribution.png` | Visualisasi distribusi skor ROUGE |
| `reports/result_summary.md` | Ringkasan hasil eksperimen |
| `notebooks/PHI2_XSUM_LORA_SUMMARIZATION.ipynb` | Notebook utama fine-tuning Phi-2 |
| `requirements.txt` | Dependency project |

---

## Project 3: Fine-Tuning T5 for Question Answering

### Project Location

```text
finetuning-t5-question-answering/
```

### Brief Overview

Project ini menerapkan fine-tuning model **T5** untuk tugas **question answering**. Model menerima input berupa konteks dan pertanyaan, kemudian menghasilkan jawaban dalam bentuk teks.

Task ini berbeda dari klasifikasi karena model tidak memilih label kelas, tetapi menghasilkan sequence jawaban. Oleh karena itu, pendekatan yang digunakan adalah text-to-text learning, sesuai dengan arsitektur T5.

### Model Description

T5 atau **Text-to-Text Transfer Transformer** adalah model encoder-decoder yang menyusun berbagai task NLP ke dalam format text-to-text. Pada project ini, input dapat diformat sebagai gabungan context dan question, lalu output berupa answer.

Pipeline utama project ini meliputi:

1. load dataset question answering;
2. preprocessing context, question, dan answer;
3. tokenisasi input-output;
4. fine-tuning model T5;
5. evaluasi hasil prediksi;
6. penyimpanan metrics;
7. penyimpanan sample predictions.

### Metrics Results

Metrik detail disimpan pada file:

```text
finetuning-t5-question-answering/outputs/metrics.json
```

Ringkasan metrik numerik belum tersedia pada prompt ini, sehingga bagian ini perlu disesuaikan berdasarkan isi file `metrics.json`.

| Metric | Score |
| ------ | ----: |
| Exact Match | BELUM ADA DATA |
| F1 Score | BELUM ADA DATA |
| Evaluation Loss | BELUM ADA DATA |

### Output Files

| File | Description |
| ---- | ----------- |
| `outputs/metrics.json` | File metrik evaluasi akhir |
| `outputs/sample_predictions.csv` | Contoh hasil prediksi model |
| `reports/result_summary.md` | Ringkasan hasil eksperimen |
| `notebooks/T5_SQUAD_QUESTION_ANSWERING.ipynb` | Notebook utama fine-tuning T5 |
| `requirements.txt` | Dependency project |

---

## How to Navigate the Repository

### 1. Start from this README

Baca file `README.md` pada root folder `UAS-DL` untuk memahami struktur umum repository dan daftar project.

### 2. Choose a Project Folder

Masuk ke salah satu folder project:

```bash
cd finetuning-distilbert-text-classification
```

atau:

```bash
cd finetuning-phi-2-text-summarization
```

atau:

```bash
cd finetuning-t5-question-answering
```

### 3. Read the Project README

Setiap folder project memiliki file `README.md` yang menjelaskan detail project tersebut.

### 4. Open the Notebook

Notebook utama berada di folder `notebooks/`.

Contoh:

```text
notebooks/DISTILBERT_AG_NEWS.ipynb
notebooks/PHI2_XSUM_LORA_SUMMARIZATION.ipynb
notebooks/T5_SQUAD_QUESTION_ANSWERING.ipynb
```

Notebook dapat dibuka menggunakan:

- Jupyter Notebook;
- JupyterLab;
- Google Colab;
- Kaggle Notebook;
- VS Code dengan extension Jupyter.

### 5. Check Outputs and Reports

Setelah membuka notebook, cek folder:

```text
outputs/
reports/
```

Folder `outputs/` berisi hasil eksperimen seperti metrics, visualisasi, dan prediksi. Folder `reports/` berisi ringkasan hasil dalam format Markdown.

---

## How to Run

Clone repository:

```bash
git clone https://github.com/heekal/ML-DL.git
cd ML-DL/UAS-DL
```

Pilih salah satu project:

```bash
cd finetuning-distilbert-text-classification
```

Install dependency:

```bash
pip install -r requirements.txt
```

Buka notebook:

```bash
jupyter notebook notebooks/DISTILBERT_AG_NEWS.ipynb
```

Untuk environment Google Colab, upload notebook ke Colab lalu jalankan cell dari atas ke bawah.

---

## Recommended Execution Environment

Project ini direkomendasikan dijalankan pada:

- Google Colab;
- Kaggle Notebook;
- local machine dengan GPU;
- Python 3.10 atau lebih baru.

Untuk project Transformer dan LLM, GPU sangat disarankan karena training di CPU akan sangat lambat.

---

## Summary of Models and Results

| Project | Model | Task | Main Metric | Result |
| ------- | ----- | ---- | ----------- | ------ |
| DistilBERT AG News | `distilbert-base-uncased` | Text Classification | Macro F1 | 0.9089 |
| Phi-2 XSum | `microsoft/phi-2` + QLoRA | Text Summarization | ROUGE-L | 0.2602 |
| T5 SQuAD | T5 | Question Answering | EM / F1 | BELUM ADA DATA |

---

## Notes

- Semua notebook dibuat untuk kebutuhan pembelajaran dan tugas UAS Deep Learning.
- Beberapa project menggunakan subset dataset agar proses training tetap praktis pada resource terbatas.
- Output numerik dapat berubah jika notebook dijalankan ulang dengan seed, subset, atau konfigurasi training yang berbeda.
- Untuk detail eksperimen, cek notebook dan `reports/result_summary.md` pada masing-masing project.
- Bagian group identification perlu diperbarui jika tugas dikumpulkan bersama anggota kelompok.

---

## Author

**Haikal Ali**  
Computer Engineering Student  
Class: TK-46-GAB  
NIM: 1103223071
