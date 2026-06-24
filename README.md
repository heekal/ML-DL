# ML-DL

> **Machine Learning and Deep Learning Coursework Repository**  
> Kumpulan notebook pembelajaran, reproduce buku, enrichment task, UTS, dan UAS untuk topik Machine Learning, Deep Learning, Statistik, Aljabar Linear, NLP, Transformer fine-tuning, model tracking, dan experiment artifact management.

---

## 📌 Overview

Repository ini berisi kumpulan notebook berbasis **Jupyter Notebook / Google Colab** untuk mendukung pembelajaran **Machine Learning** dan **Deep Learning** dari level dasar sampai implementasi proyek akhir.

Isi repository tidak hanya berfokus pada training model, tetapi juga mencakup fondasi penting dan workflow praktis seperti:

- statistik praktis untuk data science,
- aljabar linear untuk machine learning,
- supervised dan unsupervised learning,
- preprocessing dan feature engineering,
- evaluasi dan improvement model,
- pipeline dan model deployment dasar,
- deep learning dari nol,
- CNN, RNN, LSTM, NLP, dan federated learning,
- Transformer fine-tuning untuk text classification, summarization, dan question answering,
- MLflow experiment tracking,
- Optuna hyperparameter tuning,
- model artifact saving,
- proyek UTS Machine Learning,
- proyek UTS Deep Learning,
- proyek UAS Machine Learning,
- proyek UAS Deep Learning.

Repository ini dibuat sebagai dokumentasi pembelajaran dan coursework untuk membangun pemahaman teori, praktik coding, dan portofolio project berbasis notebook.

---

## 🛠️ Tech Stack

Tools dan library utama yang digunakan di repository ini:

### Core Data Science

- **Python**
- **Jupyter Notebook / Google Colab**
- **NumPy**
- **pandas**
- **SciPy**
- **matplotlib**
- **scikit-learn**

### Machine Learning

- **LightGBM**
- **XGBoost**
- **Optuna**
- **MLflow**
- **joblib**
- **skops**

### Deep Learning

- **PyTorch**
- **TensorFlow / Keras**
- **PyTorch TabNet**
- **Autoencoder**
- **MLP**
- **CNN**
- **RNN / LSTM**

### NLP and Transformer Fine-Tuning

- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **Evaluate**
- **PEFT**
- **LoRA / QLoRA**
- **BitsAndBytes**
- **TRL**
- **ROUGE**
- **DistilBERT**
- **Phi-2**
- **T5**

### Supporting Tools

- **UMAP**
- **phe** untuk eksperimen homomorphic encryption
- **requirements.txt** untuk dependency project
- **CSV / JSON / Keras / joblib artifacts** untuk output eksperimen

---

## 📁 Global File Structure

```text
ML-DL/
├── README.md
│
├── GrokkingDeepLearning/
│   ├── BuildingYourFirstDeepNeuralNetworkIntroductionToBackpropagation.ipynb
│   ├── DeepLearningOnUseenDataIntroducingFederatedLearning.ipynb
│   ├── FundamentalConceptsHowDoMachinesLearn.ipynb
│   ├── HowToPictureNeuralNetworksInYourHeadAndOnPaper.ipynb
│   ├── IntroducingAutomaticOptimizationLetsBuildADeepLearningFramework.ipynb
│   ├── IntroducingDeepLearningWhyYouShouldLearnItColab.ipynb
│   ├── IntroductionToNeuralLearningGradientDescent.ipynb
│   ├── IntroductionToNeuralPredictionForwardPropagation.ipynb
│   ├── LearningMultipleWeightsAtATimeGeneralizingGradientDescent.ipynb
│   ├── LearningSignalAndIgnoringNoiseIntroductionToRegularizationAndBatching.ipynb
│   ├── LearningToWriteLikeShakespeareLongShortTermMemory.ipynb
│   ├── ModelingProbabilitiesAndNonlinearitiesActivationFunctions.ipynb
│   ├── NeuralLearningAboutEdgesAndCornersIntroToConvolutionalNeuralNetworks.ipynb
│   ├── NeuralNetworksThatUnderstandLanguageKingManWoman.ipynb
│   ├── NeuralNetworksThatWriteLikeShakespeareRecurrentLayersForVariableLengthData.ipynb
│   ├── WhereToGoFromHereABriefGuide.ipynb
│   └── README.md
│
├── IntroductionMachineLearningwithpython/
│   ├── Introduction.ipynb
│   ├── SupervisedLearning.ipynb
│   ├── UnsupervisedLearningandPreprocessing.ipynb
│   ├── RepresentingDataandEngineeringFeatures.ipynb
│   ├── ModelEvaluationandImprovement.ipynb
│   ├── AlgorithmChainsandPipelines.ipynb
│   ├── WorkingwithTextData.ipynb
│   ├── WrappingUp.ipynb
│   └── README.md
│
├── Practical-Statistics-for-Data-Scientist-Books/
│   ├── Classification.ipynb
│   ├── Data_and_Sampling_Distributions_Colab.ipynb
│   ├── Exploratory_Data_Analysis.ipynb
│   ├── Regression_and_Prediction.ipynb
│   ├── Statistical_Experiments_and_Significance_Testing.ipynb
│   ├── Unsupervised_Learning.ipynb
│   └── README.md
│
├── PracticalLinearAlgebra/
│   ├── Introduction.ipynb
│   ├── Python_Tutorial.ipynb
│   ├── Vectors,_Part_1.ipynb
│   ├── Vectors,_Part_2.ipynb
│   ├── Vector_Applications.ipynb
│   ├── Matrices,_Part_1.ipynb
│   ├── Matrices,_Part_2.ipynb
│   ├── Matrix_Applications.ipynb
│   ├── Matrix_Inverse.ipynb
│   ├── Row_Reduction_dan_LU_Decomposition.ipynb
│   ├── Orthogonal_Matrices_and_QR_Decomposition.ipynb
│   ├── General_Linear_Models_dan_Least_Squares.ipynb
│   ├── Least_Squares_Applications.ipynb
│   ├── Eigendecomposition.ipynb
│   ├── Eigendecomposition_and_SVD_Applications.ipynb
│   ├── Singular_Value_Decomposition.ipynb
│   └── README.md
│
├── scikit-learn-cookbook/
│   ├── Common_Conventions_and_API_Elements_of_scikit_learn.ipynb
│   ├── Pre_Model_Workflow_and_Data_Preprocessing.ipynb
│   ├── Dimensionality_Reduction_Techniques.ipynb
│   ├── Building_Models_with_Distance_Metrics_and_Nearest_Neighbors.ipynb
│   ├── Linear_Models_and_Regularization.ipynb
│   ├── Advanced_Logistic_Regression_and_Extensions.ipynb
│   ├── Support_Vector_Machines_dan_Kernel_Methods.ipynb
│   ├── Tree_Based_Algorithms_and_Ensemble_Methods.ipynb
│   ├── Text_Processing_and_Multiclass_Classification.ipynb
│   ├── Clustering_Techniques.ipynb
│   ├── Novelty_and_Outlier_Detection.ipynb
│   ├── Cross_Validation_and_Model_Evaluation_Techniques.ipynb
│   ├── Deploying_scikit_learn_Models_in_Production.ipynb
│   └── README.md
│
├── UTS-ML/
│   ├── UTS_ML_Clustering.ipynb
│   ├── UTS_ML_REGRESI.ipynb
│   ├── UTS_ML_TransactionDataset.ipynb
│   └── README.md
│
├── UTS-DL/
│   ├── UTS_DL_Clustering.ipynb
│   ├── UTS_DL_REGRESI.ipynb
│   ├── UTS_DL_TransactionDataset.ipynb
│   └── README.md
│
├── UAS-ML/
│   ├── README.md
│   ├── UAS-ML-FRAUDTRANSACTION/
│   │   ├── UAS_ML_FRAUD_TRANSACTION.ipynb
│   │   ├── mlflow_tracking.db
│   │   ├── artifacts/
│   │   ├── final/artifacts/
│   │   └── mlflow_artifacts/
│   └── UAS-ML-REGRESSION/
│       ├── UAS_ML_REGRESSION.ipynb
│       ├── mlflow_song_year.db
│       ├── song_year_artifacts.zip
│       ├── mlflow_song_year_artifacts/
│       └── song_year_artifacts/
│
└── UAS-DL/
    ├── finetuning-distilbert-text-classification/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── notebooks/
    │   │   └── DISTILBERT_AG_NEWS.ipynb
    │   ├── outputs/
    │   │   ├── confusion_matrix.png
    │   │   └── metrics.json
    │   └── reports/
    ├── finetuning-phi-2-text-summarization/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── notebooks/
    │   │   └── PHI2_XSUM_LORA_SUMMARIZATION.ipynb
    │   ├── outputs/
    │   └── reports/
    │       └── result_summary.md
    └── finetuning-t5-question-answering/
        ├── README.md
        ├── requirements.txt
        ├── notebooks/
        │   └── T5_SQUAD_QUESTION_ANSWERING.ipynb
        ├── outputs/
        │   ├── metrics.json
        │   ├── result_summary.md
        │   └── sample_predictions.csv
        └── reports/
            └── result_summary.md
```

---

## 🧭 Repository Breakdown

| Folder | Fokus | Isi Utama |
| --- | --- | --- |
| `IntroductionMachineLearningwithpython` | Dasar Machine Learning | Recreate buku *Introduction to Machine Learning with Python* berisi supervised learning, unsupervised learning, preprocessing, feature engineering, model evaluation, pipelines, dan text data. |
| `scikit-learn-cookbook` | Enrichment Machine Learning | Reproduksi dan theoretical deep-dive dari *scikit-learn Cookbook*, mencakup API scikit-learn, preprocessing, dimensionality reduction, KNN, linear models, logistic regression, SVM, tree ensemble, NLP, clustering, outlier detection, cross-validation, dan deployment. |
| `Practical-Statistics-for-Data-Scientist-Books` | Statistik Data Science | Notebook statistik praktis seperti sampling distribution, exploratory data analysis, regression, significance testing, classification, dan unsupervised learning. |
| `PracticalLinearAlgebra` | Fondasi Matematika | Recreate *Practical Linear Algebra for Data Science* dengan Python, mencakup vectors, matrices, inverse, QR decomposition, LU decomposition, least squares, eigendecomposition, SVD, PCA, dan aplikasi data science. |
| `GrokkingDeepLearning` | Deep Learning dari Nol | Recreate *Grokking Deep Learning* dalam Colab, mulai dari neural prediction, gradient descent, backpropagation, regularization, CNN, NLP, RNN, LSTM, mini framework, dan federated learning. |
| `UTS-ML` | Proyek UTS Machine Learning | Tiga notebook: clustering pelanggan kartu kredit, regresi tahun rilis lagu, dan klasifikasi fraud transaction. |
| `UTS-DL` | Proyek UTS Deep Learning | Tiga notebook: deep clustering dengan autoencoder, regresi MLP PyTorch, dan fraud detection dengan TabNet. |
| `UAS-ML` | Proyek UAS Machine Learning | Dua project akhir: fraud transaction classification dan song year regression dengan artifact management, MLflow tracking, model output, dan file eksperimen. |
| `UAS-DL` | Proyek UAS Deep Learning | Tiga project fine-tuning NLP: DistilBERT untuk text classification, Phi-2 untuk summarization, dan T5 untuk question answering. |

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

4. **Proyek Machine Learning awal**
   - `UTS-ML`

5. **Deep Learning dari nol**
   - `GrokkingDeepLearning`

6. **Proyek Deep Learning awal**
   - `UTS-DL`

7. **Machine Learning end-to-end**
   - `UAS-ML`

8. **Transformer fine-tuning dan NLP**
   - `UAS-DL`

---

## 🧪 Project Output and Artifacts

Repository ini tidak hanya menyimpan notebook, tetapi juga beberapa artifact hasil eksperimen:

| Jenis Artifact | Contoh File | Fungsi |
| --- | --- | --- |
| Model Keras | `.keras` | Menyimpan model deep learning final |
| Preprocessor | `.joblib` | Menyimpan preprocessing object untuk inference |
| MLflow Database | `.db` | Menyimpan tracking eksperimen |
| MLflow Model Artifact | `MLmodel`, `conda.yaml`, `requirements.txt` | Menyimpan metadata model |
| Metrics | `.json`, `.csv` | Menyimpan hasil evaluasi model |
| Predictions | `.csv` | Menyimpan hasil prediksi test set |
| Visualization | `.png` | Menyimpan confusion matrix atau plot evaluasi |
| Report | `.md` | Menyimpan ringkasan hasil eksperimen |
| Config | `.json` | Menyimpan konfigurasi eksperimen |

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

Untuk project UAS, masuk ke folder project tertentu:

```bash
cd UAS-DL/finetuning-distilbert-text-classification
```

Install dependency:

```bash
pip install -r requirements.txt
```

Lalu buka notebook pada folder `notebooks/`.

---

## 📦 Recommended Environment

Minimal environment:

```bash
pip install numpy pandas scipy matplotlib scikit-learn jupyter
```

Untuk notebook machine learning lanjutan:

```bash
pip install lightgbm xgboost optuna mlflow joblib skops
```

Untuk notebook deep learning:

```bash
pip install torch tensorflow keras pytorch-tabnet umap-learn
```

Untuk notebook NLP dan Transformer fine-tuning:

```bash
pip install transformers datasets evaluate accelerate peft bitsandbytes trl rouge-score
```

Cek `requirements.txt` pada setiap folder project UAS untuk dependency yang lebih spesifik.

---

## 🎯 Who Is This For?

Repository ini cocok untuk:

- mahasiswa yang sedang mengambil mata kuliah Machine Learning dan Deep Learning,
- pemula yang ingin belajar ML/DL lewat notebook langsung jalan,
- pembelajar yang ingin memahami teori sambil praktik,
- calon AI Engineer, ML Engineer, Data Scientist, atau Research Engineer,
- orang yang ingin belajar dari reproduce notebook berbasis buku,
- orang yang ingin melihat contoh workflow UTS dan UAS berbasis eksperimen,
- siapa pun yang ingin membangun portofolio ML/DL berbasis project, notebook, dan artifact.

---

## 🧠 Notes

Repository ini berisi banyak notebook hasil recreate, enrichment, dan tugas akademik. Beberapa folder memiliki gaya README berbeda karena dibuat untuk konteks tugas yang berbeda.

README global ini berfungsi sebagai pintu masuk utama agar struktur repository lebih mudah dipahami. Untuk detail teknis spesifik, baca `README.md` pada masing-masing subfolder.

Beberapa project memiliki artifact hasil training seperti model, metrics, predictions, dan MLflow logs. Artifact tersebut digunakan untuk dokumentasi eksperimen dan reproduksi hasil.

---

## ⚖️ Copyright & Acknowledgments

Repository ini berisi notebook pembelajaran dan recreate dari beberapa sumber buku serta tugas akademik, termasuk:

- *Introduction to Machine Learning with Python* — Andreas C. Müller & Sarah Guido
- *Grokking Deep Learning* — Andrew W. Trask
- *Practical Linear Algebra for Data Science* — Mike X Cohen
- *scikit-learn Cookbook* — John Sukup
- materi statistik praktis untuk data science
- tugas UTS Machine Learning dan Deep Learning
- tugas UAS Machine Learning dan Deep Learning

Semua hak cipta atas buku, struktur materi asli, dan konten sumber tetap milik penulis serta penerbit masing-masing. Repository ini digunakan untuk tujuan edukasi, pembelajaran pribadi, dan dokumentasi tugas.

---

## 👤 Author

**Haikal Ali**  
Computer Engineering Student  
Machine Learning & Deep Learning Coursework Repository
