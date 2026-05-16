# scikit-learn Cookbook

**TUGAS 2 (ENRICHMENT FOR MACHINE LEARNING CLASSES) - INDIVIDUAL TASK** **Code Reproduction + Theoretical Deep-Dive from scikit-learn Cookbook**

## 📌 Project Objective
This repository contains the code reproduction and theoretical deep-dives based on the **"scikit-learn Cookbook" (Third Edition, 2025 by John Sukup, Packt Publishing)**. 

The primary goal of this repository is to deepen understanding and practical skills in implementing core Machine Learning concepts. Every chapter from the book has been reproduced into interactive Jupyter Notebooks, complete with structured theoretical explanations and summaries to provide a holistic understanding of both the *how* and the *why* behind each machine learning algorithm.

---

## 📖 Chapter Summaries

As per the task requirements, here is the general overview of every chapter covered in this repository:

* **Chapter 1: Common Conventions and API Elements of scikit-learn** Introduces the foundational design philosophy of scikit-learn, focusing on core API elements like estimators, transformers, pipelines, and best practices for hyperparameter tuning.
* **Chapter 2: Pre-Model Workflow and Data Preprocessing** Covers essential data preparation techniques, including handling missing data, scaling, encoding categorical variables, and building automated preprocessing pipelines.
* **Chapter 3: Dimensionality Reduction Techniques** Explores methods to reduce dataset complexity while preserving information, featuring practical implementations of PCA, LDA, and t-SNE for data visualization.
* **Chapter 4: Building Models with Distance Metrics and Nearest Neighbors** Dives into distance-based algorithms, specifically K-Nearest Neighbors (KNN), covering various distance metrics and hyperparameter tuning to optimize classification.
* **Chapter 5: Linear Models and Regularization** Discusses foundational linear models and how to prevent overfitting using regularization techniques such as Ridge, Lasso, and ElasticNet regression.
* **Chapter 6: Advanced Logistic Regression and Extensions** Expands on logistic regression by applying it to complex scenarios, including multiclass and multilabel classification strategies, along with proper evaluation metrics.
* **Chapter 7: Support Vector Machines and Kernel Methods** Explains the mechanics of SVMs, the application of different kernel functions for high-dimensional spaces, and how to effectively tune SVM parameters.
* **Chapter 8: Tree-Based Algorithms and Ensemble Methods** Covers powerful tree-based models, transitioning from simple Decision Trees to advanced ensemble methods like Random Forests and Gradient Boosting Machines (GBM).
* **Chapter 9: Text Processing and Multiclass Classification** Focuses on Natural Language Processing (NLP) workflows, including text vectorization, feature extraction (n-grams), and building text-based multiclass classifiers.
* **Chapter 10: Clustering Techniques** Explores unsupervised learning for grouping unlabeled data, detailing algorithms like K-Means, Hierarchical Clustering, and density-based DBSCAN.
* **Chapter 11: Novelty and Outlier Detection** Introduces techniques to identify anomalies and outliers in datasets using Isolation Forests, One-Class SVMs, and Local Outlier Factor (LOF).
* **Chapter 12: Cross-Validation and Model Evaluation Techniques** Details robust validation methods to ensure models generalize well to unseen data, utilizing advanced cross-validation and hyperparameter grid search strategies.
* **Chapter 13: Deploying scikit-learn Models in Production** Bridges the gap between development and production by covering model serialization, scaling, lifecycle management, and setting up deployment pipelines.

---

## 📂 Proposed Repository Structure

Each notebook inside the chapter folders contains the reproduced code, exercise solutions, and theoretical explanations.

```text
scikit-learn-cookbook/
├── APIConventions.ipynb
├── DataPreprocessing.ipynb
├── DimensionalityReduction.ipynb
├── KNNandDistance.ipynb
├── LinearModels.ipynb
├── LogisticRegression.ipynb
├── SVMandKernels.ipynb
├── TreeandEnsemble.ipynb
├── TextProcessing.ipynb
├── Clustering.ipynb
├── OutlierDetection.ipynb
├── ModelEvaluation.ipynb
├── Deployment.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 How to Use This Repository

Clone the repository:

```Bash
git clone [https://github.com/YOUR_USERNAME/scikit-learn-cookbook.git](https://github.com/YOUR_USERNAME/scikit-learn-cookbook.git)
cd scikit-learn-cookbook
```

Create and activate a virtual environment (Recommended):

```Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install dependencies:

```Bas
pip install -r requirements.txt
(Ensure your requirements.txt includes scikit-learn>=1.5, numpy, pandas, matplotlib, seaborn, and jupyter)
```

Launch Jupyter Notebook:

```Bash
jupyter notebook
```

---

## 🎓 Acknowledgments
Original Book Author: John Sukup
Publisher: Packt Publishing (2025)
Completed as a requirement for the Machine Learning Enrichment course task.
(Note: The assignment sheet mentions O'Reilly, but the provided book material is officially published by Packt Publishing).
