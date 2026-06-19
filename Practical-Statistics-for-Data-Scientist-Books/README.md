# Practical Statistics for Data Scientists

**TUGAS ENRICHMENT FOR MACHINE LEARNING CLASSES - INDIVIDUAL TASK**  
**Code Reproduction + Theoretical Deep-Dive from Practical Statistics for Data Scientists**

## 📌 Project Objective

This repository contains code reproduction and theoretical deep-dive notebooks based on the book **"Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python" (Second Edition, 2020) by Peter Bruce, Andrew Bruce, and Peter Gedeck, published by O'Reilly Media**.

The primary goal of this repository is to strengthen statistical understanding for data science and machine learning practice. The notebooks reproduce important statistical concepts from the book using Python-based workflows, supported by explanations, summaries, and practical examples.

This repository focuses on the connection between statistics and data science practice. Instead of treating statistics only as theory, each notebook explains how statistical concepts are used in real data workflows such as exploratory data analysis, sampling, hypothesis testing, regression, classification, statistical machine learning, and unsupervised learning.

---

## 📖 Book Overview

**Practical Statistics for Data Scientists** is designed for data scientists who already have basic familiarity with R and/or Python and some exposure to statistics. The book explains statistical concepts that are relevant to modern data science and clarifies which methods are important, useful, or commonly misused in real-world practice.

The book highlights several key topics:

- exploratory data analysis as a preliminary step in data science;
- random sampling and sample bias;
- sampling distributions and bootstrap methods;
- experimental design and significance testing;
- regression for prediction and anomaly detection;
- classification techniques;
- statistical machine learning methods;
- unsupervised learning for extracting meaning from unlabeled data.

---

## 📖 Chapter Summaries

As per the task requirements, here is the general overview of every chapter from the book and its notebook availability in this repository:

| Num | Chapter | Description | Notebook |
| --- | ------- | ----------- | -------- |
| 1 | Chapter 1: Exploratory Data Analysis | Introduces the foundations of EDA, including structured data, estimates of location, variability, distributions, categorical data, correlation, scatterplots, and visualization of multiple variables. | `Exploratory_Data_Analysis.ipynb` |
| 2 | Chapter 2: Data and Sampling Distributions | Covers random sampling, sample bias, selection bias, sampling distributions, central limit theorem, standard error, bootstrap, confidence intervals, and common probability distributions. | `Data_and_Sampling_Distributions_Colab.ipynb` |
| 3 | Chapter 3: Statistical Experiments and Significance Testing | Explains A/B testing, hypothesis tests, null and alternative hypotheses, permutation tests, p-values, t-tests, multiple testing, ANOVA, chi-square tests, multi-arm bandits, power, and sample size. | `Statistical_Experiments_and_Significance_Testing.ipynb` |
| 4 | Chapter 4: Regression and Prediction | Discusses simple and multiple linear regression, fitted values, residuals, least squares, prediction intervals, categorical variables, multicollinearity, interactions, regression diagnostics, polynomial regression, splines, and generalized additive models. | `Regression_and_Prediction.ipynb` |
| 5 | Chapter 5: Classification | Covers Naive Bayes, discriminant analysis, logistic regression, model assessment, confusion matrix, rare class problems, precision, recall, specificity, ROC curve, AUC, lift, and strategies for imbalanced data. | `Classification.ipynb` |
| 6 | Chapter 6: Statistical Machine Learning | Covers K-Nearest Neighbors, distance metrics, one-hot encoding, standardization, tree models, bagging, random forest, variable importance, boosting, XGBoost, regularization, hyperparameters, and cross-validation. | **BELUM ADA NOTEBOOK DI STRUKTUR SAAT INI** |
| 7 | Chapter 7: Unsupervised Learning | Explores PCA, correspondence analysis, K-Means clustering, hierarchical clustering, model-based clustering, scaling, categorical variables, and mixed-data clustering problems. | `Unsupervised_Learning.ipynb` |

---

## 📂 Proposed Repository Structure

Each notebook in this repository contains reproduced code, theoretical explanation, chapter summary, and practical notes based on the corresponding chapter.

```text
Practical-Statistics-for-Data-Scientist-Books/
├── Classification.ipynb
├── Data_and_Sampling_Distributions_Colab.ipynb
├── Exploratory_Data_Analysis.ipynb
├── README.md
├── Regression_and_Prediction.ipynb
├── Statistical_Experiments_and_Significance_Testing.ipynb
└── Unsupervised_Learning.ipynb
```

---

## 🧭 Notebook Navigation Guide

| Notebook | Main Topic | Recommended Order |
| -------- | ---------- | ----------------: |
| `Exploratory_Data_Analysis.ipynb` | Descriptive statistics, distribution analysis, correlation, and data visualization | 1 |
| `Data_and_Sampling_Distributions_Colab.ipynb` | Sampling, bias, bootstrap, confidence intervals, and distributions | 2 |
| `Statistical_Experiments_and_Significance_Testing.ipynb` | A/B testing, hypothesis testing, p-values, ANOVA, and chi-square tests | 3 |
| `Regression_and_Prediction.ipynb` | Linear regression, prediction, diagnostics, and regression extensions | 4 |
| `Classification.ipynb` | Classification methods and evaluation metrics | 5 |
| `Unsupervised_Learning.ipynb` | PCA, clustering, scaling, and unsupervised analysis | 6 |

Recommended learning path:

1. Start with **Exploratory Data Analysis** to understand how to inspect and summarize data.
2. Continue with **Data and Sampling Distributions** to understand sampling behavior and uncertainty.
3. Study **Statistical Experiments and Significance Testing** to understand decision-making from experiments.
4. Move to **Regression and Prediction** to learn statistical modeling for numerical prediction.
5. Continue with **Classification** to understand category prediction and classification evaluation.
6. Finish with **Unsupervised Learning** to explore dimensionality reduction and clustering.

---

## 🧪 Topics Covered

This repository covers several essential statistical concepts for data science:

### Exploratory Data Analysis

- structured data;
- data frames and indexes;
- mean, median, and robust location estimates;
- standard deviation and percentile-based variability;
- boxplots, histograms, and density plots;
- categorical data analysis;
- correlation and scatterplots;
- multivariable visualization.

### Sampling and Distributions

- random sampling;
- sample bias;
- selection bias;
- regression to the mean;
- sampling distribution;
- central limit theorem;
- standard error;
- bootstrap;
- confidence interval;
- normal distribution;
- long-tailed distribution;
- t-distribution;
- binomial distribution;
- chi-square distribution;
- F-distribution;
- Poisson, exponential, and Weibull distributions.

### Statistical Testing

- A/B testing;
- control group design;
- hypothesis testing;
- null and alternative hypotheses;
- permutation test;
- statistical significance;
- p-value;
- Type I and Type II errors;
- t-test;
- multiple testing;
- ANOVA;
- chi-square test;
- Fisher's exact test;
- multi-arm bandit;
- power and sample size.

### Regression

- simple linear regression;
- multiple linear regression;
- least squares;
- fitted values and residuals;
- prediction vs explanation;
- cross-validation;
- model selection;
- weighted regression;
- prediction intervals;
- dummy variables;
- multicollinearity;
- confounding variables;
- interactions;
- regression diagnostics;
- polynomial regression;
- splines;
- generalized additive models.

### Classification

- Naive Bayes;
- discriminant analysis;
- logistic regression;
- generalized linear models;
- confusion matrix;
- rare class problem;
- precision;
- recall;
- specificity;
- ROC curve;
- AUC;
- lift;
- imbalanced data handling.

### Unsupervised Learning

- principal components analysis;
- correspondence analysis;
- K-Means clustering;
- hierarchical clustering;
- dendrograms;
- model-based clustering;
- scaling;
- categorical variables;
- Gower's distance;
- mixed-data clustering problems.

---

## 🚀 How to Use This Repository

Clone the repository:

```bash
git clone https://github.com/heekal/ML-DL.git
cd ML-DL/Practical-Statistics-for-Data-Scientist-Books
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the common dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open any notebook and run the cells from top to bottom.

---

## 📦 Recommended Environment

Recommended environment:

| Component | Recommendation |
| --------- | -------------- |
| Python | Python 3.10 or newer |
| Notebook | Jupyter Notebook / JupyterLab / Google Colab |
| Main Libraries | NumPy, pandas, SciPy, matplotlib, seaborn, scikit-learn, statsmodels |
| Optional Platform | Google Colab for easier setup |

Minimal installation:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels jupyter
```

---

## 📌 Notes

- This repository is intended for educational use and coursework documentation.
- The notebooks are based on chapter-level concepts from the book, but written as practical Python notebooks.
- Some original book examples include both R and Python. This repository focuses on Python-based implementation.
- Chapter 6, **Statistical Machine Learning**, appears in the book but is not present as a notebook in the current folder structure.
- The notebooks should be read in chapter order for the best learning flow.

---

## ⚖️ Copyright & Acknowledgments

Original Book:

**Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python**  
Second Edition  
Authors: Peter Bruce, Andrew Bruce, and Peter Gedeck  
Publisher: O'Reilly Media  
Copyright: 2020  
ISBN: 978-1-492-07294-2

Completed as a requirement for the Machine Learning Enrichment course task.

All rights for the original book, examples, structure, and related materials belong to the authors and publisher. This repository is used for educational purposes, personal learning, and academic coursework documentation.