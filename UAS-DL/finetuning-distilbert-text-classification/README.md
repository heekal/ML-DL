# Fine-Tuning DistilBERT for AG News Text Classification

## Identity

Name: Haikal Ali  
Class: TK-46-GAB  
NIM: 1103223071  
Group: Kelompok 12

## Project Overview

This repository contains an end-to-end deep learning project for fine-tuning a Transformer encoder model using HuggingFace Transformers.

The project focuses on **multi-class text classification** using the **AG News** dataset. The model is trained to classify news articles into one of four categories: World, Sports, Business, or Sci/Tech.

## Task Description

This repository is created for Task 1 of the Deep Learning final-term assignment:

- Architecture type: Encoder
- Model family: BERT / DistilBERT / TinyBERT
- Dataset: AG News
- NLP problem: Text Classification

## Dataset

Dataset: AG News  
Source: HuggingFace Datasets  
Task type: Multi-class text classification  

The dataset contains news article texts and their corresponding topic labels.

## Model

Model used: `distilbert-base-uncased`

DistilBERT is used because it is part of the BERT-family encoder models but is more lightweight and efficient than full BERT. This makes it suitable for Google Colab and limited GPU environments.

## Pipeline

1. Environment setup
2. Dataset loading
3. Exploratory data analysis
4. Dataset sampling and splitting
5. Text tokenization
6. Model initialization
7. Fine-tuning
8. Evaluation
9. Confusion matrix visualization
10. Error analysis
11. Inference testing
12. Report generation

## Evaluation Metrics

The project evaluates the model using:

| Metric | Description |
|---|---|
| Accuracy | Overall prediction correctness |
| Macro Precision | Average precision across all classes |
| Macro Recall | Average recall across all classes |
| Macro F1 | Balanced F1 score across all classes |
| Weighted F1 | F1 score weighted by class support |

After running the notebook, the metric results will be saved in:

```text
outputs/metrics.json
```

## Repository Structure

```text
finetuning-distilbert-text-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── haikal_1103223071_distilbert_ag_news.ipynb
├── reports/
│   └── result_summary.md
└── outputs/
    └── .gitkeep
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```text
notebooks/haikal_1103223071_distilbert_ag_news.ipynb
```

Recommended environment:

- Google Colab
- GPU runtime
- T4 GPU or equivalent

## Output Files

The notebook generates:

```text
outputs/metrics.json
outputs/classification_report.txt
outputs/confusion_matrix.png
outputs/label_distribution.png
outputs/text_length_distribution.png
outputs/training_history.csv
outputs/test_predictions.csv
outputs/wrong_predictions.csv
reports/result_summary.md
```

## Notes

Model checkpoints and saved model weights are ignored by Git because they can be large. The repository is intended to store the notebook, report, requirements, and lightweight output summaries.

## Conclusion

This project demonstrates a complete HuggingFace fine-tuning workflow for a BERT-family encoder model. It includes preprocessing, tokenization, training, evaluation, visualization, error analysis, inference, and reporting.