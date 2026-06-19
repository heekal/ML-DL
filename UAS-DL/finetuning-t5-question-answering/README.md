# Fine-tuning T5 for SQuAD Question Answering

## Identity

Name: Haikal Ali  
Class: TK-46-GAB  
NIM: 1103223071  
Group: Kelompok 12 

## Project Overview

This repository contains an end-to-end deep learning project for fine-tuning a Transformer encoder-decoder model using HuggingFace. The project focuses on generative question answering using the SQuAD dataset.

The model receives a question and a context paragraph, then generates the answer as text. This is different from classification because the model must produce a sequence, not only predict a fixed label.

## Task Description

- Architecture type: Encoder-Decoder / Sequence-to-Sequence Transformer
- Model: `t5-base`
- Dataset: SQuAD
- NLP problem: Question Answering

## Dataset

Dataset: SQuAD  
Source: HuggingFace Datasets  
Task type: Extractive / Generative Question Answering  

Each sample contains:

- `context`: paragraph containing the answer
- `question`: question based on the context
- `answers`: one or more reference answers

For practical training on Google Colab, this project uses a controlled subset of the dataset. The subset size can be changed from the configuration cell inside the notebook.

## Model

Model used: `t5-base`

T5 is an encoder-decoder Transformer that reformulates NLP tasks as text-to-text generation. In this project, the input text is formatted as:

```txt
question: <question> context: <context>
```

The target text is the answer.

## Pipeline

1. Install and import required libraries
2. Load the SQuAD dataset from HuggingFace
3. Inspect dataset structure and basic statistics
4. Format question-answering examples into a text-to-text format
5. Tokenize input text and target answers
6. Fine-tune `t5-base`
7. Generate answers on validation/test examples
8. Evaluate results using Exact Match and token-level F1
9. Save metrics and prediction samples
10. Write result summary

## Evaluation Metrics

| Metric | Description |
|---|---|
| Exact Match | Measures whether the generated answer exactly matches the reference answer after normalization |
| F1 Score | Measures token-level overlap between generated and reference answer |
| Average Generated Length | Measures average answer length produced by the model |

## Repository Structure

```txt
finetuning-t5-question-answering/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── haikal_1103223071_t5_squad_question_answering.ipynb
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

Open the notebook:

```txt
notebooks/haikal_1103223071_t5_squad_question_answering.ipynb
```

Recommended environment:

```txt
Google Colab with GPU runtime
```

Run the notebook from top to bottom. After training, the notebook automatically saves:

```txt
outputs/metrics.json
outputs/sample_predictions.csv
outputs/f1_distribution.png
outputs/metric_summary.png
reports/result_summary.md
```

## Notes on Computational Efficiency

The notebook uses subset-based training to make the experiment realistic for Google Colab. The configuration can be modified in the notebook:

```python
TRAIN_SAMPLES = 2000
VALID_SAMPLES = 400
TEST_SAMPLES = 200
```

This keeps the project reproducible while still demonstrating the full end-to-end fine-tuning pipeline.

## Result Summary

The final metrics are generated after running the notebook. The report will be saved automatically in:

```txt
reports/result_summary.md
```

## Conclusion

This project demonstrates how a pre-trained encoder-decoder Transformer can be fine-tuned for question answering using the HuggingFace ecosystem. The pipeline includes dataset preparation, tokenization, model training, generation, evaluation, and analysis.
