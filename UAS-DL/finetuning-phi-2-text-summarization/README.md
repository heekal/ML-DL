# Fine-tuning Phi-2 for XSum Text Summarization

## Identity

Name: Haikal Ali  
Class: TK-46-GAB  
NIM: 1103223071  
Group: [fill your group name]

## Project Overview

This repository contains an end-to-end deep learning project for fine-tuning a decoder-only Large Language Model (LLM) using HuggingFace. The project focuses on abstractive text summarization using Microsoft Phi-2 and the XSum dataset.

The workflow covers dataset loading, exploratory data analysis, prompt formatting, tokenization, parameter-efficient fine-tuning using LoRA/QLoRA, generation, evaluation using ROUGE metrics, error analysis, and repository-level reporting.

## Task Description

This project corresponds to Task 3 of the Deep Learning finalterm assignment:

- Architecture type: Decoder-only LLM
- Model: `microsoft/phi-2`
- Dataset: XSum
- NLP problem: Text Summarization

XSum is an abstractive summarization dataset. The model is trained to generate a concise summary from a longer news article.

## Dataset

Dataset: XSum  
Source: HuggingFace Datasets  
Task type: Abstractive Text Summarization  

The dataset contains news documents paired with one-sentence summaries. To make the experiment feasible on free GPU environments, this project uses a controlled subset of the dataset.

## Model

Model used: `microsoft/phi-2`

Phi-2 is a decoder-only language model. Because full fine-tuning is too expensive for common student GPU environments, this project uses LoRA/QLoRA through PEFT and bitsandbytes.

## Method

This notebook uses instruction-style prompting:

```text
### Instruction:
Summarize the following news article in one concise sentence.

### Article:
...

### Summary:
...
```

Only the summary portion is used as the supervised training target. The prompt portion is masked using `-100` labels so the loss is computed only on the expected answer.

## Pipeline

1. Install stable dependencies
2. Load XSum dataset
3. Perform exploratory data analysis
4. Create instruction-style summarization prompts
5. Tokenize prompt-summary pairs
6. Load Phi-2 using 4-bit quantization
7. Attach LoRA adapters
8. Fine-tune using HuggingFace Trainer
9. Generate summaries on the test subset
10. Evaluate generated summaries using ROUGE
11. Save metrics and generated summaries
12. Generate result report

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROUGE-1 | Unigram overlap between generated and reference summaries |
| ROUGE-2 | Bigram overlap |
| ROUGE-L | Longest common subsequence similarity |
| Average Generated Length | Average number of generated words |

## Repository Structure

```text
finetuning-phi-2-text-summarization/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── haikal_1103223071_phi2_xsum_lora_summarization.ipynb
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

```text
notebooks/haikal_1103223071_phi2_xsum_lora_summarization.ipynb
```

Recommended environment:

- Google Colab or Kaggle Notebook
- GPU runtime
- T4 GPU or better
- Do not use TPU

## Troubleshooting

### torch_xla / accelerate error

If this error appears:

```text
AttributeError: module 'torch_xla.core.xla_model' has no attribute 'xrt_world_size'
```

The runtime has a TPU/XLA package conflict. The notebook uninstalls `torch-xla` in the first dependency cell. Use GPU runtime, not TPU.

### NumPy import error

If this error appears:

```text
ImportError: cannot import name '_center' from 'numpy._core.umath'
```

The runtime has a broken NumPy installation. The notebook pins NumPy to a stable version and restarts the runtime once.

### CUDA out of memory

Reduce these values in the configuration cell:

```python
TRAIN_SAMPLES = 100
VALID_SAMPLES = 20
TEST_SAMPLES = 20
MAX_LENGTH = 384
```

## Notes

The model checkpoint is not uploaded to GitHub because it can be large. The repository stores notebooks, reports, metrics, and sample generated summaries.
