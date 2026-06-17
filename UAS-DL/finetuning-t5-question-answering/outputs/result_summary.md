# Result Summary

## Project

Fine-tuning T5 for SQuAD Question Answering

## Identity

Name: Haikal Ali  
Class: TK-46-GAB  
NIM: 1103223071  
Group: Fill in your group name  

## Dataset

The dataset used in this project is SQuAD from HuggingFace Datasets. Each sample contains a context paragraph, a question, and one or more reference answers.

A subset of the dataset was used to make the experiment feasible in Google Colab while preserving the full end-to-end training and evaluation workflow.

## Model

The model used is `t5-base`, an encoder-decoder Transformer model from the T5 family. T5 frames the question-answering task as text-to-text generation.

Input format:

```txt
question: <question> context: <context>
```

Target format:

```txt
<answer>
```

## Training Setup

| Configuration | Value |
|---|---:|
| Model | t5-base |
| Dataset | SQuAD |
| Train Samples | 2000 |
| Validation Samples | 400 |
| Test Samples | 200 |
| Epochs | 1 |
| Train Batch Size | 4 |
| Gradient Accumulation Steps | 4 |
| Learning Rate | 0.0003 |
| Max Source Length | 384 |
| Max Target Length | 48 |

## Evaluation Result

| Metric | Score |
|---|---:|
| Exact Match | 74.0000 |
| F1 Score | 85.6452 |
| Average Generated Length | 3.5800 |

## Analysis

Exact Match measures whether the generated answer exactly matches the reference answer after normalization. F1 is more flexible because it measures token-level overlap between the generated answer and the reference answer.

The error analysis shows that the model performs better when the answer is short and explicitly stated in the context. Lower scores usually happen when the model generates an incomplete answer, a paraphrased answer, or an answer that partially overlaps with the reference.

## Conclusion

This experiment demonstrates an end-to-end fine-tuning workflow for generative question answering using HuggingFace Transformers. The project covers dataset loading, preprocessing, tokenization, fine-tuning, generation, metric evaluation, and error analysis.