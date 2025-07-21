# RAG Pipeline Evaluation Report

**Evaluation Date:** 2025-07-21T13:41:41.492842  
**Source File:** responses/responses_rag_2025-07-21_12-47.jsonl

## Summary

This report presents the evaluation results for the Retrieval-Augmented Generation (RAG) pipeline for medical image captioning and concept extraction.

### Dataset Overview
- **Caption Samples Evaluated:** 300
- **Concept Samples Evaluated:** 300
- **Unique CUIs in Dataset:** 2367

## Caption Evaluation Results

### BLEU Score
- **Mean:** 0.0369
- **Standard Deviation:** 0.0510

### ROUGE Scores

| Metric | Precision | Recall | F-measure |
|--------|-----------|--------|-----------|
| ROUGE-1 | 0.2505 | 0.2257 | 0.2137 |
| ROUGE-2 | 0.0624 | 0.0595 | 0.0539 |
| ROUGE-L | 0.2039 | 0.1900 | 0.1768 |

## Concept Evaluation Results

### Classification Metrics

| Metric | Micro-averaged | Macro-averaged |
|--------|----------------|----------------|
| Precision | 0.0206 | 0.0071 |
| Recall | 0.0232 | 0.0068 |
| F1-Score | 0.0218 | 0.0054 |

### Additional Metrics
- **Exact Match Ratio:** 0.0000
- **Jaccard Score (Micro):** 0.0110
- **Jaccard Score (Macro):** 0.0040
- **Hamming Loss:** 0.0067

## Interpretation

### Caption Performance
- The BLEU score of 0.0369 indicates room for improvement in caption quality.
- ROUGE-1 F-measure of 0.2137 shows limited unigram overlap with reference captions.
- ROUGE-L F-measure of 0.1768 indicates limited structural similarity.

### Concept Performance
- Micro-averaged F1 of 0.0218 shows limited overall concept extraction performance.
- Exact match ratio of 0.0000 indicates that 0.0% of predictions exactly match the ground truth concept sets.
- Hamming loss of 0.0067 shows the fraction of labels that are incorrectly predicted.

## Recommendations

Based on the evaluation results:

1. **Caption Quality:** Consider improving caption generation with more diverse training examples or fine-tuning
2. **Concept Extraction:** Consider improving concept extraction with better CUI mapping or additional training data
3. **Overall System:** Consider refinements to the RAG retrieval and generation components

---
*Report generated automatically by RAG Evaluation System*
