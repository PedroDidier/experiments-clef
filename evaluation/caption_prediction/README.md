# Evaluation
The script computes the caption evaluation metrics on submission for ImageCLEF 2025.

This year, ranking of participants is based on an average score over all used metrics. In total 6 metrics are computed that fall into the aspects of relevance and factuality.

## Relevance
In order to evaluate the relevance aspect of generated captions the following metrics are used:

- Image and Caption Similarity
- BERT-Score (Recall) with inverse document frequency (idf) scores computed from the test corpus for importance weighting
- Recall-Oriented Understudy for Gisting Evaluation (ROUGE) for overlap of unigrams (ROUGE-1) (F-measure)
- Bilingual Evaluation Understudy with Representations from Transformers (BLEURT)

## Factuality

In order to evaluate the factuality aspect of generated captions the following metrics are used:

- Unified Medical Language System (UMLS) Concept F1
- AlignScore

## Usage
```bash
# Create a virtualenv
$ python -m venv .venv

# Install dependencies
$ pip install -r requirements.txt

# Run evaluation metrics on caption results 
$ python evaluator.py --input ./data/dummy_submission.csv --target ./data/dummy_captions.csv --output ./data/dummy_output.csv
```

