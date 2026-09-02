# Medical Image Captioning with Training-Free Multimodal RAG

A benchmarking pipeline for generating captions for radiology images with multimodal
foundation models, using retrieval-augmented generation (RAG) and **no fine-tuning**.
Given a query image, the pipeline retrieves visually similar images from a vector
database, injects their captions into the prompt as few-shot examples, and asks a
vision-language model for a caption.

This is the codebase behind the UFPE–Essex–UNED submission to **ImageCLEFmedical
Caption 2026**, where it placed **2nd** in Caption Prediction and **4th** in
Caption Prediction Synthetic, and behind our SPIE benchmark study of generative AI on
ROCOv2.

```
                 ┌──────────────────┐
  query image ──▶│ CLIP / MedSigLIP │──▶ top-k similar train images
                 └──────────────────┘             │
                                                  ▼
                 ┌───────────────────────────────────────────────┐
  prompt ───────▶│  domain prompt + k retrieved captions + image │──▶ VLM ──▶ caption
                 └───────────────────────────────────────────────┘
```

## Capabilities

| | |
|---|---|
| **Providers** | OpenAI, Google (Gemini/Gemma), Anthropic (Claude), DeepInfra (Llama 4), local HuggingFace models — all through LangChain |
| **Datasets** | ROCOv2 (auto-downloaded from HuggingFace), ImageCLEF 2026 Caption, ImageCLEF 2026 Caption Synthetic |
| **Retrieval** | FAISS index over CLIP ViT-B/32 or MedSigLIP embeddings; cached and reused across runs |
| **Prompting** | Three swappable prompts (`base`, `simple`, `specific`), selected by config |
| **Execution** | Multi-threaded API calls, incremental JSONL writes, resumable retry of failed samples |
| **Evaluation** | BLEU-1–4, ROUGE-1/2/L, METEOR, CIDEr, BERTScore, plus plots and a Markdown report |
| **Cost** | Per-run token and USD accounting with projections |
| **Submission** | JSON repair, error collection and multi-run merging into an ImageCLEF CSV |

## Installation

```bash
git clone https://github.com/MendesHeitor/experiments-clef.git
cd experiments-clef

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # then fill in the key for the provider you plan to use
```

Only the key matching your config's `model.provider` is needed: `OPENAI_API_KEY`,
`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY` or `DEEPINFRA_API_TOKEN`. Provider SDKs are
imported lazily, so you only need the package for the provider you actually use.

BERTScore and CIDEr are optional; without them the evaluator reports those metrics as
`n/a` rather than failing. To enable them: `pip install bert-score pycocoevalcap`.

## Quick start

ROCOv2 downloads automatically, so this needs no manual data setup:

```bash
python run_pipeline.py --config config_openai.yaml --samples 50 --evaluation
```

The first run builds the vector database, which is slow; it is cached under
`vectordb/` and reused by every later run with the same dataset, encoder and
retrieval-pool size.

```bash
# Switch provider by switching config file
python run_pipeline.py --config config_deepinfra.yaml
python run_pipeline.py --config config_google.yaml --model gemini-2.5-pro

# Zero-shot instead of RAG
python run_pipeline.py --config config_openai.yaml --rag-examples 0

# Re-run analysis on existing results, no API calls
python run_pipeline.py --evaluation-only responses/responses_rag_hf_<timestamp>.jsonl
```

Every CLI flag is optional and overrides the config file. Run
`python run_pipeline.py --help` for the full list; see
[CONFIGURATION.md](CONFIGURATION.md) for every YAML key.

### Retrying failed samples

Long runs lose samples to rate limits and provider errors. Collect the failures and
re-run only those; the results append to a new JSONL you can merge afterwards:

```bash
python process_submission.py collect-errors responses/run.jsonl -o error_image_ids.txt
python run_pipeline.py --config config_deepinfra.yaml --retry-ids error_image_ids.txt
```

### Building an ImageCLEF submission

```bash
python process_submission.py parse responses/run.jsonl -o processed_run1.csv
python process_submission.py merge processed_run*.csv -o merged_submission.csv
```

`parse` repairs model output that wrapped the caption in malformed or truncated JSON;
`merge` keeps the first non-error caption per image ID and sorts by numeric ID.

## Project structure

```
run_pipeline.py            CLI entry point
process_submission.py      Error collection, JSON repair, submission merging
config_*.yaml              One config per provider (openai, google, deepinfra, anthropic)
prompts/                   base | simple | specific prompt templates
src/
├── config.py              YAML config, cache paths, global config object
├── main.py                Pipeline orchestration and threaded generation
├── data/dataset.py        ROCOv2 (HuggingFace) and ImageCLEF (local) handlers
├── vectordb/              FAISS index over CLIP / MedSigLIP embeddings
├── llm/llm_utils.py       Provider factory, prompt assembly, token accounting
└── analysis/              Caption metrics, plots, reports, cost analysis
responses/                 Generated captions (JSONL, one line per sample)
evaluation_results/        Metrics, plots and Markdown reports
```

## Datasets

**ROCOv2** is pulled from HuggingFace (`eltorio/ROCOv2-radiology`) on first use — no
setup required. Use `dataset.type: rocov2`.

**ImageCLEF 2026** data is an extended version of ROCOv2, with additional and more
recent PMC-derived radiology images. It is distributed by the challenge organizers and
available on agreement and participation in the task, so it cannot be redistributed
here. Place it as follows and set `dataset.type` to `imageclef_natural` or
`imageclef_synth`:

```
data_img_clef/
├── natural/
│   ├── dev_caption/{images/, captions.csv}    # train + valid images
│   └── test/images/
└── synth/
    ├── dev_caption_synth/{images/, captions.csv}
    └── test_synth/images/
```

Splits are inferred from filenames (`ImageCLEFmedical_Caption_2026_{train,valid,test}_N`).
Caption Prediction ships 97,364 train / 19,240 validation / 15,249 test images;
the Synthetic subtask ships 97,222 / 19,239 / 15,262.

## Results

### ImageCLEFmedical Caption 2026

Our three submitted runs differ only in configuration, and each is reproducible by
editing the corresponding keys:

| Run | `max_train_samples` | `prompt.prefix` | `rag.model_name` | Overall | Rank |
|-----|--------------------|-----------------|------------------|---------|------|
| 1 | `10000` | `base` | `openai/clip-vit-base-patch32` | 0.3433 | 15th |
| 2 | `null` (full) | `specific` | `openai/clip-vit-base-patch32` | 0.3554 | 9th |
| 3 | `null` (full) | `specific` | `google/medsiglip-448` | **0.3603** | **2nd** |

All runs used Llama 4 Maverick via DeepInfra, `temperature: 0.1`, and `k=3` retrieved
examples. Expanding the retrieval pool and aligning the prompt with the dataset's
figure-caption style gave the largest gain; swapping CLIP for the medically adapted
MedSigLIP encoder added a further improvement.

Test-set scores are computed by the challenge organizers with the official
[CLEF caption evaluation](https://github.com/taubsity/clef-caption-evaluation-2026)
implementation (Relevance, Factuality, BERTScore, ROUGE, Similarity, BLEURT, MedCAT,
AlignScore). Those metrics are **not** reproduced by this repository; the built-in
evaluator reports the standard captioning metrics listed below.

### ROCOv2 benchmark (SPIE)

Retrieval consistently helps every model evaluated on the ROCOv2 test set (9,927 images):

| Model | BLEU-1 | ROUGE-L | METEOR | BERTScore F1 |
|-------|--------|---------|--------|--------------|
| GPT-4o | 0.126 | 0.129 | 0.199 | 0.523 |
| GPT-4o (RAG) | 0.208 | 0.189 | **0.275** | 0.578 |
| Gemini 2.5 Pro | 0.155 | 0.149 | 0.242 | 0.557 |
| Gemini 2.5 Pro (RAG) | 0.203 | 0.186 | 0.257 | 0.580 |
| Llama 4 Maverick | 0.167 | 0.162 | 0.225 | 0.552 |
| Llama 4 Maverick (RAG) | **0.218** | **0.198** | 0.233 | **0.583** |
| Claude 4.5 Sonnet (RAG) | 0.183 | 0.164 | 0.211 | 0.567 |

Reproduce a row with, for example:

```bash
python run_pipeline.py --config config_openai.yaml --model gpt-4o --rag-examples 3 --evaluation
```

## Related branches

Experimental work lives on unmerged branches:

| Branch | Contents |
|--------|----------|
| `ensemble` | Multi-LLM ensemble with a judge model selecting the best caption |
| `feature/judge-synthesis` | Judge-based synthesis of captions from several models |
| `Medgemma` | Running MedGemma locally instead of through a hosted API |
| `feat/evaluation` | Standalone official ImageCLEF 2025 caption evaluator |

## Citation

If you use this pipeline, please cite the benchmark study:

```bibtex
@inproceedings{10.1117/12.3084497,
  author    = {Heitor Mendes Pereira and Pedro Didier Maranh{\~a}o and
               Vitoria de Ara{\'u}jo Xavier and Tsang Ing Ren and Anoushka Duggal and
               Alba G. Seco de Herrera and Vahid Abolghasemi},
  title     = {{Evaluating generative AI for medical image captioning: a benchmark study on radiology images}},
  booktitle = {Medical Imaging 2026: Imaging Informatics},
  editor    = {Xiaofeng Yang and William Hsu},
  volume    = {13930},
  pages     = {139300F},
  publisher = {SPIE},
  organization = {International Society for Optics and Photonics},
  year      = {2026},
  doi       = {10.1117/12.3084497},
  url       = {https://doi.org/10.1117/12.3084497}
}
```

## License

MIT — see [LICENSE](LICENSE).
