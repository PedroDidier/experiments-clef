# Configuration Guide

The pipeline is driven by a YAML config file. Four are provided, one per provider:
`config_openai.yaml`, `config_google.yaml`, `config_deepinfra.yaml` and
`config_anthropic.yaml`. Select one with `--config`:

```bash
python run_pipeline.py --config config_deepinfra.yaml
```

## Precedence

1. Command-line flags (highest)
2. The file passed to `--config`
3. `config_local.yaml`, then `config.yaml`, if present and no `--config` is given
4. Built-in defaults (lowest)

`config_local.yaml` is gitignored — copy any provided config to it for personal
settings you do not want to commit.

> If you import the pipeline directly rather than through `run_pipeline.py`, the
> global config defaults to `config_deepinfra.yaml`. Call
> `src.config.set_config("<file>")` before constructing the pipeline to change it.

## Environment variables

Create `.env` from `.env.example`. Only the key for your provider is required:

| Variable | Needed when |
|----------|-------------|
| `OPENAI_API_KEY` | `provider: openai` |
| `GOOGLE_API_KEY` | `provider: google` |
| `ANTHROPIC_API_KEY` | `provider: anthropic` |
| `DEEPINFRA_API_TOKEN` | `provider: deepinfra` |
| `HF_HOME` | optional; overrides the HuggingFace cache location |

## YAML reference

### `model`

| Key | Description |
|-----|-------------|
| `provider` | `openai`, `google`, `deepinfra`, `anthropic` or `huggingface` (local) |
| `name` | Model identifier passed to the provider. Must be vision-capable |
| `temperature` | Sampling temperature; `0.1` was used for all published runs |
| `max_tokens` | Maximum output tokens. Set generously to avoid truncated captions |

### `prompt`

| Key | Description |
|-----|-------------|
| `prefix` | Loads `prompts/<prefix>_prompt.txt`. One of `base`, `simple`, `specific` |

- `base` — radiology-specialist framing with step-by-step instructions. Used for the
  SPIE benchmark and ImageCLEF Run 1.
- `specific` — refined variant that targets the descriptive figure-caption style of
  biomedical articles rather than full diagnostic reporting. Used for Runs 2 and 3.
- `simple` — minimal instruction, useful as a prompting-sensitivity baseline.

All three require the model to answer as `{"caption": "..."}`.

### `dataset`

| Key | Description |
|-----|-------------|
| `type` | `rocov2`, `imageclef_natural` or `imageclef_synth` |
| `name` | HuggingFace dataset id, used only when `type: rocov2` |
| `max_train_samples` | Cap on the retrieval pool. `null` uses the full training set |
| `validation_samples` | Number of samples to process |
| `previous_validation_samples` | Samples to exclude, to get a disjoint set from an earlier run |
| `random_seed` | Seed for sampling |

`max_train_samples` is part of the vector-database cache key, so changing it builds a
separate index rather than silently reusing the previous one.

### `rag`

| Key | Description |
|-----|-------------|
| `num_examples` | Retrieved examples per query. `3` in all published runs; `0` disables retrieval |
| `model_name` | Retrieval encoder: `openai/clip-vit-base-patch32` or `google/medsiglip-448` |

The encoder is also part of the vector-database cache key. MedSigLIP is a medically
adapted SigLIP variant and retrieves more clinically relevant neighbours than
general-purpose CLIP.

### `output`

| Key | Default |
|-----|---------|
| `responses_dir` | `responses` |
| `vectordb_dir` | `vectordb` |
| `cost_analysis_dir` | `cost_analysis` |
| `evaluation_dir` | `evaluation_results` |
| `pipeline_state_dir` | `pipeline_state` |

### `analysis`

| Key | Description |
|-----|-------------|
| `enable_cost_analysis` | Run token/cost analysis after generation |
| `enable_evaluation` | Run caption metrics after generation |

Evaluation always reports BLEU, BLEU-1-4, ROUGE-1/2/L and METEOR. CIDEr and BERTScore
are reported only when `pycocoevalcap` and `bert-score` are installed; otherwise they
appear as `n/a` in the report rather than as a misleading score of `0`.

### `cache`

| Key | Description |
|-----|-------------|
| `drive` | Windows drive letter for the HuggingFace cache. Ignored on Linux/macOS |
| `custom_dir` | Explicit cache directory; takes precedence over `drive` |

On Linux and macOS the cache defaults to `$HF_HOME`, or `~/.cache/huggingface`.

## Keys present but not yet read

The shipped config files contain these keys, but nothing in the code reads them.
They are kept because the published runs used config files containing them; changing
them has **no effect**. Treat them as documentation of intent, not as controls.

| Key | Actual behaviour |
|-----|------------------|
| `memory.batch_size` | Index build is hard-coded to a batch size of 16 in `src/main.py` |
| `memory.max_memory_gb`, `memory.enable_monitoring` | `ImageVectorDB` always logs memory usage |
| `rag.similarity_threshold` | All `k` retrieved neighbours are used, unfiltered |
| `cache.enable_symlinks` | Never applied |
| `output.save_vectordb`, `output.save_pipeline_state` | The index and state are always saved |
| `analysis.cost_analysis_format` | Reports are always Markdown |
| `analysis.evaluation_metrics` | Every available metric is computed |
| `api.timeout`, `api.max_retries`, `api.retry_delay` | Provider/LangChain defaults apply; a failed sample is recorded with an `error` field and retried via `--retry-ids` |
| `logging.level`, `logging.format`, `logging.file` | The pipeline prints to stdout; Python `logging` is not configured |
| `dataset.name` | Only used when `type: rocov2` |

## Command-line overrides

| Flag | Overrides |
|------|-----------|
| `--config FILE` | Which YAML file to load |
| `--model NAME` | `model.name` |
| `--samples N` | `dataset.validation_samples` |
| `--rag-examples N` | `rag.num_examples` |
| `--random-seed N` | `dataset.random_seed` |
| `--threads N` | Worker threads for parallel API calls (default 4) |
| `--retry-ids FILE` | Process only the image IDs listed in `FILE` |
| `--cost-analysis` / `--evaluation` | `analysis.enable_*` |
| `--custom-cache-dir DIR` | `cache.custom_dir` |
| `--cache-drive X` | `cache.drive` (Windows only) |

Analysis-only modes skip generation entirely and need no API key:

```bash
python run_pipeline.py --evaluation-only responses/responses_rag_hf_<timestamp>.jsonl
python run_pipeline.py --cost-analysis-only              # newest file in responses/
python run_pipeline.py --cost-analysis-only path/to.jsonl
```

## Notes on long runs

- Results are appended to the JSONL as each sample completes, so an interrupted run
  keeps everything generated so far.
- Failed samples are written with an `error` field. Collect them with
  `process_submission.py collect-errors` and re-run using `--retry-ids`.
- Raise `--threads` to speed up API-bound runs; reduce it if you hit rate limits.
- Reported costs use list prices hard-coded in `src/llm/llm_utils.py`. Unknown models
  report `0.0` rather than guessing, so add new models there if you need costs.
