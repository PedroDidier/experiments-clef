# Configuration Guide

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional: HuggingFace Configuration
HF_HOME=D:/huggingface_cache
HF_DATASETS_CACHE=D:/huggingface_cache
TRANSFORMERS_CACHE=D:/huggingface_cache
HF_HUB_CACHE=D:/huggingface_cache

# Optional: Custom Configuration
CACHE_DRIVE=D
CUSTOM_CACHE_DIR=D:/my_custom_cache
```

## Command Line Configuration

The pipeline supports various command line options:

```bash
# Basic usage
python run_pipeline.py --samples 300 --model gpt-5-mini

# With analysis
python run_pipeline.py --samples 300 --model gpt-5-mini --cost-analysis --evaluation

# Custom cache configuration
python run_pipeline.py --samples 300 --model gpt-5-mini --cache-drive D
python run_pipeline.py --samples 300 --model gpt-5-mini --custom-cache-dir "E:/my_cache"

# Analysis only modes
python run_pipeline.py --cost-analysis-only
python run_pipeline.py --evaluation-only responses/responses_rag_hf_2024-01-15_10-30.jsonl
```

## Programmatic Configuration

You can also configure the pipeline programmatically:

```python
from src.config import update_config

# Configure cache drive
update_config(cache_drive="D")

# Configure custom cache directory
update_config(custom_cache_dir="E:/my_cache")

# Configure both
update_config(cache_drive="C", custom_cache_dir="C:/cache")
```

## Memory Optimization

For systems with limited memory:

1. Use D: drive for caching: `--cache-drive D`
2. Use custom cache directory with more space
3. Start with small samples to test
4. Monitor memory usage during processing
