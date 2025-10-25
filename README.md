# Medical Image Captioning Pipeline

A professional, memory-efficient pipeline for generating medical image captions using Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG).

## Features

- **Memory-Efficient Processing**: Optimized to handle large datasets without memory overflow
- **Configurable Caching**: Flexible cache management for different storage drives
- **Multiple LLM Support**: OpenAI GPT models with LangChain integration
- **RAG Implementation**: Uses similar medical images as few-shot examples
- **Cost Analysis**: Detailed token usage and cost tracking
- **Evaluation Metrics**: BLEU and ROUGE scores for caption quality assessment
- **HuggingFace Integration**: Uses the `eltorio/ROCOv2-radiology` dataset
- **LLM Ensemble Pipeline**: Advanced multi-LLM ensemble with judge selection

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd experiments-clef2025

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Configuration

```bash
# Copy example configuration
cp config_example.yaml config_local.yaml
# Edit config_local.yaml with your settings
```

### 3. Basic Usage

```bash
# Run with configuration file
python run_pipeline.py --config config_local.yaml

# Run with command line overrides
python run_pipeline.py --samples 100 --model gpt-4o --cost-analysis --evaluation

# Use different cache drive
python run_pipeline.py --samples 100 --model gpt-4o --cache-drive C

# Use custom cache directory
python run_pipeline.py --samples 100 --model gpt-4o --custom-cache-dir "E:/my_cache"
```

## Configuration

The pipeline uses a flexible YAML-based configuration system. You can configure:

- **Model Settings**: Model name, temperature, max tokens
- **Dataset Settings**: Sample sizes, random seed, train/validation splits
- **RAG Settings**: Number of examples, similarity thresholds
- **Memory Settings**: Batch sizes, memory monitoring, cleanup
- **Cache Settings**: Drive selection, custom directories
- **Output Settings**: Directory paths, file formats
- **Analysis Settings**: Cost analysis, evaluation metrics

### Configuration Files

1. **`config.yaml`**: Default configuration template
2. **`config_local.yaml`**: Your local configuration (not tracked in git)
3. **`config_example.yaml`**: Example configuration with common settings

### Configuration Priority

1. Command line arguments (highest priority)
2. `config_local.yaml` (if exists)
3. `config.yaml` (default)
4. Built-in defaults (lowest priority)

## Architecture

```
src/
├── config.py              # Configuration management
├── main.py                # Main pipeline orchestration
├── data/
│   └── dataset.py         # HuggingFace dataset handling
├── vectordb/
│   └── image_vectordb.py  # Vector database for image similarity
├── llm/
│   └── llm_utils.py       # LLM integration and RAG
└── analysis/
    ├── cost_analysis.py   # Cost tracking and analysis
    └── evaluation_visualizer.py  # Evaluation metrics and visualization
```

## Memory Optimization

The pipeline is designed to handle large datasets efficiently:

- **Batch Processing**: Images are processed in small batches
- **Memory Management**: Images are not stored in memory after processing
- **Garbage Collection**: Automatic memory cleanup after each batch
- **Configurable Caching**: Use different drives to avoid memory issues

## Models Supported

- `gpt-5-mini` (Recommended for cost-effectiveness)
- `gpt-4o-mini` (Balanced performance and cost)
- `gpt-4o` (Highest quality)
- `gpt-4-turbo` (High performance)
- `gpt-4` (Legacy high performance)
- `gpt-3.5-turbo` (Budget option)

## Output Files

The pipeline generates several output files:

- **Responses**: `responses/responses_rag_hf_<timestamp>.jsonl`
- **Cost Analysis**: `cost_analysis/cost_analysis_<timestamp>.md`
- **Evaluation Results**: `evaluation_results/evaluation_results_<timestamp>/`
- **Vector Database**: `vectordb/image_vectordb/`

## Configuration Options

### Command Line Arguments

```bash
python run_pipeline.py --help
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `HF_HOME`: HuggingFace cache directory
- `HF_DATASETS_CACHE`: HuggingFace datasets cache
- `TRANSFORMERS_CACHE`: Transformers model cache

### Programmatic Configuration

```python
from src.config import update_config

# Configure cache drive
update_config(cache_drive="D")

# Configure custom cache directory
update_config(custom_cache_dir="E:/my_cache")

# Configure both
update_config(cache_drive="C", custom_cache_dir="C:/cache")
```

## LLM Ensemble Pipeline

The project includes an advanced ensemble pipeline that uses multiple LLMs with RAG and a final judge LLM to select the best caption.

### Quick Start with Ensemble

```bash
# Copy ensemble configuration
cp config_ensemble_example.yaml config_ensemble.yaml

# Run ensemble pipeline
python run_ensemble.py --samples 50 --config config_ensemble.yaml
```

### Ensemble Features

- **Multi-Provider Support**: OpenAI, Anthropic, and extensible to other providers
- **RAG Integration**: Uses similar examples to guide caption generation
- **LLM Judge**: Final LLM evaluates and selects the best caption
- **Configurable**: YAML-based configuration for easy customization
- **Async Processing**: Efficient async/await implementation

### Ensemble Configuration

```yaml
# Example configuration
ensemble:
  num_llms: 3
  use_rag: true
  rag:
    num_examples: 3

llms:
  llm_1:
    provider: openai
    model: gpt-4o
    temperature: 0.1
    enabled: true
    
  llm_2:
    provider: openai
    model: gpt-4-turbo
    temperature: 0.2
    enabled: true
    
  llm_3:
    provider: openai
    model: gpt-4o-mini
    temperature: 0.3
    enabled: true

judge:
  provider: openai
  model: gpt-4o
  temperature: 0.1
  enabled: true
```

For detailed ensemble documentation, see [ENSEMBLE_README.md](ENSEMBLE_README.md).

## Performance Tips

1. **Use D: drive for caching** if you have limited C: drive space
2. **Start with small samples** (e.g., 10-50) to test your setup
3. **Monitor memory usage** during large runs
4. **Use GPT-5-mini** for cost-effective processing
5. **Enable analysis only when needed** to save processing time
6. **Use ensemble pipeline** for higher quality captions

## Troubleshooting

### Memory Issues
- Use `--cache-drive D` to use D: drive for caching
- Use `--custom-cache-dir` to specify a directory with more space
- Reduce batch size in the code if needed

### API Issues
- Ensure your OpenAI API key is set correctly
- Check your API quota and billing
- Verify model availability in your region

### Dataset Issues
- The pipeline automatically downloads the ROCOv2-radiology dataset
- First run may take longer due to dataset download
- Ensure stable internet connection for dataset download

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{medical_image_captioning_pipeline,
  title={Medical Image Captioning Pipeline with RAG},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/experiments-clef2025}
}
```