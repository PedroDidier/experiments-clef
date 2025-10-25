# LLM Ensemble Pipeline for Medical Image Captioning

This module provides an advanced ensemble pipeline that uses multiple Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) and a final judge LLM to select the best caption for medical images.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, and extensible to other providers
- **RAG Integration**: Uses similar examples to guide caption generation
- **LLM Judge**: Final LLM evaluates and selects the best caption
- **Configurable**: YAML-based configuration for easy customization
- **Async Processing**: Efficient async/await implementation
- **Comprehensive Logging**: Detailed logging and error handling
- **Cost Tracking**: Token usage and cost analysis

## Architecture

```
Image → [LLM1 + RAG] → Caption 1
      → [LLM2 + RAG] → Caption 2
      → [LLM3 + RAG] → Caption 3
                        ↓
                    [Judge LLM] → Final Caption
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic (optional)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Configure Ensemble

Copy the example configuration:
```bash
cp config_ensemble_example.yaml config_ensemble.yaml
```

Edit `config_ensemble.yaml` to customize your LLM providers and settings.

### 4. Run Ensemble Pipeline

```bash
python run_ensemble.py --samples 50 --config config_ensemble.yaml
```

## Configuration

### Ensemble Settings

```yaml
ensemble:
  num_llms: 3                    # Number of LLMs to use
  use_rag: true                  # Enable RAG
  rag:
    num_examples: 3              # Number of RAG examples
    similarity_threshold: 0.8    # Similarity threshold
```

### LLM Configuration

```yaml
llms:
  llm_1:
    provider: openai             # Provider (openai, anthropic)
    model: gpt-4o               # Model name
    temperature: 0.1            # Generation temperature
    max_tokens: 500             # Maximum tokens
    enabled: true               # Enable/disable this LLM
```

### Judge Configuration

```yaml
judge:
  provider: openai
  model: gpt-4o
  temperature: 0.1
  max_tokens: 300
  enabled: true
```

## Command Line Options

### Basic Usage

```bash
python run_ensemble.py --samples 100 --config config_ensemble.yaml
```

### Advanced Options

```bash
python run_ensemble.py \
  --samples 200 \
  --config config_ensemble.yaml \
  --output-dir results \
  --batch-size 20 \
  --max-concurrent 5 \
  --request-delay 1.0 \
  --disable-rag \
  --verbose
```

### Available Options

- `--samples`: Number of validation samples to process
- `--config`: Path to ensemble configuration file
- `--output-dir`: Output directory for results
- `--batch-size`: Batch size for processing
- `--max-concurrent`: Maximum concurrent requests
- `--request-delay`: Delay between requests (seconds)
- `--disable-rag`: Disable RAG for ensemble LLMs
- `--rag-examples`: Number of RAG examples to use
- `--save-individual`: Save individual LLM outputs
- `--save-judge-reasoning`: Save judge reasoning
- `--verbose`: Enable verbose logging
- `--dry-run`: Show configuration without processing

## Output Format

The ensemble pipeline generates JSONL files with the following structure:

```json
{
  "image_id": "ROCOv2_2023_valid_000001",
  "image_path": "/path/to/image.jpg",
  "individual_captions": {
    "llm_1": "Caption from LLM 1",
    "llm_2": "Caption from LLM 2", 
    "llm_3": "Caption from LLM 3"
  },
  "judge_selection": 0,
  "final_caption": "Selected caption",
  "judge_reasoning": "Reasoning for selection",
  "judge_improvements": "Suggested improvements",
  "tokens_used": {
    "llm_1": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    "llm_2": {"input_tokens": 95, "output_tokens": 45, "total_tokens": 140},
    "llm_3": {"input_tokens": 90, "output_tokens": 40, "total_tokens": 130},
    "judge": {"input_tokens": 200, "output_tokens": 30, "total_tokens": 230}
  },
  "processing_time": 5.2,
  "timestamp": "2025-01-01T12:00:00"
}
```

## Provider Support

### OpenAI

```yaml
llm_1:
  provider: openai
  model: gpt-4o              # gpt-4o, gpt-4-turbo, gpt-4, gpt-4o-mini
  temperature: 0.1
  max_tokens: 500
```

### Anthropic

```yaml
llm_1:
  provider: anthropic
  model: claude-3-opus-20240229    # claude-3-opus, claude-3-sonnet, claude-3-haiku
  temperature: 0.1
  max_tokens: 500
```

## Extending the System

### Adding New Providers

1. Create a new provider class inheriting from `LLMProvider`
2. Implement the required methods: `generate_caption` and `judge_captions`
3. Add the provider to the `LLMManager._initialize_providers` method

Example:
```python
class CustomProvider(LLMProvider):
    def _get_api_key(self) -> str:
        return os.getenv('CUSTOM_API_KEY', '')
    
    async def generate_caption(self, image_path: str, prompt: str, rag_examples: Optional[List[Dict]] = None) -> LLMResponse:
        # Implementation here
        pass
    
    async def judge_captions(self, image_path: str, captions: List[str], criteria: List[str]) -> LLMResponse:
        # Implementation here
        pass
```

### Custom Judge Criteria

Modify the `_get_judge_criteria` method in `EnsemblePipeline`:

```python
def _get_judge_criteria(self) -> List[str]:
    return [
        "Medical accuracy and terminology",
        "Completeness of description", 
        "Clarity and readability",
        "Technical precision",
        "Your custom criterion"
    ]
```

## Performance Considerations

### Memory Usage

- The ensemble pipeline loads the vector database into memory
- Consider using `--batch-size` to control memory usage
- Use `--max-concurrent` to limit concurrent API calls

### API Rate Limits

- Use `--request-delay` to add delays between requests
- Adjust `--max-concurrent` based on your API limits
- Monitor token usage to avoid exceeding quotas

### Cost Optimization

- Use different models for different LLMs (e.g., gpt-4o-mini for cost-effective option)
- Adjust `max_tokens` based on expected caption length
- Monitor token usage in the output files

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are set correctly
2. **Memory Issues**: Reduce batch size or use smaller models
3. **Rate Limiting**: Increase request delay or reduce concurrent requests
4. **JSON Parsing Errors**: Check judge LLM output format

### Debug Mode

Use `--verbose` and `--dry-run` for debugging:

```bash
python run_ensemble.py --verbose --dry-run --config config_ensemble.yaml
```

## Integration with Existing Pipeline

The ensemble pipeline integrates seamlessly with the existing medical image captioning pipeline:

- Uses the same vector database and RAG system
- Compatible with existing configuration system
- Can be run alongside the standard pipeline
- Results can be evaluated using the same evaluation tools

## Examples

### Basic Ensemble Run

```bash
python run_ensemble.py --samples 50
```

### Custom Configuration

```bash
python run_ensemble.py \
  --samples 100 \
  --config my_custom_config.yaml \
  --output-dir custom_results \
  --save-individual \
  --save-judge-reasoning
```

### Cost-Optimized Run

```bash
python run_ensemble.py \
  --samples 200 \
  --disable-rag \
  --request-delay 2.0 \
  --max-concurrent 1
```

## License

This ensemble pipeline is part of the medical image captioning project and follows the same license terms.
