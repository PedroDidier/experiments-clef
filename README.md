# Medical Image Captioning Pipeline - Refactored

This document describes the major refactor of the medical image captioning pipeline to use HuggingFace datasets, LangChain, and a modular architecture.

## Overview of Changes

### 1. HuggingFace Dataset Integration
- **Before**: Local CSV files with train/test/validation splits
- **After**: Direct integration with `eltorio/ROCOv2-radiology` dataset from HuggingFace
- **Benefits**: 
  - No need to download and manage local data files
  - Access to the full 79,789 image dataset
  - Automatic handling of train/validation/test splits
  - Easy access to image metadata

### 2. LangChain Integration
- **Before**: Direct OpenAI API calls
- **After**: LangChain-based LLM utilities
- **Benefits**:
  - Easy switching between LLM providers (OpenAI, Claude, Gemini)
  - Better prompt management
  - Structured output parsing
  - Future extensibility

### 3. Modular Architecture
- **Before**: Scripts scattered in `/scripts` directory
- **After**: Organized modules in `/src` directory
- **Structure**:
  ```
  src/
  ├── data/           # HuggingFace dataset handling
  ├── vectordb/       # Vector database operations
  ├── llm/           # LangChain LLM utilities
  └── main.py        # Main orchestration script
  ```

### 4. Simplified Focus
- **Before**: Image captioning + CUI classification
- **After**: Image captioning only
- **Benefits**: Cleaner prompts, faster processing, focused evaluation

## New Architecture

### Data Module (`src/data/dataset.py`)
- `ROCOv2DataHandler`: Handles HuggingFace dataset loading
- Methods:
  - `load_dataset()`: Load ROCOv2 dataset from HuggingFace
  - `get_validation_samples(num_samples=300)`: Sample validation images for evaluation
  - `get_train_samples_for_vectordb()`: Get all training samples for VectorDB
  - `save_sample_images()`: Save images to disk if needed

### VectorDB Module (`src/vectordb/image_vectordb.py`)
- `ImageVectorDB`: CLIP-based image similarity search
- Methods:
  - `build_from_huggingface_dataset()`: Build VectorDB from HuggingFace samples
  - `search_similar_images()`: Find similar images for RAG
  - `save()` / `load()`: Persist VectorDB to disk

### LLM Module (`src/llm/llm_utils.py`)
- `MedicalImageCaptioner`: LangChain-based caption generation
- Methods:
  - `generate_caption()`: Generate caption for single image
  - `generate_caption_with_rag()`: Generate caption with RAG examples
  - Support for multiple LLM providers

### Main Pipeline (`src/main.py`)
- `MedicalImageCaptioningPipeline`: Orchestrates the entire process
- Workflow:
  1. Load ROCOv2 dataset from HuggingFace
  2. Build VectorDB from training samples
  3. Sample 300 validation images
  4. Generate captions with RAG for each validation image
  5. Save results to JSONL file

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Run the pipeline
python src/main.py
```

### Configuration
The pipeline can be configured in `src/main.py`:
```python
pipeline = MedicalImageCaptioningPipeline(
    model_name="gpt-4o",           # OpenAI model
    num_validation_samples=300,    # Number of validation samples
    num_rag_examples=3,           # Number of RAG examples
    random_seed=42                # Random seed for reproducibility
)
```

### Testing
```bash
# Run integration tests
python test_integration.py
```

## Key Features

### 1. HuggingFace Dataset Integration
- Automatic dataset loading and caching
- Efficient sampling of validation images
- Direct access to PIL images and metadata

### 2. RAG-Enhanced Captioning
- Uses CLIP embeddings for image similarity
- Retrieves similar training images as examples
- Improves caption quality through few-shot learning

### 3. LangChain Integration
- Easy switching between LLM providers
- Structured output parsing
- Better error handling and retry logic

### 4. Modular Design
- Clear separation of concerns
- Easy to extend and modify
- Better testability

### 5. Cost Tracking
- Detailed token usage tracking
- Cost calculation per request
- Total cost reporting

## Output Format

The pipeline generates a JSONL file with the following structure:
```json
{
  "image_id": "ROCOv2_2023_validation_000001",
  "ground_truth_caption": "Chest X-ray showing...",
  "generated_caption": "Chest radiograph demonstrating...",
  "token_usage": {
    "input_tokens": 1500,
    "output_tokens": 200,
    "total_tokens": 1700,
    "cost_usd": 0.0125
  },
  "rag_examples": [
    {
      "image_id": "ROCOv2_2023_train_000123",
      "caption": "Similar chest X-ray...",
      "similarity_score": 0.85
    }
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

## Migration from Old System

### Data Migration
- No need to download local CSV files
- Dataset is automatically loaded from HuggingFace
- All 79,789 images are available

### Code Migration
- Old scripts in `/scripts` are deprecated
- Use new modular structure in `/src`
- Main entry point is now `src/main.py`

### Evaluation Migration
- Evaluation scripts can be updated to work with new JSONL format
- Ground truth captions are included in the output
- Same evaluation metrics can be used

## Future Enhancements

### 1. Multi-LLM Support
- Easy addition of Claude, Gemini, or other providers
- A/B testing between different models
- Cost comparison across providers

### 2. Advanced RAG
- Semantic search improvements
- Multi-modal embeddings
- Dynamic example selection

### 3. Evaluation Improvements
- Automated evaluation pipeline
- Real-time metrics tracking
- Comparative analysis tools

### 4. Deployment
- Docker containerization
- API endpoint creation
- Batch processing optimization

## Troubleshooting

### Common Issues

1. **HuggingFace Dataset Loading**
   - Ensure internet connection
   - Check dataset availability
   - Verify HuggingFace credentials if needed

2. **OpenAI API Issues**
   - Verify API key in `.env` file
   - Check API quota and billing
   - Ensure model availability

3. **Memory Issues**
   - Reduce batch size in VectorDB building
   - Use smaller validation sample size
   - Consider using GPU for CLIP embeddings

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes `src/`
   - Verify module structure

### Performance Optimization

1. **VectorDB Building**
   - Use GPU if available for CLIP embeddings
   - Increase batch size for faster processing
   - Save and reuse VectorDB between runs

2. **Caption Generation**
   - Use smaller models for faster generation
   - Implement request batching
   - Add retry logic for failed requests

3. **Memory Usage**
   - Process images in smaller batches
   - Clear unused variables
   - Use memory profiling tools

## Conclusion

The refactored pipeline provides a more robust, scalable, and maintainable solution for medical image captioning. The integration with HuggingFace datasets eliminates data management overhead, while LangChain provides flexibility for future LLM integrations. The modular architecture makes the system easier to understand, test, and extend.
