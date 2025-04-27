# RAG-Based Few-Shot Image Captioning

This project implements a Retrieval-Augmented Generation (RAG) approach for image captioning, using similar images from a training set as few-shot examples to improve captioning quality.

## Overview

The system works in three stages:
1. **Vectorization**: Images from the training set are encoded using CLIP into a vector database (FAISS)
2. **Retrieval**: When captioning a new image, similar images from the training set are retrieved
3. **Generation**: The similar images and their captions are provided to the LLM as few-shot examples

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## File Structure

```
├── prompts/
│   └── base_prompt.txt     # The base prompt for image captioning
├── train/
│   ├── [images]            # Training images
│   └── train_captions.csv  # CSV file with image captions
├── test/
│   └── [images]            # Test images to caption
├── responses/              # Output JSONL files with generated captions
├── vectordb/              # Vector database files
└── scripts/
    ├── image_vectordb.py   # Vector database implementation
    ├── build_vectordb.py   # Script to build the vector database
    ├── ask_openai.py       # Base OpenAI API call implementation
    ├── ask_openai_rag.py   # RAG-enhanced OpenAI API call implementation
    ├── generate_captions.py      # Original caption generation script
    └── generate_captions_rag.py  # RAG-enhanced caption generation script
```

## Usage

### 1. Build the Vector Database

First, build the vector database from your training images and captions:

```bash
python scripts/build_vectordb.py --image_dir train --captions_file train/train_captions.csv --output_dir vectordb
```

Options:
- `--image_dir`: Directory containing training images (default: "train")
- `--captions_file`: Path to the CSV file with captions (default: "train/train_captions.csv")
- `--output_dir`: Directory to save the vector database (default: "vectordb")
- `--batch_size`: Batch size for processing images (default: 32)

### 2. Generate Captions with RAG

Once the vector database is built, you can generate captions for new images using the RAG approach:

```bash
python scripts/generate_captions_rag.py --dataset_path test --prompt_path prompts/base_prompt.txt --num_examples 3
```

Options:
- `--dataset_path`: Path to the directory with test images (default: "test")
- `--prompt_path`: Path to the prompt file (default: "prompts/base_prompt.txt")
- `--vectordb_path`: Path to the vector database (default: "vectordb")
- `--num_examples`: Number of similar images to use as examples (default: 3)
- `--max_images`: Maximum number of images to process (default: all)

### 3. Analyze the Results

The script generates a JSONL file in the `responses/` directory with the generated captions and metadata, including:
- The generated caption
- The token usage and cost
- The similar images used as examples

## Technical Details

### Vector Database

The system uses FAISS (Facebook AI Similarity Search) as the vector database, which enables efficient similarity search of high-dimensional vectors. For image embeddings, we use OpenAI's CLIP model, which provides high-quality image representations that align well with text.

### Few-Shot Learning

By retrieving similar images and their human-written captions, the system provides the LLM with relevant examples that help it understand the desired captioning style and level of detail. This approach combines the benefits of retrieval and generation, leading to more accurate and contextually relevant captions.

### OpenAI API Integration

The system uses the OpenAI API with GPT-4 Vision capabilities to generate captions. The API calls include:
- The base prompt that explains the task
- Similar images and their captions as few-shot examples
- The target image to caption 