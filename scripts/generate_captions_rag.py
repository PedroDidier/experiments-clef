import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from ask_openai_rag import get_llm_response_with_rag

def generate_captions_with_rag(
    dataset_path: str,
    prompt_path: str,
    vectordb_path: str = "vectordb",
    num_examples: int = 3,
    max_images: int = None
):
    """
    Generate captions for images using RAG with similar images as few-shot examples.
    
    Args:
        dataset_path (str): Path to the directory containing test images
        prompt_path (str): Path to the prompt file
        vectordb_path (str): Path to the vector database
        num_examples (int): Number of similar images to use as examples
        max_images (int): Maximum number of images to process (None for all)
    """
    # Get all image files in the dataset folder
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    # Create responses directory if it doesn't exist
    responses_dir = Path("responses")
    responses_dir.mkdir(exist_ok=True)
    
    # Open JSONL file for writing
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    jsonl_path = responses_dir / f"responses_rag_{timestamp}.jsonl"
    
    total_images = len(image_files)
    print(f"Generating captions for {total_images} images using RAG with {num_examples} few-shot examples...")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(dataset_path, image_file)
        
        # Get caption from LLM with RAG
        try:
            caption, token_usage, similar_images = get_llm_response_with_rag(
                image_path=image_path,
                prompt_path=prompt_path,
                vectordb_path=vectordb_path,
                num_examples=num_examples
            )
            
            # Create similarity examples data
            examples_data = []
            for sim_img in similar_images:
                examples_data.append({
                    "image_name": sim_img["image_name"],
                    "caption": sim_img["caption"],
                    "similarity_score": sim_img["similarity_score"]
                })
            
            # Create JSON object
            caption_data = {
                "generated_caption": caption,
                "image_file": image_file,
                "token_usage": token_usage,
                "similar_examples": examples_data
            }
            
            # Append to JSONL file one row at a time
            with open(jsonl_path, "a") as jsonl_file:
                jsonl_file.write(json.dumps(caption_data) + "\n")
            
            print(f"[{i+1}/{total_images}] Generated caption for {image_file}")
            print(f"  - Used {len(similar_images)} similar images as examples")
            print(f"  - Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output (${token_usage['cost_usd']:.4f})")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    print(f"Caption generation complete. Results saved to {jsonl_path}")
    return jsonl_path

def main():
    parser = argparse.ArgumentParser(description="Generate captions with RAG")
    parser.add_argument("--dataset_path", type=str, default="test",
                        help="Path to the directory containing test images")
    parser.add_argument("--prompt_path", type=str, default="prompts/base_prompt.txt",
                        help="Path to the prompt file")
    parser.add_argument("--vectordb_path", type=str, default="vectordb",
                        help="Path to the vector database")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of similar images to use as examples")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process (default: all)")
    
    args = parser.parse_args()
    
    generate_captions_with_rag(
        dataset_path=args.dataset_path,
        prompt_path=args.prompt_path,
        vectordb_path=args.vectordb_path,
        num_examples=args.num_examples,
        max_images=args.max_images
    )

if __name__ == "__main__":
    main() 