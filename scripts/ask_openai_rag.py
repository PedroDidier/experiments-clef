import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .image_vectordb import ImageVectorDB

load_dotenv()

# Define the model to use
MODEL = "gpt-4.1"


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r") as file:
        return file.read()


def to_data_url(img_path: str) -> str:
    path = Path(img_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf‑8")
    return f"data:{mime};base64,{b64}"


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Return the dollar cost for a request given token usage and the model
    selected in the global `MODEL` variable.

    Pricing (OpenAI – May 2024):
        GPT-4o            : $2.50 / 1M input  | $10.00 / 1M output
        GPT-4.1           : $2.00 / 1M input  | $8.00 / 1M output
        GPT-4.1-mini      : $0.40 / 1M input  | $1.60 / 1M output
    Any unknown model defaults to GPT-4.1 pricing.
    """
    model_pricing = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1": (2.00, 8.00),
        "gpt-4.1-mini": (0.40, 1.60),
    }

    # Determine the applicable rates
    input_rate, output_rate = model_pricing.get(MODEL, (2.00, 8.00))
    for key, rates in model_pricing.items():
        if key in MODEL:
            input_rate, output_rate = rates
            break

    # Compute cost (convert from per 1M tokens to per token)
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


def get_llm_response_with_rag(
    image_path: str,
    prompt_path: str,
    vectordb_path: str = "vectordb",
    num_examples: int = 3,
) -> Tuple[str, dict, List[Dict[str, Any]]]:
    """
    Get a response from the LLM using RAG with similar images as few-shot examples.
    
    Args:
        image_path (str): Path to the query image
        prompt_path (str): Path to the prompt file
        vectordb_path (str): Path to the vector database
        num_examples (int): Number of similar images to use as examples
        
    Returns:
        Tuple[str, dict, List]: The LLM's JSON response, token usage information, and the similar images used
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set the OPENAI_API_KEY in your .env file. "
            "The .env file should contain: OPENAI_API_KEY=your-api-key-here"
            " and be set at the root directory of this repo."
        )

    client = OpenAI(api_key=api_key)

    # Load vector database
    vectordb = ImageVectorDB()
    vectordb.load(vectordb_path)

    # Find similar images
    similar_images = vectordb.search_similar_images(image_path, k=num_examples)

    # Load the prompt template
    prompt_template = load_prompt(prompt_path)

    # Prepare content for the API call
    content = []

    # Add few-shot examples first
    if similar_images:
        content.append(
            {
                "type": "text",
                "text": "Here are some similar medical images and their captions as examples to guide your response style and format:\n",
            }
        )

        for i, example in enumerate(similar_images):
            example_img_path = example["image_path"]
            example_caption = example["caption"]

            # Add example image
            try:
                example_data_url = to_data_url(example_img_path)
                content.append(
                    {"type": "image_url", "image_url": {"url": example_data_url}}
                )

                # Add example caption as JSON to maintain format consistency
                content.append(
                    {
                        "type": "text",
                        "text": f"Example {i+1} response: {example_caption}\n",
                    }
                )
            except Exception as e:
                print(f"Warning: Could not load example image {example_img_path}: {e}")
                continue

        content.append(
            {
                "type": "text",
                "text": "\nNow, please analyze the following image and provide your response in the same JSON format:\n",
            }
        )

    # Add the main prompt
    content.append({"type": "text", "text": prompt_template})

    # Add the query image
    image_data_url = to_data_url(image_path)
    content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    # Make the API call with JSON format enforcement like the base implementation
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )

    # Extract token usage information
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    cost = calculate_cost(input_tokens, output_tokens)

    token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
    }

    return response.choices[0].message.content, token_usage, similar_images


if __name__ == "__main__":
    # Example usage
    response, token_usage, similar_images = get_llm_response_with_rag(
        image_path="train/ROCOv2_2023_train_000004.jpg",
        prompt_path="prompts/base_prompt.txt",
        vectordb_path="vectordb",
        num_examples=3,
    )

    print("Similar images used as examples:")
    for i, img in enumerate(similar_images):
        print(
            f"Example {i+1}: {img['image_name']} (Similarity: {img['similarity_score']:.4f})"
        )
        print(f"Caption: {img['caption']}")
        print()

    print("\nLLM Response:")
    print(response)

    print("\nToken usage:")
    print(token_usage)
