import base64
import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI

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


def get_llm_response(image_path: str, prompt_path: str) -> Tuple[str, dict]:
    """
    Get a response from the LLM for a given image and prompt.

    Args:
        image_path (str): Path to the image file
        prompt_path (str): Path to the prompt file

    Returns:
        Tuple[str, dict]: A tuple containing the LLM's response text and a dictionary with token usage information
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please set the OPENAI_API_KEY in your .env file. "
            "The .env file should contain: OPENAI_API_KEY=your-api-key-here"
            " and be set at the root directory of this repo."
        )

    client = OpenAI(api_key=api_key)

    image_data_url = to_data_url(image_path)
    prompt_text = load_prompt(prompt_path)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
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

    return response.choices[0].message.content, token_usage


if __name__ == "__main__":
    # Example usage
    response, token_usage = get_llm_response(
        image_path="train/ROCOv2_2023_train_000004.jpg",
        prompt_path="prompts/base_prompt.txt",
    )
    print("Response:", response)
    print("Token usage:", token_usage)
