import json
import os
import random
from datetime import datetime

from .ask_openai import get_llm_response


def generate_captions(dataset_path: str, n_samples: int = 100):
    image_files = [
        f for f in os.listdir(dataset_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Create captions directory if it doesn't exist
    os.makedirs("responses", exist_ok=True)

    # Open JSONL file for writing
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    jsonl_path = f"responses/responses_{timestamp}.jsonl"

    # Sample random n_samples images
    image_files = random.sample(image_files, n_samples)

    for i, image_file in enumerate(image_files):
        caption_json, token_usage = get_llm_response(
            os.path.join(dataset_path, image_file), "prompts/base_prompt.txt"
        )

        # Parse the JSON string response
        caption_data = json.loads(caption_json)

        # Add the image file and token usage information to the JSON
        caption_data["image_file"] = image_file
        caption_data["token_usage"] = token_usage

        # Append to JSONL file one row at a time
        with open(jsonl_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(caption_data) + "\n")

        print(f"Generated caption for {image_file} ({i+1}/{len(image_files)})")
        print(f"Token usage: {token_usage}")
        print(f"Caption data: {caption_data}")
        print("--------------------------------\n")


if __name__ == "__main__":
    generate_captions(dataset_path="test", n_samples=1000)
