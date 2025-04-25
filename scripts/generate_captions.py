import json
import os
from datetime import datetime

from .ask_openai import get_llm_response


def generate_captions(dataset_path: str):
    image_files = [
        f for f in os.listdir(dataset_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Create captions directory if it doesn't exist
    os.makedirs("responses", exist_ok=True)

    # Open JSONL file for writing
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    jsonl_path = f"responses/responses_{timestamp}.jsonl"

    for i, image_file in enumerate(image_files):
        caption, token_usage = get_llm_response(
            os.path.join(dataset_path, image_file), "prompts/base_prompt.txt"
        )

        # Create JSON object
        caption_data = {
            "generated_caption": caption,
            "image_file": image_file,
            "token_usage": token_usage,
        }

        # Append to JSONL file one row at a time
        with open(jsonl_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(caption_data) + "\n")

        print(f"Generated caption for {image_file} ({i+1}/{len(image_files)})")
        print(f"Token usage: {token_usage}")
        print(f"Caption: {caption}")
        print("--------------------------------\n")


if __name__ == "__main__":
    generate_captions(dataset_path="test")
