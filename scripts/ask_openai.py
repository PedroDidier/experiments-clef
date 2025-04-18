import base64
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "Please set the OPENAI_API_KEY in your .env file. "
        "The .env file should contain: OPENAI_API_KEY=your-api-key-here" 
        " and be set at the root directory of this repo."
    )

client = OpenAI(api_key=api_key)

# Helper → turn a local file into a data‑URL that the API accepts
def to_data_url(img_path: str) -> str:
    path = Path(img_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf‑8")
    return f"data:{mime};base64,{b64}"

# Build the request
image_data_url = to_data_url("train/ROCOv2_2023_train_000004.jpg")
prompt_text    = "Please describe the picture."

response = client.chat.completions.create(
    # or gpt-4o / gpt-4o-vision-preview
    model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",       "text": prompt_text},
                    {"type": "image_url",  "image_url": {"url": image_data_url}},
                ],
            }
        ],
)

# Print the assistant's reply
print(response.choices[0].message.content)