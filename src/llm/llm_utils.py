import os
import json
import base64
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

# Provider SDKs are imported lazily inside _model_factory so that a missing
# package only affects the provider that needs it. Importing them all here
# would make an absent SDK break every run, including runs on other providers.

load_dotenv()


class MedicalImageCaptioner:
    """Medical image captioner using LangChain and OpenAI."""

    def __init__(self, provider: str = "openai", 
                 model_name: str = "gpt-4o", 
                 temperature: float = 0.1, 
                 max_tokens: int = 1000,
                 prompt_prefix: str = "base"):
        """
        Initialize the medical image captioner.
        
        Args:
            model_name (str): OpenAI model name (must support vision like gpt-4o, gpt-4-turbo)
            temperature (float): Model temperature for generation
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_prefix = prompt_prefix

        # Check if model supports vision. The list below is OpenAI-specific, so
        # only warn for that provider - Gemini, Claude and Llama 4 are all
        # multimodal and would otherwise trigger a spurious warning every run.
        vision_models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision-preview"]
        if provider == "openai" and model_name not in vision_models:
            print(f"Warning: {model_name} may not support vision. Consider using gpt-4o or gpt-4-turbo for image processing.")
        
        # Initialize the LLM
        self.llm = self._model_factory(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Initialize JSON parser
        self.json_parser = JsonOutputParser()
        
        # Load the base prompt
        self.base_prompt = self._load_prompt(self.prompt_prefix)
        
    def _load_prompt(self, prefix: str) -> str:
        """Load the prompt from file."""
        print(f"Loading prompt with prefix: {prefix}")
        prompt_path = Path(f"prompts/{prefix}_prompt.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON response, handling cases where it's wrapped in markdown code blocks.
        
        Args:
            content (str): The raw response content
            
        Returns:
            Dict[str, Any]: Parsed JSON data
        """
        content = content.strip()
        
        # Check if wrapped in ```json ... ```
        if content.startswith("```json") and content.endswith("```"):
            json_str = content[7:-3].strip()  # Remove ```json and ```
            return json.loads(json_str)
        else:
            return json.loads(content)
    
    def _image_to_base64(self, image_input) -> str:
        """
        Convert image to base64 data URL.
        
        Args:
            image_input: Either a file path (str) or PIL Image object
            
        Returns:
            str: Base64 data URL
        """
        from PIL import Image
        import io
        
        # Handle PIL Image object
        if isinstance(image_input, Image.Image):
            # Resize image if too large to avoid token limit issues
            max_size = 1024  # Maximum dimension
            if max(image_input.size) > max_size:
                # Calculate new size maintaining aspect ratio
                ratio = max_size / max(image_input.size)
                new_size = (int(image_input.size[0] * ratio), int(image_input.size[1] * ratio))
                image_input = image_input.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized image to {new_size} to avoid token limits")
            
            # Convert PIL image to base64
            buffer = io.BytesIO()
            # Save as JPEG for better compression
            if image_input.mode in ('RGBA', 'LA', 'P'):
                image_input = image_input.convert('RGB')
                image_input.save(buffer, format='JPEG', quality=85, optimize=True)
                mime = "image/jpeg"
            else:
                image_input.save(buffer, format='JPEG', quality=85, optimize=True)
                mime = "image/jpeg"
            
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        
        # Handle file path (original behavior)
        elif isinstance(image_input, str):
            path = Path(image_input)
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            
            with open(image_input, "rb") as image_file:
                b64 = base64.b64encode(image_file.read()).decode("utf-8")
            
            return f"data:{mime};base64,{b64}"
        
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _extract_token_usage(self, response) -> Dict[str, Any]:
        """Extract token usage from a response, accounting for provider differences."""
        if self.provider == "anthropic":
            # Anthropic reports usage under a different key with different names.
            usage = response.response_metadata.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
        else:
            usage = response.response_metadata.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": self._calculate_cost(input_tokens, output_tokens),
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a request given token usage.
        
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            
        Returns:
            float: Cost in USD
        """
        # USD per 1M tokens, as (input, output). Rates are list prices at the
        # time the benchmark was run and drift over time - treat reported costs
        # as estimates, not invoices.
        model_pricing = {
            # OpenAI
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-5-mini": (0.25, 2.00),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-4": (30.00, 60.00),
            "gpt-3.5-turbo": (0.50, 1.50),
            # DeepInfra (open-weight)
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.15, 0.60),
            "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
            # Anthropic
            "claude-sonnet-4-5-20250929": (3.00, 15.00),
            # Google
            "gemini-2.5-pro": (1.25, 10.00),
            "gemini-2.5-flash": (0.30, 2.50),
            "gemini-1.5-pro": (1.25, 5.00),
            "gemini-1.5-flash": (0.075, 0.30),
        }

        # Get pricing for the model. Unknown models report zero rather than
        # silently billing at gpt-4o rates, which would quietly corrupt the
        # cost report for every non-OpenAI provider.
        if self.model_name not in model_pricing:
            return 0.0
        input_rate, output_rate = model_pricing[self.model_name]
        
        # Calculate cost (convert from per 1M tokens to per token)
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        
        return input_cost + output_cost
    
    # Package to install for each provider, used to give a useful error message.
    PROVIDER_PACKAGES = {
        "openai": "langchain-openai",
        "anthropic": "langchain-anthropic",
        "google": "langchain-google-genai",
        "deepinfra": "langchain-community",
        "huggingface": "langchain-huggingface transformers torch",
    }

    def _model_factory(self, provider: str, model_name: str, temperature: float, max_tokens: int):
        """Create an LLM instance for the given provider.

        Provider SDKs are imported here rather than at module level, so that
        running one provider does not require the packages of the others.
        """
        if provider not in self.PROVIDER_PACKAGES:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Expected one of {', '.join(sorted(self.PROVIDER_PACKAGES))}"
            )

        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            elif provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_budget=128
                )
            elif provider == "deepinfra":
                from langchain_community.chat_models import ChatDeepInfra
                return ChatDeepInfra(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=1000000
                )
            elif provider == "huggingface":
                import torch
                from transformers import pipeline
                from langchain_huggingface import HuggingFacePipeline
                model_kwargs = dict(
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                pipe = pipeline(model=model_name, model_kwargs=model_kwargs)
                pipe.model.generation_config.do_sample = False
                return HuggingFacePipeline(pipeline=pipe)
        except ImportError as e:
            raise ImportError(
                f"Provider '{provider}' requires a package that is not installed: {e}. "
                f"Install it with: pip install {self.PROVIDER_PACKAGES[provider]}"
            ) from e
        
    def generate_caption(self, image_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate a caption for a medical image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Caption data and token usage info
        """
        try:
            # Convert image to base64
            image_data_url = self._image_to_base64(image_path)
            
            # Create the message with image and prompt
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.base_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            )
            
            # Generate response
            response = self.llm.invoke([message])

            # Parse the JSON response
            try:
                caption_data = self._parse_json_response(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                caption_data = {"caption": response.content}

            # Calculate token usage and cost
            token_usage = self._extract_token_usage(response)
            
            return caption_data, token_usage
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return {"caption": f"Error: {str(e)}"}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    
    def generate_caption_with_rag(
        self, 
        image_input, 
        similar_examples: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a caption for a medical image using RAG with similar examples.
        
        Args:
            image_input: Either a file path (str) or PIL Image object
            similar_examples (List[Dict[str, Any]]): List of similar examples with captions
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]: Caption data, token usage, and examples used
        """
        try:
            # Convert image to base64
            image_data_url = self._image_to_base64(image_input)
            
            # Build the RAG prompt
            rag_prompt = self._build_rag_prompt(similar_examples)
            
            # Create the message with examples, prompt, and image
            content = []
            
            # Add RAG examples (text-only for memory efficiency)
            if similar_examples:
                content.append({
                    "type": "text", 
                    "text": "Here are some similar medical image captions as examples to guide your response style and format:\n"
                })
                
                for i, example in enumerate(similar_examples):
                    # Add example caption (without image for memory efficiency)
                    content.append({
                        "type": "text",
                        "text": f"Example {i+1} caption: {example['caption']}\n"
                    })
                
                content.append({
                    "type": "text",
                    "text": "\nNow, please analyze the following image and provide your response in the same JSON format:\n"
                })
            
            # Add the main prompt
            content.append({"type": "text", "text": self.base_prompt})
            
            # Add the query image
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
            
            # Create message
            message = HumanMessage(content=content)
            
            # Generate response
            response = self.llm.invoke([message])

            # Parse the JSON response
            try:
                caption_data = self._parse_json_response(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                caption_data = {"caption": response.content}
            
            # Calculate token usage and cost
            token_usage = self._extract_token_usage(response)
            
            return caption_data, token_usage, similar_examples
            
        except Exception as e:
            print(f"Error generating caption with RAG for {image_input}: {e}")
            return {"caption": f"Error: {str(e)}"}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}, similar_examples
    
    def _build_rag_prompt(self, similar_examples: List[Dict[str, Any]]) -> str:
        """
        Build the RAG prompt with similar examples.
        
        Args:
            similar_examples (List[Dict[str, Any]]): List of similar examples
            
        Returns:
            str: RAG prompt text
        """
        if not similar_examples:
            return self.base_prompt
        
        prompt_parts = [
            "Here are some similar medical images and their captions as examples to guide your response style and format:\n"
        ]
        
        for i, example in enumerate(similar_examples):
            prompt_parts.append(f"Example {i+1}: {example['caption']}\n")
        
        prompt_parts.extend([
            "\nNow, please analyze the following image and provide your response in the same JSON format:\n",
            self.base_prompt
        ])
        
        return "\n".join(prompt_parts)


def main():
    """Example usage of the MedicalImageCaptioner."""
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your .env file")
        return
    
    captioner = MedicalImageCaptioner()
    
    # Example with a single image
    image_path = "test/ROCOv2_2023_test_000001.jpg"
    if os.path.exists(image_path):
        caption_data, token_usage = captioner.generate_caption(image_path)
        print("Caption:", caption_data)
        print("Token usage:", token_usage)
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    main()
