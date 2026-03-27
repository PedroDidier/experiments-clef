import os
import json
import base64
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


class MedicalImageCaptioner:
    """Medical image captioner using LangChain and multiple LLM providers."""

    def __init__(self, provider: str = "openai", 
                 model_name: str = "gpt-4o", 
                 temperature: float = 0.1, 
                 max_tokens: int = 1000):
        """
        Initialize the medical image captioner.
        
        Args:
            provider (str): LLM provider (openai, google, together, anthropic)
            model_name (str): Model name for the provider
            temperature (float): Model temperature for generation
            max_tokens (int): Maximum tokens for generation
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
        self.base_prompt = self._load_prompt("simple")
        
    def _load_prompt(self, prefix: str) -> str:
        """Load the prompt from file."""
        print(f"Loading prompt with prefix: {prefix}")
        prompt_path = Path(f"prompts/{prefix}_prompt.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
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
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a request given token usage.
        Returns cost in USD.
        """
        # Pricing is per 1M tokens: (input_rate, output_rate)
        pricing = {
            "openai": {
                "gpt-4o": (2.50, 10.00),
                "gpt-4o-mini": (0.15, 0.60),
                "gpt-5-mini": (0.25, 2.00),
                "gpt-4-turbo": (10.00, 30.00),
                "gpt-4": (30.00, 60.00),
                "gpt-3.5-turbo": (0.50, 1.50),
            },
            "together": {
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
                "Qwen/Qwen3-VL-8B-Instruct": (0.18, 0.68),
                "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
                "Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
            },
            "anthropic": {
                # Claude 4 pricing (Dec 2024)
                "claude-sonnet-4-5-20250929": (3.00, 15.00),
            },
            "google": {}
        }

        provider = getattr(self, "provider", None)
        if provider is None:
            provider = "openai"

        model_rates = pricing.get(provider, {})
        input_rate, output_rate = model_rates.get(self.model_name, (0.0, 0.0))

        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        return input_cost + output_cost

    
    def _model_factory(self, provider: str, model_name: str, temperature: float, max_tokens: int):
        """Factory method to create LLM instances based on provider."""
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "together":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
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
                caption_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                caption_data = {"caption": response.content}
            
            # Extract token usage based on provider
            token_usage = self._extract_token_usage(response)
            
            # Calculate cost
            token_usage["cost_usd"] = self._calculate_cost(
                token_usage["input_tokens"], 
                token_usage["output_tokens"]
            )
            
            return caption_data, token_usage
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return {"caption": f"Error: {str(e)}"}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    
    def _extract_token_usage(self, response) -> Dict[str, Any]:
        """Extract token usage from response based on provider."""
        if self.provider == "anthropic":
            # Anthropic uses different field names
            usage = response.response_metadata.get("usage", {})
            return {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "cost_usd": 0.0
            }
        else:
            # OpenAI, Google, Together
            token_usage_data = response.response_metadata.get("token_usage", {})
            return {
                "input_tokens": token_usage_data.get("prompt_tokens", 0),
                "output_tokens": token_usage_data.get("completion_tokens", 0),
                "total_tokens": token_usage_data.get("total_tokens", 0),
                "cost_usd": 0.0
            }
    
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
                caption_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                caption_data = {"caption": response.content}
            
            # Extract token usage based on provider
            token_usage = self._extract_token_usage(response)
            
            # Calculate cost
            token_usage["cost_usd"] = self._calculate_cost(
                token_usage["input_tokens"], 
                token_usage["output_tokens"]
            )
            
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
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return
    
    captioner = MedicalImageCaptioner(
        provider="anthropic",
        model_name="claude-sonnet-4-20250514"
    )
    
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