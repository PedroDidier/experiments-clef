import os
import json
import base64
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()


class MedicalImageCaptioner:
    """Medical image captioner using LangChain and OpenAI."""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        """
        Initialize the medical image captioner.
        
        Args:
            model_name (str): OpenAI model name
            temperature (float): Model temperature for generation
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize JSON parser
        self.json_parser = JsonOutputParser()
        
        # Load the base prompt
        self.base_prompt = self._load_base_prompt()
        
    def _load_base_prompt(self) -> str:
        """Load the base prompt from file."""
        prompt_path = Path("prompts/base_prompt.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        Convert image to base64 data URL.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 data URL
        """
        path = Path(image_path)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        return f"data:{mime};base64,{b64}"
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost for a request given token usage.
        
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            
        Returns:
            float: Cost in USD
        """
        # Pricing (OpenAI – Updated 2024)
        model_pricing = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4-turbo": (10.00, 30.00),
            "gpt-4": (30.00, 60.00),
            "gpt-3.5-turbo": (0.50, 1.50),
        }
        
        # Get pricing for the model
        input_rate, output_rate = model_pricing.get(self.model_name, (2.50, 10.00))
        
        # Calculate cost (convert from per 1M tokens to per token)
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        
        return input_cost + output_cost
    
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
            
            # Calculate token usage and cost
            token_usage = {
                "input_tokens": response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                "output_tokens": response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
                "cost_usd": 0.0  # Will be calculated below
            }
            
            # Calculate cost
            token_usage["cost_usd"] = self._calculate_cost(
                token_usage["input_tokens"], 
                token_usage["output_tokens"]
            )
            
            return caption_data, token_usage
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return {"caption": f"Error: {str(e)}"}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    
    def generate_caption_with_rag(
        self, 
        image_path: str, 
        similar_examples: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a caption for a medical image using RAG with similar examples.
        
        Args:
            image_path (str): Path to the image file
            similar_examples (List[Dict[str, Any]]): List of similar examples with captions
            
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]: Caption data, token usage, and examples used
        """
        try:
            # Convert image to base64
            image_data_url = self._image_to_base64(image_path)
            
            # Build the RAG prompt
            rag_prompt = self._build_rag_prompt(similar_examples)
            
            # Create the message with examples, prompt, and image
            content = []
            
            # Add RAG examples
            if similar_examples:
                content.append({
                    "type": "text", 
                    "text": "Here are some similar medical images and their captions as examples to guide your response style and format:\n"
                })
                
                for i, example in enumerate(similar_examples):
                    # Add example image
                    if "image_path" in example:
                        example_image_url = self._image_to_base64(example["image_path"])
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": example_image_url}
                        })
                    
                    # Add example caption
                    content.append({
                        "type": "text",
                        "text": f"Example {i+1} response: {json.dumps({'caption': example['caption']})}\n"
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
            
            # Calculate token usage and cost
            token_usage = {
                "input_tokens": response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0),
                "output_tokens": response.response_metadata.get("token_usage", {}).get("completion_tokens", 0),
                "total_tokens": response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
                "cost_usd": 0.0  # Will be calculated below
            }
            
            # Calculate cost
            token_usage["cost_usd"] = self._calculate_cost(
                token_usage["input_tokens"], 
                token_usage["output_tokens"]
            )
            
            return caption_data, token_usage, similar_examples
            
        except Exception as e:
            print(f"Error generating caption with RAG for {image_path}: {e}")
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
