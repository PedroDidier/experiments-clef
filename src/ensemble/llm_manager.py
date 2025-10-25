"""
LLM Manager for Ensemble Pipeline

This module provides a unified interface for managing different LLM providers
in the ensemble pipeline, supporting OpenAI, Anthropic, and other providers.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for an LLM instance."""
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 500
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[Dict[str, int]] = None
    cost: Optional[float] = None
    error: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key or self._get_api_key()
    
    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key for the provider."""
        pass
    
    @abstractmethod
    async def generate_caption(self, image_path: str, prompt: str, rag_examples: Optional[List[Dict]] = None) -> LLMResponse:
        """Generate caption for an image."""
        pass
    
    @abstractmethod
    async def judge_captions(self, image_path: str, captions: List[str], criteria: List[str]) -> LLMResponse:
        """Judge multiple captions and select the best one."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def _get_api_key(self) -> str:
        return os.getenv('OPENAI_API_KEY', '')
    
    async def generate_caption(self, image_path: str, prompt: str, rag_examples: Optional[List[Dict]] = None) -> LLMResponse:
        """Generate caption using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Add RAG examples if provided
            if rag_examples:
                rag_context = "\n\nSimilar examples:\n"
                for i, example in enumerate(rag_examples, 1):
                    rag_context += f"{i}. {example.get('caption', '')}\n"
                messages[0]["content"] += rag_context
            
            # Add image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            messages[0]["content"] = [
                {"type": "text", "text": messages[0]["content"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data.hex()}"}}
            ]
            
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.config.model,
                provider='openai',
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content="",
                model=self.config.model,
                provider='openai',
                error=str(e)
            )
    
    async def judge_captions(self, image_path: str, captions: List[str], criteria: List[str]) -> LLMResponse:
        """Judge captions using OpenAI API."""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            # Prepare judge prompt
            judge_prompt = f"""You are an expert medical radiologist. Evaluate the following captions for a medical image and select the best one.

Evaluation Criteria:
{chr(10).join(f"- {criterion}" for criterion in criteria)}

Captions to evaluate:
{chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(captions))}

Please provide:
1. Your selected caption (number)
2. Brief reasoning for your choice
3. Any improvements you would suggest

Format your response as JSON:
{{
    "selected_caption": 1,
    "reasoning": "Brief explanation",
    "improvements": "Suggestions for improvement"
}}"""
            
            messages = [{"role": "user", "content": judge_prompt}]
            
            # Add image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            messages[0]["content"] = [
                {"type": "text", "text": messages[0]["content"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data.hex()}"}}
            ]
            
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.config.model,
                provider='openai',
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"OpenAI judge API error: {e}")
            return LLMResponse(
                content="",
                model=self.config.model,
                provider='openai',
                error=str(e)
            )


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def _get_api_key(self) -> str:
        return os.getenv('ANTHROPIC_API_KEY', '')
    
    async def generate_caption(self, image_path: str, prompt: str, rag_examples: Optional[List[Dict]] = None) -> LLMResponse:
        """Generate caption using Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Prepare prompt with RAG examples
            full_prompt = prompt
            if rag_examples:
                rag_context = "\n\nSimilar examples:\n"
                for i, example in enumerate(rag_examples, 1):
                    rag_context += f"{i}. {example.get('caption', '')}\n"
                full_prompt += rag_context
            
            # Read image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data.hex()}}
                    ]
                }]
            )
            
            content = response.content[0].text
            tokens_used = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.config.model,
                provider='anthropic',
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content="",
                model=self.config.model,
                provider='anthropic',
                error=str(e)
            )
    
    async def judge_captions(self, image_path: str, captions: List[str], criteria: List[str]) -> LLMResponse:
        """Judge captions using Anthropic API."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Prepare judge prompt
            judge_prompt = f"""You are an expert medical radiologist. Evaluate the following captions for a medical image and select the best one.

Evaluation Criteria:
{chr(10).join(f"- {criterion}" for criterion in criteria)}

Captions to evaluate:
{chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(captions))}

Please provide:
1. Your selected caption (number)
2. Brief reasoning for your choice
3. Any improvements you would suggest

Format your response as JSON:
{{
    "selected_caption": 1,
    "reasoning": "Brief explanation",
    "improvements": "Suggestions for improvement"
}}"""
            
            # Read image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": judge_prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data.hex()}}
                    ]
                }]
            )
            
            content = response.content[0].text
            tokens_used = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }
            
            return LLMResponse(
                content=content,
                model=self.config.model,
                provider='anthropic',
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"Anthropic judge API error: {e}")
            return LLMResponse(
                content="",
                model=self.config.model,
                provider='anthropic',
                error=str(e)
            )


class LLMManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self, configs: Dict[str, LLMConfig]):
        self.configs = configs
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        for name, config in self.configs.items():
            if not config.enabled:
                continue
                
            if config.provider == 'openai':
                self.providers[name] = OpenAIProvider(config)
            elif config.provider == 'anthropic':
                self.providers[name] = AnthropicProvider(config)
            else:
                logger.warning(f"Unknown provider: {config.provider}")
    
    async def generate_captions(self, image_path: str, prompt: str, rag_examples: Optional[List[Dict]] = None) -> Dict[str, LLMResponse]:
        """Generate captions using all enabled LLMs."""
        tasks = []
        for name, provider in self.providers.items():
            if name.startswith('llm_'):
                task = provider.generate_caption(image_path, prompt, rag_examples)
                tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"Error generating caption with {name}: {e}")
                results[name] = LLMResponse(
                    content="",
                    model=self.configs[name].model,
                    provider=self.configs[name].provider,
                    error=str(e)
                )
        
        return results
    
    async def judge_captions(self, image_path: str, captions: List[str], criteria: List[str]) -> LLMResponse:
        """Judge captions using the judge LLM."""
        judge_provider = self.providers.get('judge')
        if not judge_provider:
            raise ValueError("No judge provider configured")
        
        return await judge_provider.judge_captions(image_path, captions, criteria)
    
    def get_enabled_llms(self) -> List[str]:
        """Get list of enabled LLM names."""
        return [name for name, config in self.configs.items() if config.enabled and name.startswith('llm_')]
    
    def get_judge_config(self) -> Optional[LLMConfig]:
        """Get judge configuration."""
        return self.configs.get('judge')
