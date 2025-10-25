"""
Ensemble Package for Medical Image Captioning

This package provides LLM ensemble functionality for medical image captioning,
including support for multiple providers and RAG-based caption generation.
"""

from .llm_manager import LLMManager, LLMConfig, LLMResponse, OpenAIProvider, AnthropicProvider
from .ensemble_pipeline import EnsemblePipeline, EnsembleResult, EnsembleConfig

__all__ = [
    'LLMManager',
    'LLMConfig', 
    'LLMResponse',
    'OpenAIProvider',
    'AnthropicProvider',
    'EnsemblePipeline',
    'EnsembleResult',
    'EnsembleConfig'
]
