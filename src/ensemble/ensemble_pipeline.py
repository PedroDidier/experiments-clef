"""
Ensemble Pipeline for Medical Image Captioning

This module implements the ensemble pipeline that uses multiple LLMs with RAG
and a final judge LLM to select the best caption.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from datetime import datetime

from .llm_manager import LLMManager, LLMConfig, LLMResponse
from ..vectordb.image_vectordb import ImageVectorDB
from ..data.dataset import ROCOv2DataHandler
from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from ensemble pipeline."""
    image_id: str
    image_path: str
    individual_captions: Dict[str, str]
    judge_selection: int
    final_caption: str
    judge_reasoning: str
    judge_improvements: str
    tokens_used: Dict[str, Dict[str, int]]
    processing_time: float
    timestamp: str


@dataclass
class EnsembleConfig:
    """Configuration for ensemble pipeline."""
    num_llms: int = 3
    use_rag: bool = True
    rag_num_examples: int = 3
    rag_similarity_threshold: float = 0.8
    batch_size: int = 10
    max_retries: int = 3
    request_delay: float = 0.5
    parallel_processing: bool = True
    max_concurrent: int = 3
    save_individual_outputs: bool = True
    save_judge_reasoning: bool = True
    output_format: str = 'jsonl'


class EnsemblePipeline:
    """Main ensemble pipeline class."""
    
    def __init__(self, config_path: str, base_config: Optional[Config] = None):
        self.base_config = base_config or Config()
        self.ensemble_config = self._load_ensemble_config(config_path)
        self.llm_manager = None
        self.vectordb = None
        self.data_handler = None
        self.results = []
        
    def _load_ensemble_config(self, config_path: str) -> Tuple[EnsembleConfig, Dict[str, LLMConfig]]:
        """Load ensemble configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Load ensemble settings
        ensemble_data = config_data.get('ensemble', {})
        ensemble_config = EnsembleConfig(
            num_llms=ensemble_data.get('num_llms', 3),
            use_rag=ensemble_data.get('use_rag', True),
            rag_num_examples=ensemble_data.get('rag', {}).get('num_examples', 3),
            rag_similarity_threshold=ensemble_data.get('rag', {}).get('similarity_threshold', 0.8),
            batch_size=config_data.get('processing', {}).get('batch_size', 10),
            max_retries=config_data.get('processing', {}).get('max_retries', 3),
            request_delay=config_data.get('processing', {}).get('request_delay', 0.5),
            parallel_processing=config_data.get('processing', {}).get('parallel_processing', True),
            max_concurrent=config_data.get('processing', {}).get('max_concurrent', 3),
            save_individual_outputs=config_data.get('output', {}).get('save_individual_outputs', True),
            save_judge_reasoning=config_data.get('output', {}).get('save_judge_reasoning', True),
            output_format=config_data.get('output', {}).get('format', 'jsonl')
        )
        
        # Load LLM configurations
        llm_configs = {}
        llms_data = config_data.get('llms', {})
        judge_data = config_data.get('judge', {})
        
        # Load individual LLMs
        for llm_name, llm_data in llms_data.items():
            llm_configs[llm_name] = LLMConfig(
                provider=llm_data.get('provider', 'openai'),
                model=llm_data.get('model', 'gpt-4o'),
                temperature=llm_data.get('temperature', 0.1),
                max_tokens=llm_data.get('max_tokens', 500),
                enabled=llm_data.get('enabled', True)
            )
        
        # Load judge LLM
        if judge_data:
            llm_configs['judge'] = LLMConfig(
                provider=judge_data.get('provider', 'openai'),
                model=judge_data.get('model', 'gpt-4o'),
                temperature=judge_data.get('temperature', 0.1),
                max_tokens=judge_data.get('max_tokens', 300),
                enabled=judge_data.get('enabled', True)
            )
        
        return ensemble_config, llm_configs
    
    async def setup_pipeline(self):
        """Setup the ensemble pipeline components."""
        logger.info("Setting up ensemble pipeline...")
        
        # Initialize LLM manager
        self.llm_manager = LLMManager(self.ensemble_config[1])
        
        # Setup vector database if RAG is enabled
        if self.ensemble_config[0].use_rag:
            logger.info("Setting up vector database for RAG...")
            self.vectordb = ImageVectorDB(self.base_config)
            await self.vectordb.load_existing()
            
            if not self.vectordb.is_loaded():
                logger.info("Building vector database...")
                self.data_handler = ROCOv2DataHandler(self.base_config)
                train_dataset = self.data_handler.get_train_samples_for_vectordb()
                await self.vectordb.build_from_huggingface_dataset(
                    train_dataset,
                    batch_size=self.base_config.get_memory_config()['vectordb_batch_size'],
                    max_samples=self.base_config.get_memory_config()['vectordb_max_samples']
                )
                await self.vectordb.save()
        
        logger.info("Ensemble pipeline setup complete")
    
    async def process_single_image(self, image_path: str, image_id: str) -> EnsembleResult:
        """Process a single image through the ensemble pipeline."""
        start_time = datetime.now()
        
        try:
            # Get RAG examples if enabled
            rag_examples = None
            if self.ensemble_config[0].use_rag and self.vectordb:
                similar_images = await self.vectordb.search_similar_images(
                    image_path, 
                    k=self.ensemble_config[0].rag_num_examples
                )
                rag_examples = [
                    {
                        'caption': img['caption'],
                        'similarity': img['similarity']
                    }
                    for img in similar_images
                ]
            
            # Generate captions with all LLMs
            individual_responses = await self.llm_manager.generate_captions(
                image_path, 
                self._get_base_prompt(),
                rag_examples
            )
            
            # Extract captions
            individual_captions = {}
            tokens_used = {}
            
            for llm_name, response in individual_responses.items():
                if response.error:
                    logger.warning(f"Error with {llm_name}: {response.error}")
                    individual_captions[llm_name] = f"Error: {response.error}"
                else:
                    individual_captions[llm_name] = response.content
                
                if response.tokens_used:
                    tokens_used[llm_name] = response.tokens_used
            
            # Judge captions
            captions_list = list(individual_captions.values())
            judge_response = await self.llm_manager.judge_captions(
                image_path,
                captions_list,
                self._get_judge_criteria()
            )
            
            # Parse judge response
            judge_selection = 0
            judge_reasoning = ""
            judge_improvements = ""
            
            if judge_response.error:
                logger.warning(f"Judge error: {judge_response.error}")
                judge_reasoning = f"Error: {judge_response.error}"
            else:
                try:
                    judge_data = json.loads(judge_response.content)
                    judge_selection = judge_data.get('selected_caption', 1) - 1  # Convert to 0-based index
                    judge_reasoning = judge_data.get('reasoning', '')
                    judge_improvements = judge_data.get('improvements', '')
                except json.JSONDecodeError:
                    logger.warning("Failed to parse judge response as JSON")
                    judge_reasoning = judge_response.content
            
            # Ensure valid selection
            if judge_selection < 0 or judge_selection >= len(captions_list):
                judge_selection = 0
            
            final_caption = captions_list[judge_selection]
            
            # Add judge tokens
            if judge_response.tokens_used:
                tokens_used['judge'] = judge_response.tokens_used
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnsembleResult(
                image_id=image_id,
                image_path=image_path,
                individual_captions=individual_captions,
                judge_selection=judge_selection,
                final_caption=final_caption,
                judge_reasoning=judge_reasoning,
                judge_improvements=judge_improvements,
                tokens_used=tokens_used,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnsembleResult(
                image_id=image_id,
                image_path=image_path,
                individual_captions={},
                judge_selection=0,
                final_caption=f"Error: {str(e)}",
                judge_reasoning=f"Error: {str(e)}",
                judge_improvements="",
                tokens_used={},
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
    
    async def run_ensemble(self, samples: int = 100) -> List[EnsembleResult]:
        """Run the ensemble pipeline on validation samples."""
        logger.info(f"Starting ensemble pipeline with {samples} samples")
        
        # Setup pipeline
        await self.setup_pipeline()
        
        # Get validation samples
        if not self.data_handler:
            self.data_handler = ROCOv2DataHandler(self.base_config)
        
        validation_samples = self.data_handler.get_validation_samples(samples)
        
        # Process samples
        results = []
        for i, sample in enumerate(validation_samples):
            logger.info(f"Processing sample {i+1}/{samples}")
            
            result = await self.process_single_image(
                sample['image_path'],
                sample['image_id']
            )
            
            results.append(result)
            
            # Add delay between requests
            if self.ensemble_config[0].request_delay > 0:
                await asyncio.sleep(self.ensemble_config[0].request_delay)
        
        self.results = results
        logger.info(f"Ensemble pipeline completed. Processed {len(results)} samples")
        
        return results
    
    def save_results(self, output_path: str):
        """Save ensemble results to file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.ensemble_config[0].output_format == 'jsonl':
            with open(output_path, 'w') as f:
                for result in self.results:
                    f.write(json.dumps(asdict(result)) + '\n')
        else:
            with open(output_path, 'w') as f:
                json.dump([asdict(result) for result in self.results], f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def _get_base_prompt(self) -> str:
        """Get the base prompt for caption generation."""
        return """You are an expert medical radiologist. Analyze this medical image and provide a detailed, accurate caption describing what you observe.

Please include:
1. The imaging modality (X-ray, CT, MRI, ultrasound, etc.)
2. The anatomical region or body part
3. Any visible abnormalities, lesions, or findings
4. Technical details about the image quality or positioning
5. Any measurements or annotations visible

Provide a clear, professional medical description that would be useful for clinical documentation."""
    
    def _get_judge_criteria(self) -> List[str]:
        """Get the criteria for judging captions."""
        return [
            "Medical accuracy and terminology",
            "Completeness of description",
            "Clarity and readability",
            "Technical precision"
        ]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the ensemble run."""
        if not self.results:
            return {}
        
        total_tokens = {}
        total_cost = 0.0
        successful_results = [r for r in self.results if not r.final_caption.startswith("Error:")]
        
        for result in self.results:
            for llm_name, tokens in result.tokens_used.items():
                if llm_name not in total_tokens:
                    total_tokens[llm_name] = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                
                total_tokens[llm_name]['input_tokens'] += tokens.get('input_tokens', 0)
                total_tokens[llm_name]['output_tokens'] += tokens.get('output_tokens', 0)
                total_tokens[llm_name]['total_tokens'] += tokens.get('total_tokens', 0)
        
        return {
            'total_samples': len(self.results),
            'successful_samples': len(successful_results),
            'failed_samples': len(self.results) - len(successful_results),
            'total_tokens': total_tokens,
            'average_processing_time': sum(r.processing_time for r in self.results) / len(self.results),
            'llm_selection_distribution': self._get_selection_distribution()
        }
    
    def _get_selection_distribution(self) -> Dict[str, int]:
        """Get distribution of which LLM was selected by the judge."""
        distribution = {}
        for result in self.results:
            if result.judge_selection < len(result.individual_captions):
                selected_llm = list(result.individual_captions.keys())[result.judge_selection]
                distribution[selected_llm] = distribution.get(selected_llm, 0) + 1
        return distribution
