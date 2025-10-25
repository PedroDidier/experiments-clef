#!/usr/bin/env python3
"""
Simple test script for the LLM Ensemble Pipeline

This script demonstrates how to use the ensemble pipeline with a small sample.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.ensemble.ensemble_pipeline import EnsemblePipeline
from src.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_ensemble():
    """Test the ensemble pipeline with a small sample."""
    logger.info("Testing LLM Ensemble Pipeline...")
    
    try:
        # Initialize configuration
        config = Config(cache_drive='D')
        
        # Initialize ensemble pipeline
        pipeline = EnsemblePipeline('config_ensemble.yaml', config)
        
        # Run with just 5 samples for testing
        logger.info("Running ensemble pipeline with 5 samples...")
        results = await pipeline.run_ensemble(samples=5)
        
        # Print results
        logger.info(f"Processed {len(results)} samples")
        
        for i, result in enumerate(results):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Image ID: {result.image_id}")
            logger.info(f"  Final Caption: {result.final_caption[:100]}...")
            logger.info(f"  Judge Selection: {result.judge_selection}")
            logger.info(f"  Processing Time: {result.processing_time:.2f}s")
        
        # Get summary stats
        stats = pipeline.get_summary_stats()
        logger.info(f"\nSummary:")
        logger.info(f"  Total Samples: {stats['total_samples']}")
        logger.info(f"  Successful: {stats['successful_samples']}")
        logger.info(f"  Failed: {stats['failed_samples']}")
        logger.info(f"  Average Time: {stats['average_processing_time']:.2f}s")
        
        if stats['llm_selection_distribution']:
            logger.info("  LLM Selection Distribution:")
            for llm, count in stats['llm_selection_distribution'].items():
                percentage = (count / stats['total_samples']) * 100
                logger.info(f"    {llm}: {count} ({percentage:.1f}%)")
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ensemble())
