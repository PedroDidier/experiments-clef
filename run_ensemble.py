#!/usr/bin/env python3
"""
Ensemble Pipeline Runner

This script runs the LLM ensemble pipeline for medical image captioning.
It supports multiple LLM providers and configurations.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ensemble.ensemble_pipeline import EnsemblePipeline
from src.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM ensemble pipeline for medical image captioning"
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config_ensemble.yaml',
        help='Path to ensemble configuration file'
    )
    
    parser.add_argument(
        '--base-config',
        type=str,
        default='config.yaml',
        help='Path to base configuration file'
    )
    
    # Processing options
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of validation samples to process'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ensemble_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file name (default: auto-generated)'
    )
    
    # Cache options
    parser.add_argument(
        '--cache-drive',
        type=str,
        default='D',
        help='Drive letter for caching (e.g., D, C)'
    )
    
    parser.add_argument(
        '--custom-cache-dir',
        type=str,
        default=None,
        help='Custom cache directory path'
    )
    
    # Processing options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=None,
        help='Maximum concurrent requests'
    )
    
    parser.add_argument(
        '--request-delay',
        type=float,
        default=None,
        help='Delay between requests (seconds)'
    )
    
    # LLM options
    parser.add_argument(
        '--disable-rag',
        action='store_true',
        help='Disable RAG for ensemble LLMs'
    )
    
    parser.add_argument(
        '--rag-examples',
        type=int,
        default=None,
        help='Number of RAG examples to use'
    )
    
    # Output options
    parser.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual LLM outputs'
    )
    
    parser.add_argument(
        '--save-judge-reasoning',
        action='store_true',
        help='Save judge reasoning'
    )
    
    # Debug options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - show configuration without processing'
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def validate_config(config_path: str):
    """Validate configuration file exists."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)


def generate_output_filename(samples: int, timestamp: str = None) -> str:
    """Generate output filename."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    return f"ensemble_results_{samples}samples_{timestamp}.jsonl"


async def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate configuration
    validate_config(args.config)
    
    # Generate output filename if not provided
    if args.output_file is None:
        args.output_file = generate_output_filename(args.samples)
    
    # Create output path
    output_path = Path(args.output_dir) / args.output_file
    
    logger.info("=" * 60)
    logger.info("LLM ENSEMBLE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Cache drive: {args.cache_drive}")
    
    if args.dry_run:
        logger.info("DRY RUN - Configuration validation only")
        return
    
    try:
        # Initialize base configuration
        base_config = Config(
            cache_drive=args.cache_drive,
            custom_cache_dir=args.custom_cache_dir,
            config_file=args.base_config
        )
        
        # Initialize ensemble pipeline
        pipeline = EnsemblePipeline(args.config, base_config)
        
        # Override configuration with command line arguments
        if args.batch_size is not None:
            pipeline.ensemble_config[0].batch_size = args.batch_size
        
        if args.max_concurrent is not None:
            pipeline.ensemble_config[0].max_concurrent = args.max_concurrent
        
        if args.request_delay is not None:
            pipeline.ensemble_config[0].request_delay = args.request_delay
        
        if args.disable_rag:
            pipeline.ensemble_config[0].use_rag = False
        
        if args.rag_examples is not None:
            pipeline.ensemble_config[0].rag_num_examples = args.rag_examples
        
        if args.save_individual:
            pipeline.ensemble_config[0].save_individual_outputs = True
        
        if args.save_judge_reasoning:
            pipeline.ensemble_config[0].save_judge_reasoning = True
        
        # Run ensemble pipeline
        logger.info("Starting ensemble pipeline...")
        results = await pipeline.run_ensemble(args.samples)
        
        # Save results
        pipeline.save_results(str(output_path))
        
        # Print summary
        stats = pipeline.get_summary_stats()
        logger.info("=" * 60)
        logger.info("ENSEMBLE PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Successful: {stats['successful_samples']}")
        logger.info(f"Failed: {stats['failed_samples']}")
        logger.info(f"Average processing time: {stats['average_processing_time']:.2f}s")
        
        if stats['llm_selection_distribution']:
            logger.info("LLM Selection Distribution:")
            for llm, count in stats['llm_selection_distribution'].items():
                percentage = (count / stats['total_samples']) * 100
                logger.info(f"  {llm}: {count} ({percentage:.1f}%)")
        
        if stats['total_tokens']:
            logger.info("Token Usage:")
            for llm, tokens in stats['total_tokens'].items():
                logger.info(f"  {llm}: {tokens['total_tokens']} tokens")
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error running ensemble pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
