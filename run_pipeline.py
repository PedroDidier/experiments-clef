import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from src.main import MedicalImageCaptioningPipeline


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Medical Image Captioning Pipeline with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run using a provider config file
  python run_pipeline.py --config config_deepinfra.yaml
  python run_pipeline.py --config config_openai.yaml --evaluation

  # Override the model from the config file
  python run_pipeline.py --config config_google.yaml --model gemini-2.5-pro

  # Retry only the samples that failed on a previous run
  python process_submission.py collect-errors responses/responses_rag_hf_*.jsonl -o error_image_ids.txt
  python run_pipeline.py --config config_deepinfra.yaml --retry-ids error_image_ids.txt

  # Re-run analysis on existing results, without calling any API
  python run_pipeline.py --evaluation-only responses/responses_rag_hf_2026-05-04_16-50.jsonl
  python run_pipeline.py --cost-analysis-only
        """
    )
    
    # Load environment variables
    load_dotenv()
    
    # Main pipeline arguments
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process (default: from config file)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name, passed through to the provider (default: from config file). "
             "Must be a vision-capable model, e.g. gpt-4o, gemini-2.5-pro, "
             "claude-sonnet-4-5-20250929, meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    )

    parser.add_argument(
        "--rag-examples",
        type=int,
        default=None,
        help="Number of RAG examples to retrieve; 0 disables retrieval (default: from config file)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config file)"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of worker threads for parallel API calls (default: 4)"
    )

    parser.add_argument(
        "--retry-ids",
        type=str,
        default=None,
        metavar="TXT_FILE",
        help="Process only the image IDs listed in this file (one per line). "
             "Used to retry samples that failed on a previous run."
    )
    
    # Analysis options
    parser.add_argument(
        "--cost-analysis",
        action="store_true",
        default=None,
        help="Run cost analysis after processing (overrides config)"
    )

    parser.add_argument(
        "--evaluation",
        action="store_true",
        default=None,
        help="Run evaluation analysis after processing (overrides config)"
    )
    
    # Analysis-only modes
    parser.add_argument(
        "--cost-analysis-only",
        nargs="?",
        const=True,
        default=False,
        metavar="JSONL_FILE",
        help="Run only cost analysis. Optionally takes a JSONL file; "
             "defaults to the most recent file in responses/"
    )
    
    parser.add_argument(
        "--evaluation-only", 
        type=str,
        metavar="JSONL_FILE",
        help="Run only evaluation analysis on existing JSONL file"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to YAML configuration file, e.g. config_openai.yaml, "
             "config_google.yaml, config_deepinfra.yaml"
    )
    
    parser.add_argument(
        "--cache-drive", 
        type=str, 
        default=None,
        help="Drive letter for caching (overrides config)"
    )
    
    parser.add_argument(
        "--custom-cache-dir", 
        type=str, 
        default=None,
        help="Custom cache directory path (overrides config)"
    )
    
    # Output directories
    parser.add_argument(
        "--cost-output-dir", 
        type=str, 
        default="cost_analysis",
        help="Directory for cost analysis outputs (default: cost_analysis)"
    )
    
    parser.add_argument(
        "--evaluation-output-dir", 
        type=str, 
        default="evaluation_results",
        help="Directory for evaluation outputs (default: evaluation_results)"
    )
    
    args = parser.parse_args()

    # Apply the chosen config file globally BEFORE building the pipeline: the
    # dataset and vector-database modules read it through get_config().
    from src.config import get_config, set_config, update_config
    if args.config:
        set_config(args.config)
    if args.cache_drive or args.custom_cache_dir:
        update_config(
            cache_drive=args.cache_drive,
            custom_cache_dir=args.custom_cache_dir
        )

    config = get_config()
    provider = config.get_model_config().get('provider', 'openai')

    # Check for the API key that this provider actually needs.
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "deepinfra": "DEEPINFRA_API_TOKEN",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    analysis_only = bool(args.cost_analysis_only or args.evaluation_only)
    required_key = required_keys.get(provider)
    if not analysis_only and required_key and not os.getenv(required_key):
        print(f"Error: provider '{provider}' requires {required_key} in your .env file")
        return 1

    # Handle analysis-only modes
    if args.cost_analysis_only:
        print("Running cost analysis only...")
        from src.analysis.cost_analysis import CostAnalyzer

        if isinstance(args.cost_analysis_only, str):
            target = Path(args.cost_analysis_only)
            if not target.exists():
                print(f"Error: File {target} not found")
                return 1
        else:
            # Fall back to the most recent responses file.
            responses_dir = Path("responses")
            jsonl_files = list(responses_dir.glob("responses_rag_hf_*.jsonl")) if responses_dir.exists() else []
            if not jsonl_files:
                print("No responses files found in responses/ directory")
                return 1
            target = max(jsonl_files, key=lambda x: x.stat().st_mtime)

        print(f"Analyzing: {target}")
        analyzer = CostAnalyzer()
        analyzer.load_from_jsonl(str(target))
        analyzer.create_cost_visualizations(args.cost_output_dir)
        analyzer.save_cost_report(args.cost_output_dir)
        print("Cost analysis completed!")
        return 0

    if args.evaluation_only:
        print(f"Running evaluation analysis on: {args.evaluation_only}")
        from src.analysis.evaluation_visualizer import EvaluationVisualizer
        
        if not os.path.exists(args.evaluation_only):
            print(f"Error: File {args.evaluation_only} not found")
            return 1
        
        evaluator = EvaluationVisualizer()
        results = evaluator.run_full_evaluation(args.evaluation_only, args.evaluation_output_dir)
        print(f"Evaluation completed! Results saved to {args.evaluation_output_dir}")
        print("Evaluation analysis completed!")
        return 0
    
    # Run main pipeline
    model_config = config.get_model_config()
    dataset_config = config.get_dataset_config()
    print("=" * 60)
    print("MEDICAL IMAGE CAPTIONING PIPELINE")
    print("=" * 60)
    print(f"Config file:  {args.config or config.config_file or '<defaults>'}")
    print(f"Provider:     {provider}")
    print(f"Model:        {args.model or model_config.get('name')}")
    print(f"Dataset:      {dataset_config.get('type', 'rocov2')}")
    print(f"RAG examples: {args.rag_examples if args.rag_examples is not None else config.get_rag_config().get('num_examples', 3)}")
    print(f"Cache dir:    {config.cache_dir}")
    if args.retry_ids:
        print(f"Retry IDs:    {args.retry_ids}")
    print("=" * 60)

    try:
        # Initialize pipeline
        pipeline = MedicalImageCaptioningPipeline(
            model_name=args.model,
            num_validation_samples=args.samples,
            num_rag_examples=args.rag_examples,
            random_seed=args.random_seed,
            run_cost_analysis=args.cost_analysis,
            run_evaluation=args.evaluation,
            config_file=args.config,
            num_threads=args.threads,
            retry_ids_file=args.retry_ids
        )

        # Run pipeline
        results = pipeline.run()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Processed {len(results)} samples")
        
        if pipeline.do_cost_analysis:
            print("✓ Cost analysis completed")

        if pipeline.run_evaluation:
            print("✓ Evaluation analysis completed")

        print("\nOutput files:")
        print(f"- Responses: responses/responses_rag_hf_*.jsonl")
        if pipeline.do_cost_analysis:
            print(f"- Cost analysis: {args.cost_output_dir}/")
        if pipeline.run_evaluation:
            print(f"- Evaluation results: {args.evaluation_output_dir}/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())