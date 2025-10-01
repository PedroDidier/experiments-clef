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
  # Run with default settings (300 validation samples, GPT-5-mini)
  python run_pipeline.py
  
  # Run with cost analysis and evaluation
  python run_pipeline.py --cost-analysis --evaluation
  
  # Run with custom settings
  python run_pipeline.py --samples 100 --rag-examples 5 --model gpt-4o-mini --cost-analysis --evaluation
  
  # Use different cache drive
  python run_pipeline.py --cache-drive D --model gpt-5-mini
  
  # Run only cost analysis on existing results
  python run_pipeline.py --cost-analysis-only responses/responses_rag_hf_2024-01-15_10-30.jsonl
        """
    )
    
    # Load environment variables
    load_dotenv()
    
    # Main pipeline arguments
    parser.add_argument(
        "--samples", 
        type=int, 
        default=300,
        help="Number of validation samples to process (default: 300)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4o-mini", "gpt-5-mini", "gpt-3.5-turbo"],
        help="OpenAI model to use (default: gpt-4o). Note: Only gpt-4o, gpt-4-turbo, and gpt-4 support vision."
    )
    
    parser.add_argument(
        "--rag-examples", 
        type=int, 
        default=3,
        help="Number of RAG examples to use (default: 3)"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Analysis options
    parser.add_argument(
        "--cost-analysis", 
        action="store_true",
        help="Run cost analysis after processing"
    )
    
    parser.add_argument(
        "--evaluation", 
        action="store_true",
        help="Run evaluation analysis after processing"
    )
    
    # Analysis-only modes
    parser.add_argument(
        "--cost-analysis-only", 
        action="store_true",
        help="Run only cost analysis on existing results"
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
        help="Path to YAML configuration file (default: config.yaml or config_local.yaml)"
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
    
    # Set up configuration
    from src.config import update_config
    if args.cache_drive or args.custom_cache_dir:
        update_config(
            cache_drive=args.cache_drive,
            custom_cache_dir=args.custom_cache_dir
        )
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY in your .env file")
        return 1
    
    # Handle analysis-only modes
    if args.cost_analysis_only:
        print("Running cost analysis only...")
        from src.analysis.cost_analysis import CostAnalyzer
        
        # Find the most recent responses file if not specified
        responses_dir = Path("responses")
        if responses_dir.exists():
            jsonl_files = list(responses_dir.glob("responses_rag_hf_*.jsonl"))
            if jsonl_files:
                latest_file = max(jsonl_files, key=lambda x: x.stat().st_mtime)
                print(f"Analyzing: {latest_file}")
                
                analyzer = CostAnalyzer()
                analyzer.analyze_from_jsonl(str(latest_file))
                analyzer.save_report(args.cost_output_dir)
                print("Cost analysis completed!")
            else:
                print("No responses files found in responses/ directory")
                return 1
        else:
            print("Responses directory not found")
            return 1
        
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
    print("=" * 60)
    print("MEDICAL IMAGE CAPTIONING PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Validation samples: {args.samples}")
    print(f"RAG examples: {args.rag_examples}")
    print(f"Random seed: {args.random_seed}")
    print(f"Cache drive: {args.cache_drive}")
    if args.custom_cache_dir:
        print(f"Custom cache dir: {args.custom_cache_dir}")
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
            config_file=args.config
        )
        
        # Run pipeline
        results = pipeline.run()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Processed {len(results)} samples")
        
        if args.cost_analysis:
            print("✓ Cost analysis completed")
        
        if args.evaluation:
            print("✓ Evaluation analysis completed")
        
        print("\nOutput files:")
        print(f"- Responses: responses/responses_rag_hf_*.jsonl")
        if args.cost_analysis:
            print(f"- Cost analysis: {args.cost_output_dir}/")
        if args.evaluation:
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