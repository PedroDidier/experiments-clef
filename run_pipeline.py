import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import MedicalImageCaptioningPipeline


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Medical Image Captioning Pipeline with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (300 validation samples, no analysis)
  python run_pipeline.py
  
  # Run with cost analysis and evaluation
  python run_pipeline.py --cost-analysis --evaluation
  
  # Run with custom settings
  python run_pipeline.py --samples 100 --rag-examples 5 --model gpt-4o-mini --cost-analysis --evaluation
  
  # Run only cost analysis on existing results
  python run_pipeline.py --cost-analysis-only responses/responses_rag_hf_2024-01-15_10-30.jsonl
        """
    )
    
    # Pipeline configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
        help="OpenAI model to use (default: gpt-4o)"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=300,
        help="Number of validation samples to process (default: 300)"
    )
    
    parser.add_argument(
        "--rag-examples", 
        type=int, 
        default=3,
        help="Number of RAG examples to use (default: 3)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    # Analysis options
    parser.add_argument(
        "--cost-analysis", 
        action="store_true",
        help="Run cost analysis after generation"
    )
    
    parser.add_argument(
        "--evaluation", 
        action="store_true",
        help="Run evaluation analysis after generation"
    )
    
    # Analysis-only options
    parser.add_argument(
        "--cost-analysis-only", 
        type=str,
        metavar="JSONL_FILE",
        help="Run only cost analysis on existing JSONL file"
    )
    
    parser.add_argument(
        "--evaluation-only", 
        type=str,
        metavar="JSONL_FILE",
        help="Run only evaluation analysis on existing JSONL file"
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
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY in your .env file")
        return 1
    
    # Handle analysis-only modes
    if args.cost_analysis_only:
        print("Running cost analysis only...")
        from src.analysis.cost_analysis import CostAnalyzer
        
        analyzer = CostAnalyzer()
        analyzer.load_from_jsonl(args.cost_analysis_only)
        
        summary = analyzer.get_cost_summary()
        print("Cost Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        analyzer.create_cost_visualizations(args.cost_output_dir)
        analyzer.save_cost_report(args.cost_output_dir)
        
        print(f"Cost analysis complete! Check {args.cost_output_dir} for results.")
        return 0
    
    if args.evaluation_only:
        print("Running evaluation analysis only...")
        from src.analysis.evaluation_visualizer import EvaluationVisualizer
        
        visualizer = EvaluationVisualizer()
        results = visualizer.load_from_jsonl(args.evaluation_only)
        
        caption_metrics = visualizer.calculate_caption_metrics(results)
        print("Caption Metrics:")
        print(f"  BLEU Mean: {caption_metrics['bleu_mean']:.4f}")
        print(f"  ROUGE-1 Mean: {caption_metrics['rouge1_mean']:.4f}")
        print(f"  ROUGE-2 Mean: {caption_metrics['rouge2_mean']:.4f}")
        print(f"  ROUGE-L Mean: {caption_metrics['rougeL_mean']:.4f}")
        
        visualizer.create_caption_visualizations(caption_metrics, args.evaluation_output_dir)
        visualizer.create_prediction_examples_visualization(caption_metrics, args.evaluation_output_dir)
        visualizer.create_evaluation_report(caption_metrics, args.evaluation_output_dir)
        
        print(f"Evaluation analysis complete! Check {args.evaluation_output_dir} for results.")
        return 0
    
    # Run full pipeline
    print("=" * 60)
    print("MEDICAL IMAGE CAPTIONING PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Validation samples: {args.samples}")
    print(f"RAG examples: {args.rag_examples}")
    print(f"Random seed: {args.seed}")
    print(f"Cost analysis: {'Yes' if args.cost_analysis else 'No'}")
    print(f"Evaluation analysis: {'Yes' if args.evaluation else 'No'}")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = MedicalImageCaptioningPipeline(
            model_name=args.model,
            num_validation_samples=args.samples,
            num_rag_examples=args.rag_examples,
            random_seed=args.seed,
            run_cost_analysis=args.cost_analysis,
            run_evaluation=args.evaluation
        )
        
        # Setup pipeline
        pipeline.setup_pipeline()
        
        # Generate captions
        results = pipeline.generate_captions(save_results=True)
        
        # Run analysis if enabled
        if args.cost_analysis:
            pipeline.run_cost_analysis(results, args.cost_output_dir)
        
        if args.evaluation:
            pipeline.run_evaluation_analysis(results, args.evaluation_output_dir)
        
        # Save pipeline state
        pipeline.save_pipeline_state()
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 60)
        print("Check the 'responses' directory for the generated captions JSONL file.")
        print("Check the 'pipeline_state' directory for the saved vector database.")
        if args.cost_analysis:
            print(f"Check the '{args.cost_output_dir}' directory for cost analysis results.")
        if args.evaluation:
            print(f"Check the '{args.evaluation_output_dir}' directory for evaluation results.")
        
        return 0
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
