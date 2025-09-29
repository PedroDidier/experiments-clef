import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Import configuration first to set up environment
from .config import get_config, update_config

from .data.dataset import ROCOv2DataHandler
from .vectordb.image_vectordb import ImageVectorDB
from .llm.llm_utils import MedicalImageCaptioner
from .analysis.cost_analysis import CostAnalyzer
from .analysis.evaluation_visualizer import EvaluationVisualizer

load_dotenv()


class MedicalImageCaptioningPipeline:
    """Main pipeline for medical image captioning with RAG."""
    
    def __init__(
        self, 
        model_name: str = None,
        num_validation_samples: int = None,
        num_rag_examples: int = None,
        random_seed: int = None,
        run_cost_analysis: bool = None,
        run_evaluation: bool = None,
        config_file: str = None
    ):
        """
        Initialize the medical image captioning pipeline.
        
        Args:
            model_name (str): OpenAI model name (overrides config)
            num_validation_samples (int): Number of validation samples to process (overrides config)
            num_rag_examples (int): Number of RAG examples to use (overrides config)
            random_seed (int): Random seed for reproducibility (overrides config)
            run_cost_analysis (bool): Whether to run cost analysis after generation (overrides config)
            run_evaluation (bool): Whether to run evaluation analysis after generation (overrides config)
            config_file (str): Path to configuration file
        """
        # Load configuration
        self.config = get_config()
        if config_file:
            from .config import Config
            self.config = Config(config_file=config_file)
        
        # Get configuration values with overrides
        model_config = self.config.get_model_config()
        dataset_config = self.config.get_dataset_config()
        rag_config = self.config.get_rag_config()
        analysis_config = self.config.get_analysis_config()
        
        self.model_name = model_name or model_config.get('name', 'gpt-4o')
        self.num_validation_samples = num_validation_samples or dataset_config.get('validation_samples', 300)
        self.num_rag_examples = num_rag_examples or rag_config.get('num_examples', 3)
        self.random_seed = random_seed or dataset_config.get('random_seed', 42)
        self.run_cost_analysis = run_cost_analysis if run_cost_analysis is not None else analysis_config.get('enable_cost_analysis', False)
        self.run_evaluation = run_evaluation if run_evaluation is not None else analysis_config.get('enable_evaluation', False)
        
        # Set random seed
        random.seed(random_seed)
        
        # Initialize components
        self.data_handler = ROCOv2DataHandler()
        self.vectordb = ImageVectorDB()
        self.captioner = MedicalImageCaptioner(model_name=model_name)
        
        # Results storage
        self.results = []
        
    def setup_pipeline(self) -> None:
        """Setup the pipeline by loading dataset and building vector database."""
        print("=" * 60)
        print("SETTING UP MEDICAL IMAGE CAPTIONING PIPELINE")
        print("=" * 60)
        
        # Load dataset
        print("1. Loading ROCOv2 dataset from HuggingFace...")
        self.data_handler.load_dataset()
        
        # Get dataset info
        info = self.data_handler.get_dataset_info()
        print(f"   Dataset info: {info}")
        
        # Check if vector database already exists
        config = get_config()
        vectordb_path = config.get_vectordb_path()
        
        if vectordb_path.exists() and (vectordb_path / "image_index.faiss").exists() and (vectordb_path / "metadata.json").exists():
            print("2. Loading existing vector database...")
            print(f"   Found existing vector database at: {vectordb_path}")
            self.vectordb.load(str(vectordb_path))
            print("   Vector database loaded successfully!")
        else:
            print("2. Building vector database from training data...")
            print("   Using memory-efficient streaming to prevent RAM explosion...")
            train_dataset = self.data_handler.get_train_samples_for_vectordb()
            
            # Use a reasonable limit for initial testing to prevent memory issues
            # You can increase this or set to None for full dataset
            max_samples = 10000  # Start with 10k samples, adjust as needed
            print(f"   Processing up to {max_samples} training samples for vector database")
            
            self.vectordb.build_from_huggingface_dataset(
                train_dataset, 
                batch_size=8,  # Smaller batch size for memory efficiency
                max_samples=max_samples
            )
            
            # Save the vector database for future use
            print("3. Saving vector database for future use...")
            self.vectordb.save(str(vectordb_path))
            print(f"   Vector database saved to: {vectordb_path}")
        
        print("Pipeline setup complete!")
        print("=" * 60)
    
    def generate_captions(self, save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Generate captions for validation samples using RAG.
        
        Args:
            save_results (bool): Whether to save results to JSONL file
            
        Returns:
            List[Dict[str, Any]]: List of generated captions with metadata
        """
        print("=" * 60)
        print("GENERATING CAPTIONS WITH RAG")
        print("=" * 60)
        
        # Get validation samples
        print(f"1. Sampling {self.num_validation_samples} validation images...")
        validation_samples = self.data_handler.get_validation_samples(
            num_samples=self.num_validation_samples,
            random_seed=self.random_seed
        )
        
        print(f"   Processing {len(validation_samples)} validation samples")
        
        # Get configuration
        config = get_config()
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        jsonl_path = config.get_responses_path(timestamp)
        
        # Process each validation sample
        results = []
        total_cost = 0.0
        
        for i, sample in enumerate(validation_samples):
            print(f"\n2.{i+1} Processing sample {i+1}/{len(validation_samples)}")
            print(f"   Image ID: {sample['image_id']}")
            
            try:
                # Find similar images for RAG
                similar_images = self.vectordb.search_similar_images(
                    sample['image'], 
                    k=self.num_rag_examples
                )
                
                print(f"   Found {len(similar_images)} similar images for RAG")
                
                # Generate caption with RAG
                caption_data, token_usage, examples_used = self.captioner.generate_caption_with_rag(
                    sample['image'], 
                    similar_images
                )
                
                # Prepare result data
                result = {
                    "image_id": sample['image_id'],
                    "ground_truth_caption": sample['caption'],
                    "generated_caption": caption_data.get('caption', ''),
                    "token_usage": token_usage,
                    "rag_examples": [
                        {
                            "image_id": ex.get('image_id', 'unknown'),
                            "caption": ex['caption'],
                            "similarity_score": ex['similarity_score']
                        }
                        for ex in examples_used
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                total_cost += token_usage.get('cost_usd', 0.0)
                
                # Save to JSONL file
                if save_results:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                print(f"   Generated caption: {caption_data.get('caption', '')[:100]}...")
                print(f"   Token usage: {token_usage}")
                print(f"   Cost: ${token_usage.get('cost_usd', 0.0):.4f}")
                
            except Exception as e:
                print(f"   Error processing sample {sample['image_id']}: {e}")
                
                # Add error result
                error_result = {
                    "image_id": sample['image_id'],
                    "ground_truth_caption": sample['caption'],
                    "generated_caption": f"Error: {str(e)}",
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
                    "rag_examples": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                
                results.append(error_result)
                
                if save_results:
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
        
        # Print summary
        print("\n" + "=" * 60)
        print("CAPTION GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total samples processed: {len(results)}")
        print(f"Successful generations: {len([r for r in results if 'error' not in r])}")
        print(f"Failed generations: {len([r for r in results if 'error' in r])}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Average cost per sample: ${total_cost/len(results):.4f}")
        
        if save_results:
            print(f"Results saved to: {jsonl_path}")
        
        self.results = results
        return results
    
    def save_pipeline_state(self, output_dir: str = None) -> None:
        """
        Save the pipeline state including vector database.
        
        Args:
            output_dir (str): Directory to save pipeline state (uses config if None)
        """
        config = get_config()
        if output_dir is None:
            output_dir = str(config.get_pipeline_state_path())
        
        print(f"Saving pipeline state to {output_dir}...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save vector database
        vectordb_dir = output_path / "vectordb"
        self.vectordb.save(str(vectordb_dir))
        
        # Save pipeline configuration
        pipeline_config = {
            "model_name": self.model_name,
            "num_validation_samples": self.num_validation_samples,
            "num_rag_examples": self.num_rag_examples,
            "random_seed": self.random_seed,
            "timestamp": datetime.now().isoformat()
        }
        
        config_path = output_path / "pipeline_config.json"
        with open(config_path, "w") as f:
            json.dump(pipeline_config, f, indent=2)
        
        print(f"Pipeline state saved to {output_dir}")
    
    def load_pipeline_state(self, state_dir: str = "pipeline_state") -> None:
        """
        Load the pipeline state including vector database.
        
        Args:
            state_dir (str): Directory containing pipeline state
        """
        print(f"Loading pipeline state from {state_dir}...")
        
        # Load vector database
        vectordb_dir = Path(state_dir) / "vectordb"
        if vectordb_dir.exists():
            self.vectordb.load(str(vectordb_dir))
        else:
            print(f"Vector database not found at {vectordb_dir}")
        
        # Load configuration
        config_path = Path(state_dir) / "pipeline_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"Loaded configuration: {config}")
        
        print("Pipeline state loaded")
    
    def run_cost_analysis(self, results: List[Dict[str, Any]], output_dir: str = "cost_analysis") -> None:
        """
        Run cost analysis on the generated results.
        
        Args:
            results (List[Dict[str, Any]]): List of generation results
            output_dir (str): Directory to save cost analysis results
        """
        print("=" * 60)
        print("RUNNING COST ANALYSIS")
        print("=" * 60)
        
        analyzer = CostAnalyzer()
        analyzer.load_from_results(results)
        
        # Get cost summary
        summary = analyzer.get_cost_summary()
        print("Cost Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Create visualizations
        analyzer.create_cost_visualizations(output_dir)
        
        # Save report
        report_path = analyzer.save_cost_report(output_dir)
        
        print(f"Cost analysis complete! Check {output_dir} for results.")
    
    def run_evaluation_analysis(self, results: List[Dict[str, Any]], output_dir: str = "evaluation_results") -> None:
        """
        Run evaluation analysis on the generated results.
        
        Args:
            results (List[Dict[str, Any]]): List of generation results
            output_dir (str): Directory to save evaluation results
        """
        print("=" * 60)
        print("RUNNING EVALUATION ANALYSIS")
        print("=" * 60)
        
        visualizer = EvaluationVisualizer()
        
        # Calculate caption metrics
        caption_metrics = visualizer.calculate_caption_metrics(results)
        
        print("Caption Metrics:")
        print(f"  BLEU Mean: {caption_metrics['bleu_mean']:.4f}")
        print(f"  ROUGE-1 Mean: {caption_metrics['rouge1_mean']:.4f}")
        print(f"  ROUGE-2 Mean: {caption_metrics['rouge2_mean']:.4f}")
        print(f"  ROUGE-L Mean: {caption_metrics['rougeL_mean']:.4f}")
        
        # Create visualizations
        visualizer.create_caption_visualizations(caption_metrics, output_dir)
        visualizer.create_prediction_examples_visualization(caption_metrics, output_dir)
        
        # Create report
        report_path = visualizer.create_evaluation_report(caption_metrics, output_dir)
        
        print(f"Evaluation analysis complete! Check {output_dir} for results.")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the complete medical image captioning pipeline.
        
        Returns:
            List[Dict[str, Any]]: List of generation results
        """
        try:
            # Setup pipeline
            self.setup_pipeline()
            
            # Generate captions
            results = self.generate_captions(save_results=True)
            
            # Run analysis if enabled
            if self.run_cost_analysis:
                self.run_cost_analysis(results)
            
            if self.run_evaluation:
                self.run_evaluation_analysis(results)
            
            # Save pipeline state
            self.save_pipeline_state()
            
            print("\n" + "=" * 60)
            print("PIPELINE EXECUTION COMPLETE")
            print("=" * 60)
            print("Check the 'responses' directory for the generated captions JSONL file.")
            print("Check the 'pipeline_state' directory for the saved vector database.")
            if self.run_cost_analysis:
                print("Check the 'cost_analysis' directory for cost analysis results.")
            if self.run_evaluation:
                print("Check the 'evaluation_results' directory for evaluation results.")
            
            return results
            
        except Exception as e:
            print(f"Error running pipeline: {e}")
            raise


def main():
    """Main function to run the medical image captioning pipeline."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY in your .env file")
        return
    
    # Initialize pipeline
    pipeline = MedicalImageCaptioningPipeline(
        model_name="gpt-4o",
        num_validation_samples=300,
        num_rag_examples=3,
        random_seed=42,
        run_cost_analysis=True,  # Enable cost analysis
        run_evaluation=True      # Enable evaluation analysis
    )
    
    try:
        # Setup pipeline
        pipeline.setup_pipeline()
        
        # Generate captions
        results = pipeline.generate_captions(save_results=True)
        
        # Run analysis if enabled
        if pipeline.run_cost_analysis:
            pipeline.run_cost_analysis(results)
        
        if pipeline.run_evaluation:
            pipeline.run_evaluation_analysis(results)
        
        # Save pipeline state
        pipeline.save_pipeline_state()
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 60)
        print("Check the 'responses' directory for the generated captions JSONL file.")
        print("Check the 'pipeline_state' directory for the saved vector database.")
        if pipeline.run_cost_analysis:
            print("Check the 'cost_analysis' directory for cost analysis results.")
        if pipeline.run_evaluation:
            print("Check the 'evaluation_results' directory for evaluation results.")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
