"""
Evaluation visualization module for medical image captioning results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu import BLEU, sentence_bleu
from sklearn.metrics import classification_report, f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


class EvaluationVisualizer:
    """Visualizer for medical image captioning evaluation results."""
    
    def __init__(self):
        """Initialize the evaluation visualizer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bleu_scorer = BLEU()
        
    def load_from_jsonl(self, jsonl_file: str) -> List[Dict[str, Any]]:
        """
        Load evaluation data from a JSONL file.
        
        Args:
            jsonl_file (str): Path to the JSONL file with results
            
        Returns:
            List[Dict[str, Any]]: List of evaluation results
        """
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        
        print(f"Loading evaluation data from {jsonl_file}")
        
        results = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(results)} evaluation results")
        return results
    
    def calculate_caption_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate caption evaluation metrics.
        
        Args:
            results (List[Dict[str, Any]]): List of evaluation results
            
        Returns:
            Dict[str, Any]: Caption metrics
        """
        bleu_scores = []
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        detailed_results = []
        
        for result in results:
            if "error" in result:
                continue
                
            ground_truth = result.get("ground_truth_caption", "")
            generated = result.get("generated_caption", "")
            
            if not ground_truth or not generated:
                continue
            
            # BLEU Score
            try:
                bleu_score = sentence_bleu(generated, [ground_truth]).score / 100.0
            except Exception as e:
                print(f"Warning: BLEU calculation failed: {e}")
                bleu_score = 0.0
            
            bleu_scores.append(bleu_score)
            
            # ROUGE Scores
            rouge_result = self.rouge_scorer.score(ground_truth, generated)
            for metric in ["rouge1", "rouge2", "rougeL"]:
                rouge_scores[metric].append(rouge_result[metric].fmeasure)
            
            # Store detailed results
            detailed_results.append({
                "image_id": result.get("image_id", "unknown"),
                "ground_truth": ground_truth,
                "generated": generated,
                "bleu_score": bleu_score,
                "rouge1_f": rouge_result["rouge1"].fmeasure,
                "rouge2_f": rouge_result["rouge2"].fmeasure,
                "rougeL_f": rouge_result["rougeL"].fmeasure,
            })
        
        # Calculate aggregate metrics
        metrics = {
            "num_samples": len(bleu_scores),
            "bleu_mean": np.mean(bleu_scores) if bleu_scores else 0,
            "bleu_std": np.std(bleu_scores) if bleu_scores else 0,
            "bleu_scores": bleu_scores,
            "detailed_results": detailed_results,
        }
        
        # Add ROUGE metrics
        for metric in ["rouge1", "rouge2", "rougeL"]:
            scores = rouge_scores[metric]
            metrics[f"{metric}_mean"] = np.mean(scores) if scores else 0
            metrics[f"{metric}_std"] = np.std(scores) if scores else 0
        
        return metrics
    
    def create_caption_visualizations(self, caption_metrics: Dict[str, Any], output_dir: str) -> None:
        """
        Create caption evaluation visualizations.
        
        Args:
            caption_metrics (Dict[str, Any]): Caption metrics
            output_dir (str): Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use("default")
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Caption Evaluation Metrics", fontsize=16)
        
        # 1. BLEU scores distribution
        bleu_scores = caption_metrics["bleu_scores"]
        if bleu_scores:
            axes[0, 0].hist(bleu_scores, bins=30, alpha=0.7, edgecolor="black")
            axes[0, 0].axvline(
                caption_metrics["bleu_mean"],
                color="red",
                linestyle="--",
                label=f'Mean: {caption_metrics["bleu_mean"]:.3f}'
            )
            axes[0, 0].set_xlabel("BLEU Score")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("BLEU Score Distribution")
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, "No BLEU data available", ha="center", va="center")
            axes[0, 0].set_title("BLEU Score Distribution")
        
        # 2. ROUGE scores comparison
        rouge_metrics = ["rouge1_mean", "rouge2_mean", "rougeL_mean"]
        rouge_values = [caption_metrics[metric] for metric in rouge_metrics]
        rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
        
        bars = axes[0, 1].bar(rouge_labels, rouge_values, alpha=0.7)
        axes[0, 1].set_ylabel("F-measure")
        axes[0, 1].set_title("ROUGE Metrics Comparison")
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, rouge_values):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
        
        # 3. Caption length analysis
        detailed = caption_metrics["detailed_results"]
        if detailed:
            true_lengths = [len(r["ground_truth"].split()) for r in detailed]
            generated_lengths = [len(r["generated"].split()) for r in detailed]
            
            axes[1, 0].scatter(true_lengths, generated_lengths, alpha=0.6)
            max_length = max(max(true_lengths), max(generated_lengths)) if true_lengths and generated_lengths else 100
            axes[1, 0].plot([0, max_length], [0, max_length], "r--", alpha=0.8)
            axes[1, 0].set_xlabel("Ground Truth Length (words)")
            axes[1, 0].set_ylabel("Generated Length (words)")
            axes[1, 0].set_title("Caption Length Comparison")
        else:
            axes[1, 0].text(0.5, 0.5, "No length data available", ha="center", va="center")
            axes[1, 0].set_title("Caption Length Comparison")
        
        # 4. BLEU vs ROUGE correlation
        if detailed:
            bleu_scores = [r["bleu_score"] for r in detailed]
            rouge1_scores = [r["rouge1_f"] for r in detailed]
            
            axes[1, 1].scatter(bleu_scores, rouge1_scores, alpha=0.6)
            axes[1, 1].set_xlabel("BLEU Score")
            axes[1, 1].set_ylabel("ROUGE-1 F-measure")
            axes[1, 1].set_title("BLEU vs ROUGE-1 Correlation")
        else:
            axes[1, 1].text(0.5, 0.5, "No correlation data available", ha="center", va="center")
            axes[1, 1].set_title("BLEU vs ROUGE-1 Correlation")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "caption_metrics.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Caption visualizations saved to {output_dir}/caption_metrics.png")
    
    def create_prediction_examples_visualization(self, caption_metrics: Dict[str, Any], output_dir: str, num_examples: int = 6) -> None:
        """
        Create visualization showing prediction examples.
        
        Args:
            caption_metrics (Dict[str, Any]): Caption metrics
            output_dir (str): Directory to save visualizations
            num_examples (int): Number of examples to show
        """
        os.makedirs(output_dir, exist_ok=True)
        
        detailed = caption_metrics["detailed_results"]
        if not detailed:
            print("No detailed results available for examples")
            return
        
        # Sort by BLEU score for best/worst examples
        detailed_sorted = sorted(detailed, key=lambda x: x["bleu_score"], reverse=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Prediction Examples", fontsize=16)
        
        # Show best and worst examples
        examples_to_show = detailed_sorted[:num_examples//2] + detailed_sorted[-(num_examples//2):]
        
        for i, example in enumerate(examples_to_show[:num_examples]):
            row = i // 3
            col = i % 3
            
            # Create text content
            image_id = example["image_id"]
            bleu_score = example["bleu_score"]
            ground_truth = example["ground_truth"]
            generated = example["generated"]
            
            # Truncate long texts
            max_length = 100
            if len(ground_truth) > max_length:
                ground_truth = ground_truth[:max_length] + "..."
            if len(generated) > max_length:
                generated = generated[:max_length] + "..."
            
            # Create text display
            text_content = f"""
Image ID: {image_id}
BLEU Score: {bleu_score:.3f}

Ground Truth:
{ground_truth}

Generated:
{generated}
            """
            
            axes[row, col].text(0.05, 0.95, text_content, transform=axes[row, col].transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].axis('off')
            
            # Add title
            title = f"Best Example {i+1}" if i < num_examples//2 else f"Worst Example {i+1-num_examples//2}"
            axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_examples.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Prediction examples saved to {output_dir}/prediction_examples.png")
    
    def create_evaluation_report(self, caption_metrics: Dict[str, Any], output_dir: str) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            caption_metrics (Dict[str, Any]): Caption metrics
            output_dir (str): Directory to save the report
            
        Returns:
            str: Path to the saved report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
        
        with open(report_path, "w") as f:
            f.write("# Medical Image Captioning Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Number of Samples:** {caption_metrics['num_samples']}\n\n")
            
            f.write("## Caption Evaluation Results\n\n")
            f.write(f"### BLEU Score\n")
            f.write(f"- **Mean:** {caption_metrics['bleu_mean']:.4f}\n")
            f.write(f"- **Standard Deviation:** {caption_metrics['bleu_std']:.4f}\n\n")
            
            f.write("### ROUGE Scores\n\n")
            f.write("| Metric | F-measure |\n")
            f.write("|--------|----------|\n")
            f.write(f"| ROUGE-1 | {caption_metrics['rouge1_mean']:.4f} |\n")
            f.write(f"| ROUGE-2 | {caption_metrics['rouge2_mean']:.4f} |\n")
            f.write(f"| ROUGE-L | {caption_metrics['rougeL_mean']:.4f} |\n\n")
            
            f.write("## Performance Analysis\n\n")
            
            # Performance interpretation
            bleu_mean = caption_metrics['bleu_mean']
            rouge1_mean = caption_metrics['rouge1_mean']
            
            f.write("### Caption Quality Assessment\n")
            if bleu_mean > 0.3:
                f.write("- **BLEU Score:** Excellent caption quality\n")
            elif bleu_mean > 0.15:
                f.write("- **BLEU Score:** Good caption quality\n")
            elif bleu_mean > 0.05:
                f.write("- **BLEU Score:** Moderate caption quality\n")
            else:
                f.write("- **BLEU Score:** Poor caption quality - needs improvement\n")
            
            if rouge1_mean > 0.4:
                f.write("- **ROUGE-1:** Strong unigram overlap with reference captions\n")
            elif rouge1_mean > 0.25:
                f.write("- **ROUGE-1:** Moderate unigram overlap\n")
            else:
                f.write("- **ROUGE-1:** Limited unigram overlap - needs improvement\n")
            
            f.write("\n## Recommendations\n\n")
            
            if bleu_mean < 0.1:
                f.write("1. **Improve Caption Generation:** Consider fine-tuning the model or using better prompts\n")
            if rouge1_mean < 0.2:
                f.write("2. **Enhance RAG Examples:** Improve the quality and relevance of retrieved examples\n")
            if caption_metrics['bleu_std'] > bleu_mean:
                f.write("3. **Reduce Variability:** High variance suggests inconsistent performance - investigate causes\n")
            
            f.write("4. **Consider Model Upgrades:** Evaluate using larger or specialized medical models\n")
            f.write("5. **Expand Training Data:** Include more diverse medical image examples\n")
            
            f.write(f"\n---\n*Report generated automatically by Evaluation Visualization System*\n")
        
        print(f"Evaluation report saved to {report_path}")
        return report_path
    
    def run_full_evaluation(self, jsonl_file: str, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Run full evaluation with visualizations and report.
        
        Args:
            jsonl_file (str): Path to JSONL file with results
            output_dir (str): Directory to save outputs
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        print("=" * 60)
        print("RUNNING FULL EVALUATION")
        print("=" * 60)
        
        # Load data
        results = self.load_from_jsonl(jsonl_file)
        
        # Calculate metrics
        print("Calculating caption metrics...")
        caption_metrics = self.calculate_caption_metrics(results)
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_caption_visualizations(caption_metrics, output_dir)
        self.create_prediction_examples_visualization(caption_metrics, output_dir)
        
        # Create report
        print("Generating evaluation report...")
        report_path = self.create_evaluation_report(caption_metrics, output_dir)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_path, "w") as f:
            json.dump({
                "evaluation_timestamp": datetime.now().isoformat(),
                "source_file": jsonl_file,
                "caption_metrics": caption_metrics
            }, f, indent=2)
        
        print(f"Detailed results saved to {results_path}")
        print(f"Evaluation complete! Check {output_dir} for all outputs.")
        
        return caption_metrics


def main():
    """Example usage of the EvaluationVisualizer."""
    visualizer = EvaluationVisualizer()
    
    # Example with sample data
    sample_results = [
        {
            "image_id": "test_001",
            "ground_truth_caption": "Chest X-ray showing normal lung fields",
            "generated_caption": "Chest radiograph demonstrating clear lung fields"
        },
        {
            "image_id": "test_002",
            "ground_truth_caption": "MRI scan of the brain showing normal anatomy",
            "generated_caption": "Brain MRI revealing normal anatomical structures"
        }
    ]
    
    # Calculate metrics
    caption_metrics = visualizer.calculate_caption_metrics(sample_results)
    print("Caption Metrics:")
    for key, value in caption_metrics.items():
        if key != "detailed_results":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
