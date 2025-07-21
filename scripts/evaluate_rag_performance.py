import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
# Evaluation libraries
from rouge_score import rouge_scorer
from sacrebleu import BLEU, sentence_bleu
from sklearn.metrics import (classification_report, f1_score, hamming_loss,
                             jaccard_score, multilabel_confusion_matrix,
                             precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class RAGEvaluator:
    def __init__(self):
        """Initialize the RAG evaluation system."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bleu_scorer = BLEU()

    def load_ground_truth(
        self, captions_file: str, concepts_file: str
    ) -> Tuple[Dict, Dict]:
        """
        Load ground truth captions and concepts.
        
        Args:
            captions_file: Path to test_captions.csv
            concepts_file: Path to test_concepts.csv
            
        Returns:
            Tuple of (captions_dict, concepts_dict)
        """
        # Load captions
        captions_df = pd.read_csv(captions_file)
        captions_dict = dict(zip(captions_df["ID"], captions_df["Caption"]))

        # Load concepts
        concepts_df = pd.read_csv(concepts_file)
        concepts_dict = {}
        for _, row in concepts_df.iterrows():
            image_id = row["ID"]
            cuis = row["CUIs"].split(";") if pd.notna(row["CUIs"]) else []
            concepts_dict[image_id] = cuis

        return captions_dict, concepts_dict

    def load_predictions(self, jsonl_file: str) -> List[Dict]:
        """
        Load predictions from JSONL file.
        
        Args:
            jsonl_file: Path to the JSONL predictions file
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        with open(jsonl_file, "r") as f:
            for line in f:
                pred = json.loads(line.strip())
                # Extract image ID from filename
                image_file = pred["image_file"]
                image_id = (
                    image_file.replace(".jpg", "")
                    .replace(".jpeg", "")
                    .replace(".png", "")
                )
                pred["image_id"] = image_id
                predictions.append(pred)
        return predictions

    def evaluate_captions(
        self, ground_truth: Dict[str, str], predictions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate caption quality using BLEU and ROUGE metrics.
        
        Args:
            ground_truth: Dictionary mapping image_id to ground truth caption
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary containing caption evaluation metrics
        """
        bleu_scores = []
        rouge_scores = defaultdict(list)
        detailed_results = []

        for pred in predictions:
            image_id = pred["image_id"]

            if image_id not in ground_truth:
                print(f"Warning: No ground truth caption found for {image_id}")
                continue

            pred_caption = pred.get("caption", "")
            true_caption = ground_truth[image_id]

            # BLEU Score using sacrebleu (more robust)
            try:
                # sacrebleu expects list of references and hypothesis as strings
                bleu_score = (
                    sentence_bleu(pred_caption, [true_caption]).score / 100.0
                )  # Convert to 0-1 range
            except Exception as e:
                print(f"Warning: BLEU calculation failed for {image_id}: {e}")
                bleu_score = 0.0

            bleu_scores.append(bleu_score)

            # ROUGE Scores
            rouge_result = self.rouge_scorer.score(true_caption, pred_caption)
            for metric, score in rouge_result.items():
                rouge_scores[f"{metric}_precision"].append(score.precision)
                rouge_scores[f"{metric}_recall"].append(score.recall)
                rouge_scores[f"{metric}_fmeasure"].append(score.fmeasure)

            # Store detailed results for analysis
            detailed_results.append(
                {
                    "image_id": image_id,
                    "true_caption": true_caption,
                    "pred_caption": pred_caption,
                    "bleu_score": bleu_score,
                    "rouge1_f": rouge_result["rouge1"].fmeasure,
                    "rouge2_f": rouge_result["rouge2"].fmeasure,
                    "rougeL_f": rouge_result["rougeL"].fmeasure,
                }
            )

        # Calculate aggregate metrics
        caption_metrics = {
            "num_samples": len(bleu_scores),
            "bleu_mean": np.mean(bleu_scores),
            "bleu_std": np.std(bleu_scores),
            "bleu_scores": bleu_scores,
            "detailed_results": detailed_results,
        }

        # Add ROUGE metrics
        for metric, scores in rouge_scores.items():
            caption_metrics[f"{metric}_mean"] = np.mean(scores)
            caption_metrics[f"{metric}_std"] = np.std(scores)

        return caption_metrics

    def evaluate_concepts(
        self, ground_truth: Dict[str, List[str]], predictions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate concept prediction using classification metrics.
        
        Args:
            ground_truth: Dictionary mapping image_id to list of CUIs
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary containing concept evaluation metrics
        """
        true_concepts = []
        pred_concepts = []
        detailed_results = []

        # Collect all unique CUIs for binarization
        all_cuis = set()
        for cuis in ground_truth.values():
            all_cuis.update(cuis)

        for pred in predictions:
            image_id = pred["image_id"]

            if image_id not in ground_truth:
                print(f"Warning: No ground truth concepts found for {image_id}")
                continue

            true_cui_list = ground_truth[image_id]
            pred_cui_list = pred.get("cuis_found", [])

            # Add to collections
            true_concepts.append(true_cui_list)
            pred_concepts.append(pred_cui_list)
            all_cuis.update(true_cui_list)
            all_cuis.update(pred_cui_list)

            # Store detailed results
            detailed_results.append(
                {
                    "image_id": image_id,
                    "true_cuis": true_cui_list,
                    "pred_cuis": pred_cui_list,
                    "true_count": len(true_cui_list),
                    "pred_count": len(pred_cui_list),
                    "intersection": len(set(true_cui_list) & set(pred_cui_list)),
                }
            )

        # Convert to binary format for sklearn metrics
        mlb = MultiLabelBinarizer()
        all_concepts = true_concepts + pred_concepts
        mlb.fit(all_concepts)

        y_true = mlb.transform(true_concepts)
        y_pred = mlb.transform(pred_concepts)

        # Calculate metrics
        concept_metrics = {
            "num_samples": len(true_concepts),
            "num_unique_cuis": len(all_cuis),
            "detailed_results": detailed_results,
        }

        # Micro-averaged metrics (global)
        concept_metrics["precision_micro"] = precision_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        concept_metrics["recall_micro"] = recall_score(
            y_true, y_pred, average="micro", zero_division=0
        )
        concept_metrics["f1_micro"] = f1_score(
            y_true, y_pred, average="micro", zero_division=0
        )

        # Macro-averaged metrics (per-class then averaged)
        concept_metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        concept_metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        concept_metrics["f1_macro"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        # Sample-wise metrics
        concept_metrics["jaccard_micro"] = jaccard_score(
            y_true, y_pred, average="micro"
        )
        concept_metrics["jaccard_macro"] = jaccard_score(
            y_true, y_pred, average="macro"
        )
        concept_metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

        # Exact match accuracy (all CUIs must match)
        exact_matches = sum(
            1
            for i in range(len(true_concepts))
            if set(true_concepts[i]) == set(pred_concepts[i])
        )
        concept_metrics["exact_match_ratio"] = exact_matches / len(true_concepts)

        return concept_metrics

    def create_visualizations(
        self, caption_metrics: Dict, concept_metrics: Dict, output_dir: str
    ):
        """
        Create visualization plots for the evaluation results.
        
        Args:
            caption_metrics: Caption evaluation results
            concept_metrics: Concept evaluation results  
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Caption Metrics Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Caption Evaluation Metrics Distribution", fontsize=16)

        # BLEU scores distribution
        axes[0, 0].hist(
            caption_metrics["bleu_scores"], bins=30, alpha=0.7, edgecolor="black"
        )
        axes[0, 0].axvline(
            caption_metrics["bleu_mean"],
            color="red",
            linestyle="--",
            label=f'Mean: {caption_metrics["bleu_mean"]:.3f}',
        )
        axes[0, 0].set_xlabel("BLEU Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("BLEU Score Distribution")
        axes[0, 0].legend()

        # ROUGE scores comparison
        rouge_metrics = [
            "rouge1_fmeasure_mean",
            "rouge2_fmeasure_mean",
            "rougeL_fmeasure_mean",
        ]
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

        # Caption length analysis
        detailed = caption_metrics["detailed_results"]
        true_lengths = [len(r["true_caption"].split()) for r in detailed]
        pred_lengths = [len(r["pred_caption"].split()) for r in detailed]

        axes[1, 0].scatter(true_lengths, pred_lengths, alpha=0.6)
        axes[1, 0].plot(
            [0, max(max(true_lengths), max(pred_lengths))],
            [0, max(max(true_lengths), max(pred_lengths))],
            "r--",
            alpha=0.8,
        )
        axes[1, 0].set_xlabel("True Caption Length (words)")
        axes[1, 0].set_ylabel("Predicted Caption Length (words)")
        axes[1, 0].set_title("Caption Length Comparison")

        # BLEU vs ROUGE correlation
        bleu_scores = [r["bleu_score"] for r in detailed]
        rouge1_scores = [r["rouge1_f"] for r in detailed]

        axes[1, 1].scatter(bleu_scores, rouge1_scores, alpha=0.6)
        axes[1, 1].set_xlabel("BLEU Score")
        axes[1, 1].set_ylabel("ROUGE-1 F-measure")
        axes[1, 1].set_title("BLEU vs ROUGE-1 Correlation")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "caption_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Concept Metrics Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Concept Evaluation Metrics", fontsize=16)

        # Precision, Recall, F1 comparison
        micro_metrics = [
            concept_metrics["precision_micro"],
            concept_metrics["recall_micro"],
            concept_metrics["f1_micro"],
        ]
        macro_metrics = [
            concept_metrics["precision_macro"],
            concept_metrics["recall_macro"],
            concept_metrics["f1_macro"],
        ]

        x = np.arange(3)
        width = 0.35

        axes[0, 0].bar(
            x - width / 2, micro_metrics, width, label="Micro-avg", alpha=0.7
        )
        axes[0, 0].bar(
            x + width / 2, macro_metrics, width, label="Macro-avg", alpha=0.7
        )
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Precision, Recall, F1-Score")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(["Precision", "Recall", "F1-Score"])
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)

        # Add value labels
        for i, (micro, macro) in enumerate(zip(micro_metrics, macro_metrics)):
            axes[0, 0].text(
                i - width / 2, micro + 0.01, f"{micro:.3f}", ha="center", va="bottom"
            )
            axes[0, 0].text(
                i + width / 2, macro + 0.01, f"{macro:.3f}", ha="center", va="bottom"
            )

        # CUI count distribution
        detailed_concepts = concept_metrics["detailed_results"]
        true_counts = [r["true_count"] for r in detailed_concepts]
        pred_counts = [r["pred_count"] for r in detailed_concepts]

        axes[0, 1].hist(
            true_counts, bins=20, alpha=0.7, label="True", edgecolor="black"
        )
        axes[0, 1].hist(
            pred_counts, bins=20, alpha=0.7, label="Predicted", edgecolor="black"
        )
        axes[0, 1].set_xlabel("Number of CUIs")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("CUI Count Distribution")
        axes[0, 1].legend()

        # Intersection analysis
        intersections = [r["intersection"] for r in detailed_concepts]
        intersection_ratios = [
            r["intersection"] / max(r["true_count"], 1) for r in detailed_concepts
        ]

        axes[1, 0].hist(intersection_ratios, bins=20, alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("Intersection Ratio (Intersection / True Count)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("CUI Intersection Ratio Distribution")

        # Performance summary
        summary_metrics = {
            "Exact Match": concept_metrics["exact_match_ratio"],
            "Jaccard (Micro)": concept_metrics["jaccard_micro"],
            "F1 (Micro)": concept_metrics["f1_micro"],
            "Hamming Loss": concept_metrics["hamming_loss"],
        }

        metric_names = list(summary_metrics.keys())
        metric_values = list(summary_metrics.values())

        bars = axes[1, 1].bar(metric_names, metric_values, alpha=0.7)
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Concept Performance Summary")
        axes[1, 1].set_ylim(0, 1)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add value labels
        for bar, value in zip(bars, metric_values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "concept_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Example predictions visualization
        self.plot_prediction_examples(caption_metrics, concept_metrics, output_dir)

    def plot_prediction_examples(
        self, caption_metrics: Dict, concept_metrics: Dict, output_dir: str
    ):
        """
        Create plots showing specific prediction examples.
        """
        caption_detailed = caption_metrics["detailed_results"]
        concept_detailed = concept_metrics["detailed_results"]

        # Sort by performance for examples
        caption_detailed.sort(key=lambda x: x["bleu_score"], reverse=True)

        # Create examples plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle("Prediction Examples", fontsize=16)

        # Best and worst caption examples
        examples_to_show = [
            ("Best Caption Predictions", caption_detailed[:3]),
            ("Worst Caption Predictions", caption_detailed[-3:]),
        ]

        for col, (title, examples) in enumerate(examples_to_show):
            axes[0, col].text(
                0.5,
                0.9,
                title,
                transform=axes[0, col].transAxes,
                fontsize=14,
                ha="center",
                weight="bold",
            )

            y_pos = 0.8
            for i, example in enumerate(examples):
                # Image ID and BLEU score
                axes[0, col].text(
                    0.05,
                    y_pos,
                    f"Image: {example['image_id']}",
                    transform=axes[0, col].transAxes,
                    fontsize=10,
                    weight="bold",
                )
                axes[0, col].text(
                    0.05,
                    y_pos - 0.05,
                    f"BLEU: {example['bleu_score']:.3f}",
                    transform=axes[0, col].transAxes,
                    fontsize=10,
                )

                # True caption
                axes[0, col].text(
                    0.05,
                    y_pos - 0.1,
                    "True:",
                    transform=axes[0, col].transAxes,
                    fontsize=10,
                    weight="bold",
                    color="green",
                )
                true_wrapped = self.wrap_text(example["true_caption"], 60)
                axes[0, col].text(
                    0.05,
                    y_pos - 0.15,
                    true_wrapped,
                    transform=axes[0, col].transAxes,
                    fontsize=9,
                    color="green",
                )

                # Predicted caption
                axes[0, col].text(
                    0.05,
                    y_pos - 0.22,
                    "Pred:",
                    transform=axes[0, col].transAxes,
                    fontsize=10,
                    weight="bold",
                    color="blue",
                )
                pred_wrapped = self.wrap_text(example["pred_caption"], 60)
                axes[0, col].text(
                    0.05,
                    y_pos - 0.27,
                    pred_wrapped,
                    transform=axes[0, col].transAxes,
                    fontsize=9,
                    color="blue",
                )

                y_pos -= 0.35

            axes[0, col].set_xlim(0, 1)
            axes[0, col].set_ylim(0, 1)
            axes[0, col].axis("off")

        # Performance distribution by score ranges
        bleu_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        range_counts = []
        range_labels = []

        for low, high in bleu_ranges:
            count = sum(1 for r in caption_detailed if low <= r["bleu_score"] < high)
            range_counts.append(count)
            range_labels.append(f"{low}-{high}")

        axes[1, 0].bar(range_labels, range_counts, alpha=0.7)
        axes[1, 0].set_xlabel("BLEU Score Range")
        axes[1, 0].set_ylabel("Number of Samples")
        axes[1, 0].set_title("Distribution of BLEU Scores")

        # Concept prediction accuracy by number of true CUIs
        concept_accuracy_by_count = defaultdict(list)
        for r in concept_detailed:
            true_count = r["true_count"]
            if true_count > 0:
                accuracy = r["intersection"] / true_count
                concept_accuracy_by_count[true_count].append(accuracy)

        counts = sorted(concept_accuracy_by_count.keys())
        avg_accuracies = [np.mean(concept_accuracy_by_count[c]) for c in counts]

        axes[1, 1].plot(counts, avg_accuracies, "o-", alpha=0.7, markersize=8)
        axes[1, 1].set_xlabel("Number of True CUIs")
        axes[1, 1].set_ylabel("Average Recall")
        axes[1, 1].set_title("Concept Recall by Number of True CUIs")
        axes[1, 1].grid(True, alpha=0.3)

        # Token usage analysis
        predictions = caption_metrics.get("predictions", [])
        if predictions:
            token_costs = [
                p.get("token_usage", {}).get("cost_usd", 0) for p in predictions
            ]
            input_tokens = [
                p.get("token_usage", {}).get("input_tokens", 0) for p in predictions
            ]

            axes[2, 0].scatter(input_tokens, token_costs, alpha=0.6)
            axes[2, 0].set_xlabel("Input Tokens")
            axes[2, 0].set_ylabel("Cost (USD)")
            axes[2, 0].set_title("Token Usage vs Cost")

            # Cost distribution
            axes[2, 1].hist(token_costs, bins=20, alpha=0.7, edgecolor="black")
            axes[2, 1].set_xlabel("Cost per Prediction (USD)")
            axes[2, 1].set_ylabel("Frequency")
            axes[2, 1].set_title("Cost Distribution")
        else:
            axes[2, 0].text(
                0.5,
                0.5,
                "No token usage data available",
                transform=axes[2, 0].transAxes,
                ha="center",
                va="center",
            )
            axes[2, 1].text(
                0.5,
                0.5,
                "No token usage data available",
                transform=axes[2, 1].transAxes,
                ha="center",
                va="center",
            )
            axes[2, 0].axis("off")
            axes[2, 1].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "prediction_examples.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def wrap_text(self, text: str, width: int) -> str:
        """Helper function to wrap text for display."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def save_results(
        self,
        caption_metrics: Dict,
        concept_metrics: Dict,
        output_dir: str,
        jsonl_file: str,
    ) -> str:
        """
        Save evaluation results to files.
        
        Args:
            caption_metrics: Caption evaluation results
            concept_metrics: Concept evaluation results
            output_dir: Directory to save results
            jsonl_file: Original JSONL file name for reference
            
        Returns:
            Path to the saved results file
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create comprehensive results dictionary
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "source_file": jsonl_file,
            "caption_metrics": {
                "num_samples": caption_metrics["num_samples"],
                "bleu_mean": caption_metrics["bleu_mean"],
                "bleu_std": caption_metrics["bleu_std"],
                "rouge1_fmeasure_mean": caption_metrics["rouge1_fmeasure_mean"],
                "rouge2_fmeasure_mean": caption_metrics["rouge2_fmeasure_mean"],
                "rougeL_fmeasure_mean": caption_metrics["rougeL_fmeasure_mean"],
                "rouge1_precision_mean": caption_metrics["rouge1_precision_mean"],
                "rouge1_recall_mean": caption_metrics["rouge1_recall_mean"],
                "rouge2_precision_mean": caption_metrics["rouge2_precision_mean"],
                "rouge2_recall_mean": caption_metrics["rouge2_recall_mean"],
                "rougeL_precision_mean": caption_metrics["rougeL_precision_mean"],
                "rougeL_recall_mean": caption_metrics["rougeL_recall_mean"],
            },
            "concept_metrics": {
                "num_samples": concept_metrics["num_samples"],
                "num_unique_cuis": concept_metrics["num_unique_cuis"],
                "precision_micro": concept_metrics["precision_micro"],
                "recall_micro": concept_metrics["recall_micro"],
                "f1_micro": concept_metrics["f1_micro"],
                "precision_macro": concept_metrics["precision_macro"],
                "recall_macro": concept_metrics["recall_macro"],
                "f1_macro": concept_metrics["f1_macro"],
                "jaccard_micro": concept_metrics["jaccard_micro"],
                "jaccard_macro": concept_metrics["jaccard_macro"],
                "hamming_loss": concept_metrics["hamming_loss"],
                "exact_match_ratio": concept_metrics["exact_match_ratio"],
            },
        }

        # Save main results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save detailed results
        detailed_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        detailed_results = {
            "caption_detailed": caption_metrics["detailed_results"],
            "concept_detailed": concept_metrics["detailed_results"],
        }

        with open(detailed_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Create markdown report
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
        self.create_markdown_report(results, report_file)

        print(f"Results saved to {output_dir}")
        print(f"  - Main results: {results_file}")
        print(f"  - Detailed results: {detailed_file}")
        print(f"  - Markdown report: {report_file}")

        return results_file

    def create_markdown_report(self, results: Dict, report_file: str):
        """Create a markdown report of the evaluation results."""

        caption_metrics = results["caption_metrics"]
        concept_metrics = results["concept_metrics"]

        report = f"""# RAG Pipeline Evaluation Report

**Evaluation Date:** {results['evaluation_timestamp']}  
**Source File:** {results['source_file']}

## Summary

This report presents the evaluation results for the Retrieval-Augmented Generation (RAG) pipeline for medical image captioning and concept extraction.

### Dataset Overview
- **Caption Samples Evaluated:** {caption_metrics['num_samples']}
- **Concept Samples Evaluated:** {concept_metrics['num_samples']}
- **Unique CUIs in Dataset:** {concept_metrics['num_unique_cuis']}

## Caption Evaluation Results

### BLEU Score
- **Mean:** {caption_metrics['bleu_mean']:.4f}
- **Standard Deviation:** {caption_metrics['bleu_std']:.4f}

### ROUGE Scores

| Metric | Precision | Recall | F-measure |
|--------|-----------|--------|-----------|
| ROUGE-1 | {caption_metrics['rouge1_precision_mean']:.4f} | {caption_metrics['rouge1_recall_mean']:.4f} | {caption_metrics['rouge1_fmeasure_mean']:.4f} |
| ROUGE-2 | {caption_metrics['rouge2_precision_mean']:.4f} | {caption_metrics['rouge2_recall_mean']:.4f} | {caption_metrics['rouge2_fmeasure_mean']:.4f} |
| ROUGE-L | {caption_metrics['rougeL_precision_mean']:.4f} | {caption_metrics['rougeL_recall_mean']:.4f} | {caption_metrics['rougeL_fmeasure_mean']:.4f} |

## Concept Evaluation Results

### Classification Metrics

| Metric | Micro-averaged | Macro-averaged |
|--------|----------------|----------------|
| Precision | {concept_metrics['precision_micro']:.4f} | {concept_metrics['precision_macro']:.4f} |
| Recall | {concept_metrics['recall_micro']:.4f} | {concept_metrics['recall_macro']:.4f} |
| F1-Score | {concept_metrics['f1_micro']:.4f} | {concept_metrics['f1_macro']:.4f} |

### Additional Metrics
- **Exact Match Ratio:** {concept_metrics['exact_match_ratio']:.4f}
- **Jaccard Score (Micro):** {concept_metrics['jaccard_micro']:.4f}
- **Jaccard Score (Macro):** {concept_metrics['jaccard_macro']:.4f}
- **Hamming Loss:** {concept_metrics['hamming_loss']:.4f}

## Interpretation

### Caption Performance
- The BLEU score of {caption_metrics['bleu_mean']:.4f} indicates {'good' if caption_metrics['bleu_mean'] > 0.3 else 'moderate' if caption_metrics['bleu_mean'] > 0.15 else 'room for improvement in'} caption quality.
- ROUGE-1 F-measure of {caption_metrics['rouge1_fmeasure_mean']:.4f} shows {'strong' if caption_metrics['rouge1_fmeasure_mean'] > 0.4 else 'moderate' if caption_metrics['rouge1_fmeasure_mean'] > 0.25 else 'limited'} unigram overlap with reference captions.
- ROUGE-L F-measure of {caption_metrics['rougeL_fmeasure_mean']:.4f} indicates {'good' if caption_metrics['rougeL_fmeasure_mean'] > 0.35 else 'moderate' if caption_metrics['rougeL_fmeasure_mean'] > 0.2 else 'limited'} structural similarity.

### Concept Performance
- Micro-averaged F1 of {concept_metrics['f1_micro']:.4f} shows {'strong' if concept_metrics['f1_micro'] > 0.6 else 'moderate' if concept_metrics['f1_micro'] > 0.4 else 'limited'} overall concept extraction performance.
- Exact match ratio of {concept_metrics['exact_match_ratio']:.4f} indicates that {concept_metrics['exact_match_ratio']*100:.1f}% of predictions exactly match the ground truth concept sets.
- Hamming loss of {concept_metrics['hamming_loss']:.4f} shows the fraction of labels that are incorrectly predicted.

## Recommendations

Based on the evaluation results:

1. **Caption Quality:** {'Continue with current approach' if caption_metrics['bleu_mean'] > 0.25 else 'Consider improving caption generation with more diverse training examples or fine-tuning'}
2. **Concept Extraction:** {'Performance is satisfactory' if concept_metrics['f1_micro'] > 0.5 else 'Consider improving concept extraction with better CUI mapping or additional training data'}
3. **Overall System:** {'The RAG approach shows promising results' if caption_metrics['bleu_mean'] > 0.2 and concept_metrics['f1_micro'] > 0.4 else 'Consider refinements to the RAG retrieval and generation components'}

---
*Report generated automatically by RAG Evaluation System*
"""

        with open(report_file, "w") as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline performance")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the JSONL predictions file",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="test_captions.csv",
        help="Path to test captions CSV file",
    )
    parser.add_argument(
        "--concepts_file",
        type=str,
        default="test_concepts.csv",
        help="Path to test concepts CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    if not os.path.exists(args.captions_file):
        raise FileNotFoundError(f"Captions file not found: {args.captions_file}")
    if not os.path.exists(args.concepts_file):
        raise FileNotFoundError(f"Concepts file not found: {args.concepts_file}")

    # Initialize evaluator
    evaluator = RAGEvaluator()

    print("Loading ground truth data...")
    captions_gt, concepts_gt = evaluator.load_ground_truth(
        args.captions_file, args.concepts_file
    )

    print("Loading predictions...")
    predictions = evaluator.load_predictions(args.predictions)

    print(f"Evaluating {len(predictions)} predictions...")

    # Evaluate captions
    print("Evaluating captions...")
    caption_metrics = evaluator.evaluate_captions(captions_gt, predictions)

    # Evaluate concepts
    print("Evaluating concepts...")
    concept_metrics = evaluator.evaluate_concepts(concepts_gt, predictions)

    # Store predictions for token analysis
    caption_metrics["predictions"] = predictions

    # Create visualizations
    print("Creating visualizations...")
    evaluator.create_visualizations(caption_metrics, concept_metrics, args.output_dir)

    # Save results
    print("Saving results...")
    results_file = evaluator.save_results(
        caption_metrics, concept_metrics, args.output_dir, args.predictions
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Caption Metrics:")
    print(
        f"  BLEU Score: {caption_metrics['bleu_mean']:.4f} ± {caption_metrics['bleu_std']:.4f}"
    )
    print(f"  ROUGE-1 F1: {caption_metrics['rouge1_fmeasure_mean']:.4f}")
    print(f"  ROUGE-2 F1: {caption_metrics['rouge2_fmeasure_mean']:.4f}")
    print(f"  ROUGE-L F1: {caption_metrics['rougeL_fmeasure_mean']:.4f}")
    print(f"\nConcept Metrics:")
    print(f"  Precision (Micro): {concept_metrics['precision_micro']:.4f}")
    print(f"  Recall (Micro): {concept_metrics['recall_micro']:.4f}")
    print(f"  F1 (Micro): {concept_metrics['f1_micro']:.4f}")
    print(f"  Exact Match: {concept_metrics['exact_match_ratio']:.4f}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
