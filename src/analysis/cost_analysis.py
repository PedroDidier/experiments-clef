"""
Cost analysis module for medical image captioning pipeline.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CostAnalyzer:
    """Analyzer for cost tracking and analysis of LLM API usage."""
    
    def __init__(self):
        """Initialize the cost analyzer."""
        self.results = []
        self.total_cost = 0.0
        self.total_tokens = 0
        
    def load_from_jsonl(self, jsonl_file: str) -> None:
        """
        Load cost data from a JSONL file.
        
        Args:
            jsonl_file (str): Path to the JSONL file with results
        """
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        
        print(f"Loading cost data from {jsonl_file}")
        
        self.results = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    self.results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(self.results)} results")
        
        # Calculate totals
        self._calculate_totals()
    
    def load_from_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Load cost data from results list.
        
        Args:
            results (List[Dict[str, Any]]): List of result dictionaries
        """
        self.results = results
        self._calculate_totals()
    
    def _calculate_totals(self) -> None:
        """Calculate total cost and token usage."""
        self.total_cost = 0.0
        self.total_tokens = 0
        
        for result in self.results:
            token_usage = result.get("token_usage", {})
            if token_usage:
                self.total_cost += token_usage.get("cost_usd", 0.0)
                self.total_tokens += token_usage.get("total_tokens", 0)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of cost analysis.
        
        Returns:
            Dict[str, Any]: Cost summary statistics
        """
        if not self.results:
            return {"error": "No data loaded"}
        
        # Calculate statistics
        costs = [r.get("token_usage", {}).get("cost_usd", 0.0) for r in self.results]
        input_tokens = [r.get("token_usage", {}).get("input_tokens", 0) for r in self.results]
        output_tokens = [r.get("token_usage", {}).get("output_tokens", 0) for r in self.results]
        total_tokens = [r.get("token_usage", {}).get("total_tokens", 0) for r in self.results]
        
        # Filter out zero values for meaningful statistics
        costs_nonzero = [c for c in costs if c > 0]
        input_tokens_nonzero = [t for t in input_tokens if t > 0]
        output_tokens_nonzero = [t for t in output_tokens if t > 0]
        total_tokens_nonzero = [t for t in total_tokens if t > 0]
        
        summary = {
            "total_requests": len(self.results),
            "successful_requests": len(costs_nonzero),
            "failed_requests": len(self.results) - len(costs_nonzero),
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "average_cost_per_request": self.total_cost / len(self.results) if self.results else 0,
            "average_cost_per_successful_request": sum(costs_nonzero) / len(costs_nonzero) if costs_nonzero else 0,
            "average_input_tokens": sum(input_tokens_nonzero) / len(input_tokens_nonzero) if input_tokens_nonzero else 0,
            "average_output_tokens": sum(output_tokens_nonzero) / len(output_tokens_nonzero) if output_tokens_nonzero else 0,
            "average_total_tokens": sum(total_tokens_nonzero) / len(total_tokens_nonzero) if total_tokens_nonzero else 0,
            "min_cost": min(costs_nonzero) if costs_nonzero else 0,
            "max_cost": max(costs_nonzero) if costs_nonzero else 0,
            "cost_std": pd.Series(costs_nonzero).std() if costs_nonzero else 0,
        }
        
        return summary
    
    def create_cost_projections(self, target_requests: List[int] = None) -> Dict[int, float]:
        """
        Create cost projections for different numbers of requests.
        
        Args:
            target_requests (List[int]): List of target request numbers for projections
            
        Returns:
            Dict[int, float]: Projected costs for each target
        """
        if not self.results:
            return {}
        
        if target_requests is None:
            target_requests = [100, 1000, 10000, 50000, 100000]
        
        avg_cost_per_request = self.total_cost / len(self.results) if self.results else 0
        
        projections = {}
        for target in target_requests:
            projections[target] = target * avg_cost_per_request
        
        return projections
    
    def create_cost_visualizations(self, output_dir: str = "cost_analysis") -> None:
        """
        Create cost analysis visualizations.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        if not self.results:
            print("No data available for visualization")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use("default")
        sns.set_palette("husl")
        
        # Extract data
        costs = [r.get("token_usage", {}).get("cost_usd", 0.0) for r in self.results]
        input_tokens = [r.get("token_usage", {}).get("input_tokens", 0) for r in self.results]
        output_tokens = [r.get("token_usage", {}).get("output_tokens", 0) for r in self.results]
        total_tokens = [r.get("token_usage", {}).get("total_tokens", 0) for r in self.results]
        
        # Filter out zero values
        costs_nonzero = [c for c in costs if c > 0]
        input_tokens_nonzero = [t for t in input_tokens if t > 0]
        output_tokens_nonzero = [t for t in output_tokens if t > 0]
        total_tokens_nonzero = [t for t in total_tokens if t > 0]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Cost Analysis Dashboard", fontsize=16)
        
        # 1. Cost distribution
        if costs_nonzero:
            axes[0, 0].hist(costs_nonzero, bins=30, alpha=0.7, edgecolor="black")
            axes[0, 0].axvline(
                sum(costs_nonzero) / len(costs_nonzero),
                color="red",
                linestyle="--",
                label=f'Mean: ${sum(costs_nonzero) / len(costs_nonzero):.4f}'
            )
            axes[0, 0].set_xlabel("Cost per Request (USD)")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Cost Distribution")
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, "No cost data available", ha="center", va="center")
            axes[0, 0].set_title("Cost Distribution")
        
        # 2. Token usage vs cost
        if costs_nonzero and total_tokens_nonzero:
            axes[0, 1].scatter(total_tokens_nonzero, costs_nonzero, alpha=0.6)
            axes[0, 1].set_xlabel("Total Tokens")
            axes[0, 1].set_ylabel("Cost (USD)")
            axes[0, 1].set_title("Token Usage vs Cost")
        else:
            axes[0, 1].text(0.5, 0.5, "No token/cost data available", ha="center", va="center")
            axes[0, 1].set_title("Token Usage vs Cost")
        
        # 3. Input vs Output tokens
        if input_tokens_nonzero and output_tokens_nonzero:
            axes[1, 0].scatter(input_tokens_nonzero, output_tokens_nonzero, alpha=0.6)
            axes[1, 0].set_xlabel("Input Tokens")
            axes[1, 0].set_ylabel("Output Tokens")
            axes[1, 0].set_title("Input vs Output Tokens")
        else:
            axes[1, 0].text(0.5, 0.5, "No token data available", ha="center", va="center")
            axes[1, 0].set_title("Input vs Output Tokens")
        
        # 4. Cost over time (if timestamps available)
        timestamps = []
        costs_with_time = []
        for result in self.results:
            if "timestamp" in result and result.get("token_usage", {}).get("cost_usd", 0) > 0:
                try:
                    timestamp = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
                    timestamps.append(timestamp)
                    costs_with_time.append(result["token_usage"]["cost_usd"])
                except:
                    continue
        
        if timestamps and costs_with_time:
            axes[1, 1].plot(timestamps, costs_with_time, marker="o", alpha=0.7)
            axes[1, 1].set_xlabel("Time")
            axes[1, 1].set_ylabel("Cost per Request (USD)")
            axes[1, 1].set_title("Cost Over Time")
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, "No timestamp data available", ha="center", va="center")
            axes[1, 1].set_title("Cost Over Time")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cost_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Cost visualizations saved to {output_dir}/cost_analysis.png")
    
    def save_cost_report(self, output_dir: str = "cost_analysis") -> str:
        """
        Save a detailed cost analysis report.
        
        Args:
            output_dir (str): Directory to save the report
            
        Returns:
            str: Path to the saved report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get summary
        summary = self.get_cost_summary()
        
        # Get projections
        projections = self.create_cost_projections()
        
        # Create report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        report_path = os.path.join(output_dir, f"cost_analysis_{timestamp}.md")
        
        with open(report_path, "w") as f:
            f.write("# Cost Analysis Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Total Requests Analyzed:** {summary['total_requests']}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Cost:** ${summary['total_cost_usd']:.4f}\n")
            f.write(f"- **Total Tokens:** {summary['total_tokens']:,}\n")
            f.write(f"- **Successful Requests:** {summary['successful_requests']}\n")
            f.write(f"- **Failed Requests:** {summary['failed_requests']}\n")
            f.write(f"- **Average Cost per Request:** ${summary['average_cost_per_request']:.4f}\n")
            f.write(f"- **Average Cost per Successful Request:** ${summary['average_cost_per_successful_request']:.4f}\n\n")
            
            f.write("## Token Usage Statistics\n\n")
            f.write(f"- **Average Input Tokens:** {summary['average_input_tokens']:.1f}\n")
            f.write(f"- **Average Output Tokens:** {summary['average_output_tokens']:.1f}\n")
            f.write(f"- **Average Total Tokens:** {summary['average_total_tokens']:.1f}\n\n")
            
            f.write("## Cost Distribution\n\n")
            f.write(f"- **Minimum Cost:** ${summary['min_cost']:.4f}\n")
            f.write(f"- **Maximum Cost:** ${summary['max_cost']:.4f}\n")
            f.write(f"- **Standard Deviation:** ${summary['cost_std']:.4f}\n\n")
            
            f.write("## Cost Projections\n\n")
            f.write("| Number of Requests | Projected Cost |\n")
            f.write("|-------------------|----------------|\n")
            for requests, cost in projections.items():
                f.write(f"| {requests:,} requests      | ${cost:.2f} |\n")
            
            f.write("\n## Recommendations\n\n")
            if summary['average_cost_per_request'] > 0.01:
                f.write("- Consider using a smaller model for cost optimization\n")
            if summary['failed_requests'] > 0:
                f.write("- Review failed requests to improve success rate\n")
            if summary['cost_std'] > summary['average_cost_per_request']:
                f.write("- High cost variance suggests inconsistent token usage\n")
            
            f.write(f"\n---\n*Report generated automatically by Cost Analysis System*\n")
        
        print(f"Cost analysis report saved to {report_path}")
        return report_path


def main():
    """Example usage of the CostAnalyzer."""
    analyzer = CostAnalyzer()
    
    # Example with sample data
    sample_results = [
        {
            "image_id": "test_001",
            "token_usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "cost_usd": 0.012
            }
        },
        {
            "image_id": "test_002", 
            "token_usage": {
                "input_tokens": 1200,
                "output_tokens": 250,
                "total_tokens": 1450,
                "cost_usd": 0.015
            }
        }
    ]
    
    analyzer.load_from_results(sample_results)
    
    # Get summary
    summary = analyzer.get_cost_summary()
    print("Cost Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Create visualizations
    analyzer.create_cost_visualizations()
    
    # Save report
    analyzer.save_cost_report()


if __name__ == "__main__":
    main()
