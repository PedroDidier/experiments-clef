import json
from datetime import datetime
from pathlib import Path


def analyze_costs(jsonl_path: str):
    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    entries = [json.loads(line) for line in lines]

    # Calculate statistics
    total_requests = len(entries)
    total_cost = sum(entry["token_usage"]["cost_usd"] for entry in entries)
    avg_cost = total_cost / total_requests

    # Calculate token statistics
    total_input_tokens = sum(entry["token_usage"]["input_tokens"] for entry in entries)
    total_output_tokens = sum(
        entry["token_usage"]["output_tokens"] for entry in entries
    )
    avg_input_tokens = total_input_tokens / total_requests
    avg_output_tokens = total_output_tokens / total_requests

    # Generate markdown report
    report = f"""# Cost Analysis Report

## Dataset Information
- **File analyzed**: {jsonl_path}
- **Number of requests analyzed**: {total_requests}
- **Date of analysis**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Token Usage Statistics
- **Average input tokens per request**: {avg_input_tokens:.2f}
- **Average output tokens per request**: {avg_output_tokens:.2f}
- **Total input tokens**: {total_input_tokens:,}
- **Total output tokens**: {total_output_tokens:,}

## Cost Analysis
- **Total cost of analyzed requests**: ${total_cost:.2f}
- **Average cost per request**: ${avg_cost:.2f}

## Cost Projections
| Number of Requests | Projected Cost |
|-------------------|----------------|
| 100 requests      | ${avg_cost * 100:.2f} |
| 1,000 requests    | ${avg_cost * 1000:.2f} |
| 10,000 requests   | ${avg_cost * 10000:.2f} |
"""
    # Save the report
    cost_analysis_dir = Path(jsonl_path).parent.parent / "cost_analysis"
    cost_analysis_dir.mkdir(exist_ok=True)

    timestamp = Path(jsonl_path).stem.split("_")[1]
    report_path = cost_analysis_dir / f"cost_analysis_{timestamp}.md"

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Cost analysis report generated: {report_path}")


if __name__ == "__main__":
    analyze_costs(jsonl_path="./responses/responses_2025-04-24_21-57.jsonl")
