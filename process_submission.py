"""Post-processing utilities for turning raw pipeline output into ImageCLEF submissions.

Three stages, in the order they are normally used:

1. ``collect-errors``  - list the image IDs that failed, to feed back into
                         ``run_pipeline.py --retry-ids``.
2. ``parse``           - repair model output into a clean ``ID,Caption`` CSV.
3. ``merge``           - combine several CSVs into one submission, preferring
                         the first non-error caption for each ID.
"""

import argparse
import csv
import json
import re
import sys
from typing import List

import pandas as pd

# Counters for the two JSON-repair paths, reported after parsing.
case_1 = 0
case_2 = 0

# Substrings that mark a generated caption as a failed generation rather than
# real model output. The pipeline writes "Error: <exception>" when a sample
# raises, and provider errors sometimes come back as prose in the caption field.
ERROR_MARKERS = ("deepinfra", "error:", "rate limit", "internal server error")


def is_error_caption(caption: object) -> bool:
    """Return True when a caption is a failure rather than a real generation."""
    if caption is None or not isinstance(caption, str) or not caption.strip():
        return True
    lowered = caption.strip().lower()
    if lowered == "error":
        return True
    return any(marker in lowered for marker in ERROR_MARKERS)


def parse_json_response(content: str) -> str:
    """Extract the caption from a model response, repairing malformed JSON.

    Returns the caption string, or "error" when nothing usable can be recovered.
    """
    if is_error_caption(content):
        return "error"

    content = content.strip().replace('\n', ' ').replace('\t', ' ')

    # Find the last JSON object in the content (greedy, to survive nested
    # braces from LaTeX or markdown fences in the model output).
    json_matches = re.findall(r'\{.*\}', content, re.DOTALL)
    global case_1, case_2

    if json_matches:
        json_str = json_matches[-1]
        try:
            parsed = json.loads(json_str)
            case_1 += 1
            return parsed.get("caption", json_str)
        except json.JSONDecodeError:
            # Fallback 1: extract the caption with a proper closing quote.
            match = re.search(r'"caption"\s*:\s*"([^"]*)"', json_str, re.DOTALL)
            if match:
                case_2 += 1
                return match.group(1).replace('\n', ' ').replace('\t', ' ').strip()

            # Fallback 2: the caption was never closed - take everything up to
            # the end of the string or the next closing brace.
            match_incomplete = re.search(r'"caption"\s*:\s*"([^"]*?)(?=\s*[}\]]|$)', json_str, re.DOTALL)
            if match_incomplete:
                caption = match_incomplete.group(1).replace('\n', ' ').replace('\t', ' ').strip()
                if caption:
                    case_2 += 1
                    return caption

            print(f"Could not parse: {json_str[:80]}")
            return "error"

    print(f"No JSON found in: {content[:80]}")
    return "error"


def collect_errors(jsonl_file: str, output: str) -> List[str]:
    """Write the image IDs of all failed samples to a text file, one per line."""
    df = pd.read_json(jsonl_file, lines=True)

    # A sample failed if the pipeline recorded an exception, or if the caption
    # itself looks like an error message.
    failed = df["generated_caption"].apply(is_error_caption)
    if "error" in df.columns:
        failed |= df["error"].notna()

    ids = df.loc[failed, "image_id"].tolist()
    print(f"Found {len(ids)} failed responses out of {len(df)}.")

    with open(output, "w", encoding="utf-8") as f:
        for image_id in ids:
            f.write(f"{image_id}\n")
    print(f"Wrote error IDs to {output}")
    print(f"Retry them with: python run_pipeline.py --config <config> --retry-ids {output}")
    return ids


def parse_responses(jsonl_file: str, output: str) -> None:
    """Repair model output in a JSONL results file and write a submission CSV."""
    df = pd.read_json(jsonl_file, lines=True)

    # Only rows containing JSON need repairing; plain captions pass through.
    mask = df["generated_caption"].astype(str).str.contains(r"\{", regex=True)
    df.loc[mask, "generated_caption"] = df.loc[mask, "generated_caption"].apply(parse_json_response)

    print(f"After parsing, case_1: {case_1}, case_2: {case_2}")

    still_errors = int(df["generated_caption"].apply(lambda x: x == "error").sum())
    print(f"Successfully parsed: {len(df) - still_errors}, Still errors: {still_errors}")

    submission_df = df[["image_id", "generated_caption"]].rename(
        columns={"generated_caption": "Caption", "image_id": "ID"}
    )
    submission_df.to_csv(output, index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {output}")


def merge_submissions(csv_paths: List[str], output: str) -> None:
    """Merge several submission CSVs, keeping the first non-error caption per ID."""
    dfs = [pd.read_csv(path) for path in csv_paths]
    dfs = [df[df["Caption"] != "error"] for df in dfs]

    id_sets = [set(df["ID"]) for df in dfs]
    print(f"IDs in each file: {[len(s) for s in id_sets]}")
    if len(id_sets) > 1:
        common_ids = id_sets[0].intersection(*id_sets[1:])
        print(f"Intersection of IDs (common to all files): {len(common_ids)}")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="ID", keep="first")[["ID", "Caption"]]

    still_errors = int(merged_df["Caption"].apply(lambda x: x == "error").sum())
    print(f"After merging, still errors: {still_errors}")
    print(merged_df.head(), merged_df.columns, merged_df.shape)

    # Sort numerically by the trailing index in the ID, so the submission is in
    # the order the organizers expect rather than lexicographic order.
    merged_df["sort_key"] = merged_df["ID"].str.extract(r'(\d+)$').astype(int)
    merged_df.sort_values(by="sort_key", inplace=True)
    merged_df.drop(columns=["sort_key"], inplace=True)

    with open(output, "w", newline="", encoding="utf-8") as f:
        f.write("ID,Caption\n")
        for _, row in merged_df.iterrows():
            f.write(f'{row["ID"]},"{row["Caption"]}"\n')
    print(f"Wrote {output} ({len(merged_df)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-process pipeline responses into an ImageCLEF submission.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical workflow:
  python process_submission.py collect-errors responses/run.jsonl -o error_image_ids.txt
  python run_pipeline.py --config config_deepinfra.yaml --retry-ids error_image_ids.txt
  python process_submission.py parse responses/run.jsonl -o processed_run.csv
  python process_submission.py merge processed_run*.csv -o merged_submission.csv
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_err = sub.add_parser("collect-errors", help="List image IDs of failed samples")
    p_err.add_argument("jsonl_file")
    p_err.add_argument("-o", "--output", default="error_image_ids.txt")

    p_parse = sub.add_parser("parse", help="Repair model output into a submission CSV")
    p_parse.add_argument("jsonl_file")
    p_parse.add_argument("-o", "--output", required=True)

    p_merge = sub.add_parser("merge", help="Merge submission CSVs into one")
    p_merge.add_argument("csv_paths", nargs="+")
    p_merge.add_argument("-o", "--output", default="merged_submission.csv")

    args = parser.parse_args()

    if args.command == "collect-errors":
        collect_errors(args.jsonl_file, args.output)
    elif args.command == "parse":
        parse_responses(args.jsonl_file, args.output)
    elif args.command == "merge":
        merge_submissions(args.csv_paths, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
