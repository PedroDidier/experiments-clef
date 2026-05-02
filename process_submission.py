import pandas as pd
import json
import re
import csv
from typing import Dict, Any, List

case_1 = 0
case_2 = 0

def parse_json_response(content: str) -> str:
    """
    Parse JSON response, extracting the last JSON object by {} brackets.
    Returns the 'caption' value from the JSON.
    
    Args:
        content (str): The raw response content
        
    Returns:
        str: The caption string
    """
    content = content.strip().replace('\n', ' ').replace('\t', ' ')
    
    # Find the last JSON object in the content
    json_matches = re.findall(r'\{.*?\}', content, re.DOTALL)
    global case_1, case_2

    if json_matches:
        json_str = json_matches[-1]  # Take the last JSON object
        try:
            parsed = json.loads(json_str)
            case_1 += 1
            return parsed.get("caption", json_str)  # Return caption if exists, else the json_str
        except json.JSONDecodeError:
            # Fallback to regex extraction
            match = re.search(r'"caption"\s*:\s*"([^"]*)"', json_str, re.DOTALL)
            if match:
                caption = match.group(1).replace('\n', ' ').replace('\t', ' ').strip()
                case_2 += 1
                return caption
            else:
                return "error"
    else:
        return "error"

def process_errors(jsonl_file: str) -> List[str]:
    df = pd.read_json(jsonl_file, lines=True)
    df_errors = df[df["generated_caption"].str.contains("DeepInfra")]
    print(f"Found {len(df_errors)} error responses.")

    ids = df_errors["image_id"].tolist()
    
    return ids

def process_unparsed(jsonl_file: str):
    df = pd.read_json(jsonl_file, lines=True)
    df_unparsed = df[df["generated_caption"].str.contains("```json")]
    print(f"Found {len(df_unparsed)} unparsed responses.")

    # Only apply parsing to rows that contain ```json
    mask = df["generated_caption"].str.contains("{")
    df.loc[mask, "generated_caption"] = df.loc[mask, "generated_caption"].apply(parse_json_response)
    
    print(f"After parsing, case_1: {case_1}, case_2: {case_2}")

    # count how many rows were successfully parsed vs how many still contain errors
    successfully_parsed = df["generated_caption"].apply(lambda x: x != "error").sum()
    still_errors = df["generated_caption"].apply(lambda x: x == "error").sum()
    print(f"Successfully parsed: {successfully_parsed}, Still errors: {still_errors}")

    submission_df = df[["image_id", "generated_caption"]]
    submission_df.rename(columns={"generated_caption": "Caption", "image_id": "ID"}, inplace=True)
    
    # Write CSV with all fields quoted to ensure consistent formatting
    submission_df.to_csv("processed_responses_6.csv", index=False, quoting=csv.QUOTE_ALL)



def process_submission(csv_paths: List[str]):
    dfs = [pd.read_csv(path) for path in csv_paths]
    # Merge all DataFrames on ID


    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i]['Caption'] != "error"]

    # Check intersection of IDs across all dataframes
    id_sets = [set(df["ID"]) for df in dfs]
    common_ids = id_sets[0].intersection(*id_sets[1:])
    print(f"IDs in each file: {[len(s) for s in id_sets]}")
    print(f"Intersection of IDs (common to all files): {len(common_ids)}")

    # Concatenate all dataframes (stack rows)
    merged_df = pd.concat(dfs, ignore_index=True)

    # For duplicate IDs, keep the first non-empty caption
    merged_df = merged_df.drop_duplicates(subset="ID", keep="first")
    merged_df = merged_df[["ID", "Caption"]]

    
    still_errors = merged_df["Caption"].apply(lambda x: x == "error").sum()
    print(f"After merging, still errors: {still_errors}")

    print(merged_df.head(), merged_df.columns, merged_df.shape)

    # Sort by ID to ensure correct order (natural sorting)
    # Extract the numeric part from the ID for proper numerical ordering
    merged_df["sort_key"] = merged_df["ID"].str.extract(r'(\d+)$').astype(int)
    merged_df.sort_values(by="sort_key", inplace=True)
    merged_df.drop(columns=["sort_key"], inplace=True)

    # Save merged CSV with only Caption column quoted
    with open("merged_submission.csv", "w", newline="", encoding="utf-8") as f:
        f.write("ID,Caption\n")
        for _, row in merged_df.iterrows():
            f.write(f'{row["ID"]},"{row["Caption"]}"\n')

if __name__ == "__main__":
    # jsonl_path = "responses/responses_rag_hf_2026-04-30_02-19.jsonl"
    # error_ids = process_errors(jsonl_path)

    # # Save into txt
    # with open("error_image_ids.txt", "w") as f:
    #     for image_id in error_ids:
    #         f.write(f"{image_id}\n")

    # process_unparsed(jsonl_path)

    csv_paths = ["processed_responses_4.csv", "processed_responses_5.csv", "processed_responses_6.csv"]
    process_submission(csv_paths)