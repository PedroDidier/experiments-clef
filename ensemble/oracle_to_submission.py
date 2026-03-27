import pandas as pd
import argparse
from pathlib import Path

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="oracle_submission.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    submission = pd.DataFrame({
        "ID": df["image_id"],
        "Caption": df["oracle_generated_caption"]
    })

    submission.to_csv(args.output, index=False)

    print(f"[OK] Submission salva em: {args.output}")
    print(f"[OK] Total de linhas: {len(submission)}")

if __name__ == "__main__":
    main()