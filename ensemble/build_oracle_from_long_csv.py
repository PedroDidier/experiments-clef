from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import re
import json

def clean_caption(text):
    if text is None:
        return ""

    text = str(text).strip()

    # remover bloco ```json
    if text.startswith("```"):
        text = re.sub(r"^```.*?\n", "", text)
        text = text.replace("```", "").strip()

    # tentar parsear JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "caption" in obj:
            return obj["caption"]
    except:
        pass

    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monta oracle por imagem escolhendo o melhor modelo com base em uma métrica."
    )
    parser.add_argument("--input_csv", required=True, help="CSV longo gerado anteriormente")
    parser.add_argument(
        "--metric",
        required=True,
        choices=[
            "bleu1", "bleu2", "bleu3", "bleu4",
            "rouge1", "rouge2", "rougeL",
            "meteor",
            "bertscore_precision", "bertscore_recall", "bertscore_f1",
        ],
        help="Métrica usada para escolher o melhor modelo por imagem"
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="CSV de saída com a caption oracle"
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    if args.metric not in df.columns:
        raise ValueError(f"Métrica {args.metric} não encontrada no CSV.")

    # pega o índice da melhor linha por image_id
    idx = df.groupby("image_id")[args.metric].idxmax()
    oracle_df = df.loc[idx].copy().reset_index(drop=True)

    # ordenar por image_id
    oracle_df = oracle_df.sort_values("image_id").reset_index(drop=True)

    # renomear a caption escolhida
    oracle_df = oracle_df.rename(columns={
        "generated_caption": "oracle_generated_caption",
        "model_key": "oracle_selected_model",
    })

    cols_first = [
        "image_id",
        "ground_truth_caption",
        "oracle_generated_caption",
        "oracle_selected_model",
        args.metric,
    ]
    other_cols = [c for c in oracle_df.columns if c not in cols_first]
    oracle_df = oracle_df[cols_first + other_cols]

    if args.output_csv is None:
        args.output_csv = f"oracle_by_{args.metric}.csv"

    output_csv = Path(args.output_csv)
    oracle_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[OK] Oracle salvo em: {output_csv}")
    print(f"[OK] Total de imagens: {len(oracle_df)}")


if __name__ == "__main__":
    main()