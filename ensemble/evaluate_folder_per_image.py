#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HAS_NLTK = True
HAS_ROUGE = True
HAS_BERTSCORE = True

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except Exception:
    HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
except Exception:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score
except Exception:
    HAS_BERTSCORE = False


# =========================
# Utils
# =========================

def ensure_nltk_resources() -> None:
    if not HAS_NLTK:
        return
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Erro no arquivo {path.name}, linha {line_num}: {e}") from e
    return rows


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if HAS_NLTK:
        return word_tokenize(text.lower())
    return text.lower().split()


def sanitize_model_name(text: str) -> str:
    """
    Gera um nome seguro para usar em colunas.
    Ex.: "gemini-2.5-pro__rag3__base" -> "gemini_2_5_pro__rag3__base"
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_model_key(row: Dict[str, Any], fallback_name: str) -> str:
    """
    Monta identificador do modelo usando:
    model_name + num_rag_examples + prompt_prefix
    """
    model_name = normalize_text(row.get("model_name")) or fallback_name
    num_rag = row.get("num_rag_examples", None)
    prompt_prefix = normalize_text(row.get("prompt_prefix"))

    parts = [sanitize_model_name(model_name)]

    if num_rag is not None:
        parts.append(f"rag{num_rag}")

    if prompt_prefix:
        parts.append(sanitize_model_name(prompt_prefix))

    return "__".join(parts)


# =========================
# Métricas por exemplo
# =========================

def compute_bleu(reference: str, candidate: str) -> Dict[str, float]:
    if not HAS_NLTK:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) == 0:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    smoothing = SmoothingFunction().method1

    return {
        "bleu1": float(sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)),
        "bleu2": float(sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)),
        "bleu3": float(sentence_bleu([ref_tokens], cand_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)),
        "bleu4": float(sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)),
    }


def compute_rouge(reference: str, candidate: str, scorer: Any) -> Dict[str, float]:
    if not HAS_ROUGE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scores = scorer.score(reference, candidate)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def compute_meteor(reference: str, candidate: str) -> float:
    if not HAS_NLTK:
        return 0.0

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) == 0:
        return 0.0

    return float(meteor_score([ref_tokens], cand_tokens))


def compute_bertscore_per_example(
    references: List[str],
    candidates: List[str],
    batch_size: int = 32,
) -> List[Dict[str, float]]:
    """
    Calcula BERTScore por exemplo.
    Retorna lista de dicts com precision, recall e f1 por linha.
    """
    if not HAS_BERTSCORE:
        return [
            {
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
            }
            for _ in range(len(references))
        ]

    all_p, all_r, all_f1 = [], [], []

    for start in range(0, len(references), batch_size):
        end = start + batch_size
        refs_batch = references[start:end]
        cands_batch = candidates[start:end]

        P, R, F1 = bert_score(
            cands_batch,
            refs_batch,
            lang="en",
            verbose=False,
            device="cpu",
        )

        all_p.extend(P.tolist())
        all_r.extend(R.tolist())
        all_f1.extend(F1.tolist())

    results = []
    for p, r, f1 in zip(all_p, all_r, all_f1):
        results.append(
            {
                "bertscore_precision": float(p),
                "bertscore_recall": float(r),
                "bertscore_f1": float(f1),
            }
        )
    return results


# =========================
# Pipeline principal
# =========================

def evaluate_one_file(path: Path) -> pd.DataFrame:
    rows = load_jsonl(path)
    if not rows:
        return pd.DataFrame()

    rouge_scorer_obj = None
    if HAS_ROUGE:
        rouge_scorer_obj = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False
        )

    valid_examples: List[Dict[str, Any]] = []

    fallback_name = sanitize_model_name(path.stem)

    for idx, row in enumerate(rows):
        image_id = normalize_text(row.get("image_id"))
        ref = normalize_text(row.get("ground_truth_caption"))
        pred = normalize_text(row.get("generated_caption"))

        if not image_id or not ref or not pred:
            continue

        model_key = build_model_key(row, fallback_name)

        example = {
            "source_file": path.name,
            "image_id": image_id,
            "ground_truth_caption": ref,
            "generated_caption": pred,
            "model_key": model_key,
            "model_name_raw": normalize_text(row.get("model_name")),
            "num_rag_examples": row.get("num_rag_examples"),
            "prompt_prefix": normalize_text(row.get("prompt_prefix")),
        }

        example.update(compute_bleu(ref, pred))
        example.update(compute_rouge(ref, pred, rouge_scorer_obj) if rouge_scorer_obj else {
            "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0
        })
        example["meteor"] = compute_meteor(ref, pred)

        valid_examples.append(example)

    if not valid_examples:
        return pd.DataFrame()

    refs = [x["ground_truth_caption"] for x in valid_examples]
    preds = [x["generated_caption"] for x in valid_examples]
    bertscores = compute_bertscore_per_example(refs, preds)

    for ex, bs in zip(valid_examples, bertscores):
        ex.update(bs)

    df = pd.DataFrame(valid_examples)
    return df


def build_wide_table(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Saída:
    1 linha por image_id
    colunas do tipo model__metric
    """
    metric_cols = [
        "bleu1", "bleu2", "bleu3", "bleu4",
        "rouge1", "rouge2", "rougeL",
        "meteor",
        "bertscore_precision", "bertscore_recall", "bertscore_f1"
    ]

    keep_base = ["image_id", "ground_truth_caption"]

    # garantir uma legenda da caption GT por image_id
    base_df = (
        df_long[keep_base]
        .drop_duplicates(subset=["image_id"])
        .copy()
    )

    pivot_frames = []
    for metric in metric_cols:
        pivot = df_long.pivot_table(
            index="image_id",
            columns="model_key",
            values=metric,
            aggfunc="first"
        )
        pivot.columns = [f"{col}__{metric}" for col in pivot.columns]
        pivot_frames.append(pivot)

    wide_metrics = pd.concat(pivot_frames, axis=1).reset_index()

    df_wide = base_df.merge(wide_metrics, on="image_id", how="left")

    ordered_cols = ["image_id", "ground_truth_caption"] + sorted(
        [c for c in df_wide.columns if c not in {"image_id", "ground_truth_caption"}]
    )
    df_wide = df_wide[ordered_cols]
    return df_wide


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia uma pasta de arquivos JSONL e gera métricas por imagem por modelo."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Pasta contendo arquivos .jsonl"
    )
    parser.add_argument(
        "--output_long",
        default="per_image_metrics_long.csv",
        help="CSV longo: 1 linha por imagem+modelo"
    )
    parser.add_argument(
        "--output_wide",
        default="per_image_metrics_wide.csv",
        help="CSV wide: 1 linha por imagem, colunas por modelo+métrica"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")

    ensure_nltk_resources()

    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .jsonl encontrado em: {input_dir}")

    all_dfs = []
    for path in files:
        print(f"[INFO] Processando: {path.name}")
        df_file = evaluate_one_file(path)
        if not df_file.empty:
            all_dfs.append(df_file)

    if not all_dfs:
        raise ValueError("Nenhuma amostra válida encontrada nos arquivos.")

    df_long = pd.concat(all_dfs, ignore_index=True)

    # remove duplicata caso o mesmo image_id/model_key apareça repetido
    df_long = df_long.drop_duplicates(subset=["image_id", "model_key"], keep="first").copy()

    df_wide = build_wide_table(df_long)

    output_long = Path(args.output_long)
    output_wide = Path(args.output_wide)

    df_long.to_csv(output_long, index=False, encoding="utf-8")
    df_wide.to_csv(output_wide, index=False, encoding="utf-8")

    print(f"\n[OK] CSV longo salvo em: {output_long}")
    print(f"[OK] CSV wide salvo em : {output_wide}")
    print(f"[OK] Total de linhas no longo: {len(df_long)}")
    print(f"[OK] Total de imagens no wide: {len(df_wide)}")


if __name__ == "__main__":
    main()