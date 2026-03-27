#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def classify_candidate(name: str) -> str:
    low = name.lower()
    if "no_rag" in low or "sem_rag" in low or "without_rag" in low:
        return "no_rag"
    if "rag" in low:
        return "rag"
    return "unknown"


def load_inputs(base_dir: Path):
    individual_path = base_dir / "individual_candidate_metrics.csv"
    oracle_comp_path = base_dir / "oracle_vs_best_individual.csv"
    oracles_dir = base_dir / "oracles"

    df_individual = pd.read_csv(individual_path)
    df_oracle_comp = pd.read_csv(oracle_comp_path)

    winner_files = {
        "oracle_all6_bleu": oracles_dir / "oracle_all6_bleu_winner_stats.csv",
        "oracle_all6_rougeL": oracles_dir / "oracle_all6_rougeL_winner_stats.csv",
        "oracle_rag_only_bleu": oracles_dir / "oracle_rag_only_bleu_winner_stats.csv",
        "oracle_rag_only_rougeL": oracles_dir / "oracle_rag_only_rougeL_winner_stats.csv",
        "oracle_no_rag_only_bleu": oracles_dir / "oracle_no_rag_only_bleu_winner_stats.csv",
        "oracle_no_rag_only_rougeL": oracles_dir / "oracle_no_rag_only_rougeL_winner_stats.csv",
    }

    winner_dfs = {}
    for key, path in winner_files.items():
        if path.exists():
            winner_dfs[key] = pd.read_csv(path)

    return df_individual, df_oracle_comp, winner_dfs


def add_group_columns(df_individual: pd.DataFrame) -> pd.DataFrame:
    df = df_individual.copy()
    df["group"] = df["candidate"].apply(classify_candidate)
    return df


def plot_individual_models(df_individual: pd.DataFrame, outdir: Path) -> None:
    df = df_individual.sort_values("rougeL_mean", ascending=False).copy()
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, df["rougeL_mean"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["candidate"], rotation=30, ha="right")
    ax.set_ylabel("ROUGE-L médio")
    ax.set_title("Modelos individuais - ROUGE-L")
    for i, v in enumerate(df["rougeL_mean"]):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "01_individual_models_rougeL.png", dpi=200)
    plt.close(fig)

    df_bleu = df_individual.sort_values("bleu_mean", ascending=False).copy()
    x = range(len(df_bleu))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, df_bleu["bleu_mean"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_bleu["candidate"], rotation=30, ha="right")
    ax.set_ylabel("BLEU médio")
    ax.set_title("Modelos individuais - BLEU")
    for i, v in enumerate(df_bleu["bleu_mean"]):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "02_individual_models_bleu.png", dpi=200)
    plt.close(fig)


def plot_rag_vs_no_rag_summary(df_individual: pd.DataFrame, outdir: Path) -> None:
    df = add_group_columns(df_individual)
    grouped = df.groupby("group")[["bleu_mean", "rougeL_mean", "rouge1_mean", "rouge2_mean"]].mean().reset_index()

    metrics = ["bleu_mean", "rouge1_mean", "rouge2_mean", "rougeL_mean"]
    labels = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35

    rag_vals = grouped[grouped["group"] == "rag"][metrics].iloc[0].tolist() if (grouped["group"] == "rag").any() else [0]*len(metrics)
    no_rag_vals = grouped[grouped["group"] == "no_rag"][metrics].iloc[0].tolist() if (grouped["group"] == "no_rag").any() else [0]*len(metrics)

    ax.bar([i - width/2 for i in x], rag_vals, width=width, label="RAG")
    ax.bar([i + width/2 for i in x], no_rag_vals, width=width, label="Sem RAG")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Média")
    ax.set_title("Média dos modelos com RAG vs sem RAG")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "03_rag_vs_no_rag_summary.png", dpi=200)
    plt.close(fig)


def plot_oracle_vs_best(df_oracle_comp: pd.DataFrame, outdir: Path) -> None:
    # Foco nos cenários principais
    keep = df_oracle_comp[df_oracle_comp["oracle_name"].isin([
        "oracle_all6_bleu",
        "oracle_all6_rougeL",
        "oracle_rag_only_bleu",
        "oracle_rag_only_rougeL",
        "oracle_no_rag_only_bleu",
        "oracle_no_rag_only_rougeL",
    ])].copy()

    # BLEU
    bleu_rows = keep[keep["oracle_metric"] == "bleu"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(bleu_rows))
    ax.bar(x, bleu_rows["oracle_bleu_mean"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(bleu_rows["oracle_name"], rotation=25, ha="right")
    ax.set_ylabel("BLEU médio")
    ax.set_title("Oracle - BLEU médio")
    for i, v in enumerate(bleu_rows["oracle_bleu_mean"]):
        gain = bleu_rows.iloc[i]["gain_vs_best_bleu"]
        ax.text(i, v + 0.001, f"{v:.3f}\n(+{gain:.3f})", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "04_oracle_bleu.png", dpi=200)
    plt.close(fig)

    # ROUGE-L
    rl_rows = keep[keep["oracle_metric"] == "rougeL"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(rl_rows)), rl_rows["oracle_rougeL_mean"])
    ax.set_xticks(list(range(len(rl_rows))))
    ax.set_xticklabels(rl_rows["oracle_name"], rotation=25, ha="right")
    ax.set_ylabel("ROUGE-L médio")
    ax.set_title("Oracle - ROUGE-L médio")
    for i, v in enumerate(rl_rows["oracle_rougeL_mean"]):
        gain = rl_rows.iloc[i]["gain_vs_best_rougeL"]
        ax.text(i, v + 0.002, f"{v:.3f}\n(+{gain:.3f})", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "05_oracle_rougeL.png", dpi=200)
    plt.close(fig)


def plot_best_individual_vs_oracles(df_individual: pd.DataFrame, df_oracle_comp: pd.DataFrame, outdir: Path) -> None:
    best_bleu_row = df_individual.sort_values("bleu_mean", ascending=False).iloc[0]
    best_rl_row = df_individual.sort_values("rougeL_mean", ascending=False).iloc[0]

    # BLEU
    bleu_items = [
        ("best_individual", best_bleu_row["bleu_mean"]),
    ]
    for oracle_name in ["oracle_no_rag_only_bleu", "oracle_rag_only_bleu", "oracle_all6_bleu"]:
        row = df_oracle_comp[df_oracle_comp["oracle_name"] == oracle_name]
        if not row.empty:
            bleu_items.append((oracle_name, row.iloc[0]["oracle_bleu_mean"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [x[0] for x in bleu_items]
    vals = [x[1] for x in bleu_items]
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(list(range(len(vals))))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("BLEU médio")
    ax.set_title("Melhor individual vs oracles (BLEU)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "06_best_individual_vs_oracles_bleu.png", dpi=200)
    plt.close(fig)

    # ROUGE-L
    rl_items = [
        ("best_individual", best_rl_row["rougeL_mean"]),
    ]
    for oracle_name in ["oracle_no_rag_only_rougeL", "oracle_rag_only_rougeL", "oracle_all6_rougeL"]:
        row = df_oracle_comp[df_oracle_comp["oracle_name"] == oracle_name]
        if not row.empty:
            rl_items.append((oracle_name, row.iloc[0]["oracle_rougeL_mean"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [x[0] for x in rl_items]
    vals = [x[1] for x in rl_items]
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(list(range(len(vals))))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ROUGE-L médio")
    ax.set_title("Melhor individual vs oracles (ROUGE-L)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "07_best_individual_vs_oracles_rougeL.png", dpi=200)
    plt.close(fig)


def plot_winner_distribution(winner_df: pd.DataFrame, title: str, outfile: Path) -> None:
    df = winner_df.sort_values("wins", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(df)), df["wins"])
    ax.set_xticks(list(range(len(df))))
    ax.set_xticklabels(df["candidate"], rotation=25, ha="right")
    ax.set_ylabel("Número de vitórias")
    ax.set_title(title)
    for i, row in df.iterrows():
        pass
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.text(idx, row["wins"] + max(df["wins"]) * 0.01, f"{row['win_rate']:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_rag_vs_no_rag_wins(winner_df: pd.DataFrame, title: str, outfile: Path) -> None:
    df = winner_df.copy()
    df["group"] = df["candidate"].apply(classify_candidate)
    grouped = df.groupby("group")["wins"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(grouped["group"], grouped["wins"])
    ax.set_ylabel("Número de vitórias")
    ax.set_title(title)
    total = grouped["wins"].sum()
    for i, row in grouped.iterrows():
        rate = row["wins"] / total if total else 0
        ax.text(i, row["wins"] + total * 0.01, f"{rate:.1%}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def write_summary_txt(df_individual: pd.DataFrame, df_oracle_comp: pd.DataFrame, winner_dfs: dict, outdir: Path) -> None:
    best_bleu = df_individual.sort_values("bleu_mean", ascending=False).iloc[0]
    best_rl = df_individual.sort_values("rougeL_mean", ascending=False).iloc[0]

    oracle_all6_bleu = df_oracle_comp[df_oracle_comp["oracle_name"] == "oracle_all6_bleu"].iloc[0]
    oracle_all6_rl = df_oracle_comp[df_oracle_comp["oracle_name"] == "oracle_all6_rougeL"].iloc[0]
    oracle_rag_rl = df_oracle_comp[df_oracle_comp["oracle_name"] == "oracle_rag_only_rougeL"].iloc[0]
    oracle_no_rag_rl = df_oracle_comp[df_oracle_comp["oracle_name"] == "oracle_no_rag_only_rougeL"].iloc[0]

    lines = []
    lines.append("RESUMO AUTOMÁTICO DOS RESULTADOS\n")
    lines.append(f"Melhor modelo individual em BLEU: {best_bleu['candidate']} ({best_bleu['bleu_mean']:.6f})")
    lines.append(f"Melhor modelo individual em ROUGE-L: {best_rl['candidate']} ({best_rl['rougeL_mean']:.6f})\n")

    lines.append("Oracle all6:")
    lines.append(f"- BLEU: {oracle_all6_bleu['oracle_bleu_mean']:.6f} | ganho vs melhor individual: {oracle_all6_bleu['gain_vs_best_bleu']:.6f}")
    lines.append(f"- ROUGE-L: {oracle_all6_rl['oracle_rougeL_mean']:.6f} | ganho vs melhor individual: {oracle_all6_rl['gain_vs_best_rougeL']:.6f}\n")

    lines.append("Comparação entre cenários ROUGE-L:")
    lines.append(f"- oracle_no_rag_only_rougeL: {oracle_no_rag_rl['oracle_rougeL_mean']:.6f}")
    lines.append(f"- oracle_rag_only_rougeL: {oracle_rag_rl['oracle_rougeL_mean']:.6f}")
    lines.append(f"- oracle_all6_rougeL: {oracle_all6_rl['oracle_rougeL_mean']:.6f}\n")

    if "oracle_all6_rougeL" in winner_dfs:
        dfw = winner_dfs["oracle_all6_rougeL"].copy()
        dfw["group"] = dfw["candidate"].apply(classify_candidate)
        rag_wins = dfw[dfw["group"] == "rag"]["wins"].sum()
        no_rag_wins = dfw[dfw["group"] == "no_rag"]["wins"].sum()
        total = dfw["wins"].sum()
        lines.append("Vitórias no oracle_all6_rougeL:")
        lines.append(f"- modelos com RAG: {rag_wins} ({rag_wins/total:.2%})")
        lines.append(f"- modelos sem RAG: {no_rag_wins} ({no_rag_wins/total:.2%})\n")

    lines.append("Interpretação sugerida:")
    lines.append("- RAG melhorou consistentemente os modelos individuais.")
    lines.append("- O oracle mostrou complementaridade real entre os candidatos.")
    lines.append("- A maior parte do ganho do ensemble vem dos modelos com RAG.")
    lines.append("- Os modelos sem RAG ainda adicionam alguma diversidade, mas bem menor.")
    lines.append("- Vale observar que gpt4o_rag tem menos amostras (9371), então isso deve ser registrado como limitação metodológica.")

    with open(outdir / "summary_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos e resumo dos resultados do oracle ensemble.")
    parser.add_argument("--base-dir", type=str, required=True, help="Pasta de saída do ensemble_oracle_v2.py")
    parser.add_argument("--output-dir", type=str, default=None, help="Pasta onde salvar os gráficos")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "plots")
    ensure_dir(output_dir)

    df_individual, df_oracle_comp, winner_dfs = load_inputs(base_dir)

    plot_individual_models(df_individual, output_dir)
    plot_rag_vs_no_rag_summary(df_individual, output_dir)
    plot_oracle_vs_best(df_oracle_comp, output_dir)
    plot_best_individual_vs_oracles(df_individual, df_oracle_comp, output_dir)

    if "oracle_all6_bleu" in winner_dfs:
        plot_winner_distribution(
            winner_dfs["oracle_all6_bleu"],
            "Vitórias por candidato - Oracle all6 (BLEU)",
            output_dir / "08_wins_all6_bleu.png"
        )
        plot_rag_vs_no_rag_wins(
            winner_dfs["oracle_all6_bleu"],
            "Vitórias RAG vs sem RAG - Oracle all6 (BLEU)",
            output_dir / "09_wins_all6_bleu_rag_vs_no_rag.png"
        )

    if "oracle_all6_rougeL" in winner_dfs:
        plot_winner_distribution(
            winner_dfs["oracle_all6_rougeL"],
            "Vitórias por candidato - Oracle all6 (ROUGE-L)",
            output_dir / "10_wins_all6_rougeL.png"
        )
        plot_rag_vs_no_rag_wins(
            winner_dfs["oracle_all6_rougeL"],
            "Vitórias RAG vs sem RAG - Oracle all6 (ROUGE-L)",
            output_dir / "11_wins_all6_rougeL_rag_vs_no_rag.png"
        )

    write_summary_txt(df_individual, df_oracle_comp, winner_dfs, output_dir)

    print(f"Gráficos salvos em: {output_dir}")


if __name__ == "__main__":
    main()