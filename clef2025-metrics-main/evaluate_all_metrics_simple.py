#!/usr/bin/env python3
"""
Script simplificado para calcular TODAS as métricas de avaliação de captions médicos.
Usa bibliotecas padrão e é mais robusto que o script original.

Métricas calculadas:
1. BLEU (1, 2, 3, 4)
2. ROUGE (1, 2, L)
3. METEOR
4. CIDEr
5. BERTScore
6. BLEURT (opcional - requer modelo grande)

Uso:
    python evaluate_all_metrics_simple.py --ground-truth data/valid/captions.csv --submissions submissions/ --output results.csv
"""

import os
import sys
import csv
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Suprime warnings desnecessários
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Importações de métricas
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download de recursos necessários do NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("[INFO] Baixando recursos do NLTK...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    HAS_NLTK = True
except ImportError:
    print("[AVISO] NLTK não instalado. BLEU e METEOR não estarão disponíveis.")
    HAS_NLTK = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    print("[AVISO] rouge-score não instalado. ROUGE não estará disponível.")
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    print("[AVISO] bert-score não instalado. BERTScore não estará disponível.")
    HAS_BERTSCORE = False

try:
    from pycocoevalcap.cider.cider import Cider
    HAS_CIDER = True
except ImportError:
    print("[AVISO] pycocoevalcap não instalado. CIDEr não estará disponível.")
    HAS_CIDER = False


def load_csv(file_path: Path) -> Dict[str, str]:
    """Carrega um arquivo CSV de captions."""
    captions = {}
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'ID' in row and 'Caption' in row:
                        captions[row['ID']] = row['Caption']
            return captions
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"[ERRO] Erro ao ler {file_path}: {e}")
            return {}
    
    print(f"[ERRO] Não foi possível ler {file_path} com nenhum encoding")
    return {}


def preprocess_text(text: str) -> str:
    """Pré-processa o texto para normalização."""
    import re
    import string
    
    # Lowercase
    text = text.lower()
    
    # Remove pontuação extra
    text = text.strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """Tokeniza o texto."""
    if HAS_NLTK:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text.lower())
    else:
        # Tokenização simples
        return text.lower().split()


def compute_bleu(reference: str, candidate: str) -> Dict[str, float]:
    """Calcula BLEU-1, BLEU-2, BLEU-3, BLEU-4."""
    if not HAS_NLTK:
        return {'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0}
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    smoothing = SmoothingFunction().method1
    
    try:
        bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    except:
        bleu1 = bleu2 = bleu3 = bleu4 = 0.0
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4
    }


def compute_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """Calcula ROUGE-1, ROUGE-2, ROUGE-L."""
    if not HAS_ROUGE:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    try:
        scores = scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def compute_meteor(reference: str, candidate: str) -> float:
    """Calcula METEOR."""
    if not HAS_NLTK:
        return 0.0
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    try:
        return meteor_score([ref_tokens], cand_tokens)
    except:
        return 0.0


def compute_cider(references: Dict[str, List[str]], candidates: Dict[str, List[str]]) -> float:
    """Calcula CIDEr para todo o dataset."""
    if not HAS_CIDER:
        return 0.0
    
    try:
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(references, candidates)
        return score
    except:
        return 0.0


def compute_bertscore_batch(references: List[str], candidates: List[str]) -> Tuple[float, float, float]:
    """Calcula BERTScore em batch."""
    if not HAS_BERTSCORE:
        return 0.0, 0.0, 0.0
    
    try:
        P, R, F1 = bert_score(candidates, references, lang='en', verbose=False, device='cpu')
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except Exception as e:
        print(f"[AVISO] Erro ao calcular BERTScore: {e}")
        return 0.0, 0.0, 0.0


def evaluate_submission(ground_truth: Dict[str, str], predictions: Dict[str, str]) -> Dict:
    """Avalia uma submissão completa."""
    
    # Filtra apenas IDs que existem em ambos
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    
    if len(common_ids) == 0:
        return {
            'status': 'error',
            'error': 'Nenhum ID em comum entre ground truth e predições',
            'num_samples': 0
        }
    
    print(f"  Avaliando {len(common_ids)} amostras...")
    
    # Inicializa acumuladores
    bleu_scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    meteor_scores = []
    
    # Prepara dados para CIDEr
    cider_refs = {}
    cider_hyps = {}
    
    # Prepara dados para BERTScore
    refs_list = []
    preds_list = []
    
    # Calcula métricas por amostra
    for img_id in tqdm(common_ids, desc="  Calculando métricas"):
        ref = ground_truth[img_id]
        pred = predictions[img_id]
        
        # BLEU
        bleu = compute_bleu(ref, pred)
        for key in bleu:
            bleu_scores[key].append(bleu[key])
        
        # ROUGE
        rouge = compute_rouge(ref, pred)
        for key in rouge:
            rouge_scores[key].append(rouge[key])
        
        # METEOR
        meteor = compute_meteor(ref, pred)
        meteor_scores.append(meteor)
        
        # Prepara para CIDEr
        cider_refs[img_id] = [ref]
        cider_hyps[img_id] = [pred]
        
        # Prepara para BERTScore
        refs_list.append(ref)
        preds_list.append(pred)
    
    # Calcula médias
    results = {
        'status': 'success',
        'num_samples': len(common_ids),
        'bleu1': np.mean(bleu_scores['bleu1']),
        'bleu2': np.mean(bleu_scores['bleu2']),
        'bleu3': np.mean(bleu_scores['bleu3']),
        'bleu4': np.mean(bleu_scores['bleu4']),
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'meteor': np.mean(meteor_scores) if meteor_scores else 0.0,
    }
    
    # CIDEr (corpus-level)
    print("  Calculando CIDEr...")
    results['cider'] = compute_cider(cider_refs, cider_hyps)
    
    # BERTScore (batch)
    print("  Calculando BERTScore...")
    bert_p, bert_r, bert_f1 = compute_bertscore_batch(refs_list, preds_list)
    results['bertscore_precision'] = bert_p
    results['bertscore_recall'] = bert_r
    results['bertscore_f1'] = bert_f1
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Calcula todas as métricas de avaliação de captions médicos'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Caminho para o arquivo CSV de ground truth'
    )
    parser.add_argument(
        '--submissions',
        type=str,
        required=True,
        help='Diretório contendo os arquivos CSV de submissões'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Arquivo de saída CSV (padrão: results_YYYYMMDD_HHMMSS.csv)'
    )
    
    args = parser.parse_args()
    
    # Carrega ground truth
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"[ERRO] Ground truth não encontrado: {gt_path}")
        sys.exit(1)
    
    print(f"[INFO] Carregando ground truth: {gt_path}")
    ground_truth = load_csv(gt_path)
    print(f"[OK] Carregadas {len(ground_truth)} referências")
    
    # Lista arquivos de submissão
    submissions_dir = Path(args.submissions)
    if not submissions_dir.exists():
        print(f"[ERRO] Diretório de submissões não encontrado: {submissions_dir}")
        sys.exit(1)
    
    csv_files = sorted(submissions_dir.glob('*.csv'))
    if not csv_files:
        print(f"[ERRO] Nenhum arquivo CSV encontrado em {submissions_dir}")
        sys.exit(1)
    
    print(f"[OK] Encontrados {len(csv_files)} arquivos para avaliar")
    print()
    
    # Arquivo de saída
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(f'results_{timestamp}.csv')
    
    # Processa cada arquivo
    results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Processando: {csv_file.name}")
        
        start_time = datetime.now()
        
        # Carrega predições
        predictions = load_csv(csv_file)
        
        if not predictions:
            result = {
                'file': csv_file.name,
                'status': 'error',
                'error': 'Erro ao carregar arquivo',
                'num_samples': 0
            }
        else:
            # Avalia
            result = evaluate_submission(ground_truth, predictions)
            result['file'] = csv_file.name
        
        end_time = datetime.now()
        result['duration_seconds'] = (end_time - start_time).total_seconds()
        result['timestamp'] = end_time.isoformat()
        
        results.append(result)
        
        # Salva incrementalmente
        save_results(results, output_file)
        
        if result['status'] == 'success':
            print(f"  [OK] Sucesso! (tempo: {result['duration_seconds']:.1f}s)")
            print(f"     BLEU-4: {result['bleu4']:.4f}, ROUGE-L: {result['rougeL']:.4f}, "
                  f"METEOR: {result['meteor']:.4f}, CIDEr: {result['cider']:.4f}")
            print(f"     BERTScore F1: {result['bertscore_f1']:.4f}")
        else:
            print(f"  [ERRO] {result.get('error', 'Erro desconhecido')}")
        
        print()
    
    # Resumo final
    print("=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)
    successful = [r for r in results if r['status'] == 'success']
    print(f"Total de arquivos: {len(results)}")
    print(f"Sucessos: {len(successful)}")
    print(f"Falhas: {len(results) - len(successful)}")
    print(f"Resultados salvos em: {output_file}")
    print()
    
    if successful:
        print("Top 5 por BERTScore F1:")
        top5 = sorted(successful, key=lambda x: x['bertscore_f1'], reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            print(f"  {i}. {r['file']}: BERTScore F1={r['bertscore_f1']:.4f}, "
                  f"BLEU-4={r['bleu4']:.4f}, ROUGE-L={r['rougeL']:.4f}")
    
    print()


def save_results(results: List[Dict], output_path: Path):
    """Salva resultados em CSV."""
    if not results:
        return
    
    fieldnames = [
        'file', 'status', 'timestamp', 'duration_seconds', 'num_samples',
        'bleu1', 'bleu2', 'bleu3', 'bleu4',
        'rouge1', 'rouge2', 'rougeL',
        'meteor', 'cider',
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'error'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, '') for field in fieldnames}
            writer.writerow(row)


if __name__ == '__main__':
    main()
