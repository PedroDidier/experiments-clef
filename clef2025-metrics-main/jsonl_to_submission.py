#!/usr/bin/env python3
import json
import csv
import re
import argparse
from pathlib import Path

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_caption(generated_caption_field):
    """
    Seu JSONL às vezes tem generated_caption como:
      ```json
      {"caption": "..."}
      ```
    ou pode vir como string simples.
    """
    if generated_caption_field is None:
        return ""

    # Se for dict já (raro), tenta pegar diretamente
    if isinstance(generated_caption_field, dict):
        return str(generated_caption_field.get("caption", "")).strip()

    text = str(generated_caption_field).strip()

    # 1) Tenta extrair conteúdo dentro de ```...```
    m = FENCE_RE.search(text)
    if m:
        inner = m.group(1).strip()
        # tenta parsear como JSON
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "caption" in obj:
                return str(obj["caption"]).strip()
            # se não for dict esperado, devolve inner como fallback
            return inner.strip()
        except Exception:
            # se não parsear, usa o inner bruto
            return inner.strip()

    # 2) Se não tem fence, tenta parsear a string inteira como JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "caption" in obj:
            return str(obj["caption"]).strip()
    except Exception:
        pass

    # 3) Fallback final: devolve a string original
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Caminho do arquivo .jsonl")
    ap.add_argument("--out", required=True, help="Caminho do submission.csv de saída")
    ap.add_argument("--id-field", default="image_id", help="Campo do ID no jsonl (default: image_id)")
    ap.add_argument("--caption-field", default="generated_caption", help="Campo da caption gerada (default: generated_caption)")
    args = ap.parse_args()

    in_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["ID", "Caption"])

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                skipped += 1
                continue

            img_id = row.get(args.id_field, "")
            cap_raw = row.get(args.caption_field, "")

            if not img_id:
                skipped += 1
                continue

            caption = extract_caption(cap_raw)

            # opcional: remove quebras excessivas
            caption = " ".join(caption.split()).strip()

            w.writerow([img_id, caption])
            written += 1

    print(f"[OK] Lidas: {total} | Escritas: {written} | Puladas: {skipped}")
    print(f"[OK] Saída: {out_path}")

if __name__ == "__main__":
    main()
