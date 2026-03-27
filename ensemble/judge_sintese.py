from __future__ import annotations
import argparse
import base64
import io
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Ajuste se necessário para o seu projeto
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset import ROCOv2DataHandler
from src.analysis.evaluation_visualizer import EvaluationVisualizer

load_dotenv()

FENCED_BLOCK_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


def safe_strip(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def try_extract_caption_from_json_text(text: str) -> Optional[str]:
    obj = extract_first_json_object(text)
    if not obj:
        return None

    caption = obj.get("caption")
    if isinstance(caption, str) and caption.strip():
        return normalize_whitespace(caption)

    return None

def clean_generated_caption(text: Any) -> str:
    raw = safe_strip(text)
    if not raw:
        return ""

    extracted = try_extract_caption_from_json_text(raw)
    if extracted:
        return extracted

    fenced_match = FENCED_BLOCK_RE.match(raw)
    if fenced_match:
        raw = fenced_match.group(1).strip()

    return normalize_whitespace(raw)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_num}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_to_base64(image_input: Any) -> str:
    if isinstance(image_input, Image.Image):
        max_size = 1024
        if max(image_input.size) > max_size:
            ratio = max_size / max(image_input.size)
            new_size = (int(image_input.size[0] * ratio), int(image_input.size[1] * ratio))
            image_input = image_input.resize(new_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        if image_input.mode in ("RGBA", "LA", "P"):
            image_input = image_input.convert("RGB")
        image_input.save(buffer, format="JPEG", quality=85, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    if isinstance(image_input, str):
        path = Path(image_input)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        with open(image_input, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def build_model(provider: str, model_name: str, temperature: float, max_tokens: int):
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if provider == "together":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def extract_token_usage(response, provider: str) -> Dict[str, Any]:
    if provider == "anthropic":
        usage = response.response_metadata.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": 0.0,
        }

    token_usage_data = response.response_metadata.get("token_usage", {})
    return {
        "input_tokens": token_usage_data.get("prompt_tokens", 0),
        "output_tokens": token_usage_data.get("completion_tokens", 0),
        "total_tokens": token_usage_data.get("total_tokens", 0),
        "cost_usd": 0.0,
    }


def calculate_cost(provider: str, model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "openai": {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-5-mini": (0.25, 2.00),
        },
        "together": {
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
            "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
            "Qwen/Qwen3-VL-8B-Instruct": (0.18, 0.68),
            "Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
        },
        "anthropic": {
            "claude-sonnet-4-5-20250929": (3.00, 15.00),
        },
        "google": {},
    }

    input_rate, output_rate = pricing.get(provider, {}).get(model_name, (0.0, 0.0))
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return input_cost + output_cost


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()
    
def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = safe_strip(text)
    if not text:
        return None

    fenced_match = FENCED_BLOCK_RE.match(text)
    if fenced_match:
        text = fenced_match.group(1).strip()

    # tenta json completo
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # tenta encontrar um bloco {...}
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None

def parse_response_to_caption(response_text: str) -> str:
    extracted = try_extract_caption_from_json_text(response_text)
    if extracted:
        return extracted

    # Rescue: tenta encontrar "caption": "..."
    match = re.search(r'"caption"\s*:\s*"([^"]+)"', response_text, flags=re.DOTALL)
    if match:
        return normalize_whitespace(match.group(1))

    raise ValueError("Model did not return valid JSON with a 'caption' field.")


class JudgeSynthesizer:
    def __init__(
        self,
        provider: str,
        model_name: str,
        prompt_path: Path,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = load_prompt(prompt_path)
        self.llm = build_model(provider, model_name, temperature, max_tokens)

    def synthesize(
        self,
        image_input: Any,
        caption_a: str,
        caption_b: str,
        caption_c: str,
    ) -> Tuple[str, Dict[str, Any], str]:
        image_data_url = image_to_base64(image_input)

        prompt = (
            self.prompt_template
            .replace("{caption_a}", caption_a)
            .replace("{caption_b}", caption_b)
            .replace("{caption_c}", caption_c)
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )

        response = self.llm.invoke([message])
        response_text = response.content if isinstance(response.content, str) else str(response.content)

        token_usage = extract_token_usage(response, self.provider)
        token_usage["cost_usd"] = calculate_cost(
            self.provider,
            self.model_name,
            token_usage["input_tokens"],
            token_usage["output_tokens"],
        )

        caption = parse_response_to_caption(response_text)
        return caption, token_usage, response_text


def index_by_image_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in rows:
        image_id = safe_strip(row.get("image_id"))
        if image_id:
            out[image_id] = row
    return out


def build_merged_triplets(
    gpt_rows: List[Dict[str, Any]],
    gemini_rows: List[Dict[str, Any]],
    llama_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    gpt = index_by_image_id(gpt_rows)
    gemini = index_by_image_id(gemini_rows)
    llama = index_by_image_id(llama_rows)

    common_ids = sorted(set(gpt) & set(gemini) & set(llama))
    merged: List[Dict[str, Any]] = []

    for image_id in common_ids:
        g = gpt[image_id]
        ge = gemini[image_id]
        l = llama[image_id]

        gt_candidates = [
            safe_strip(g.get("ground_truth_caption")),
            safe_strip(ge.get("ground_truth_caption")),
            safe_strip(l.get("ground_truth_caption")),
        ]
        ground_truth = next((x for x in gt_candidates if x), "")

        merged.append(
            {
                "image_id": image_id,
                "ground_truth_caption": ground_truth,
                "candidate_captions": {
                    "gpt4o_rag": clean_generated_caption(g.get("generated_caption", "")),
                    "gemini_rag": clean_generated_caption(ge.get("generated_caption", "")),
                    "llama_rag": clean_generated_caption(l.get("generated_caption", "")),
                },
            }
        )

    return merged


def sample_rows(
    merged_rows: List[Dict[str, Any]],
    sample_size: Optional[int],
    random_seed: int,
    sample_ids_path: Optional[Path] = None,
    load_sample_ids_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if load_sample_ids_path is not None:
        with load_sample_ids_path.open("r", encoding="utf-8") as f:
            selected_ids = {line.strip() for line in f if line.strip()}
        rows = [row for row in merged_rows if row["image_id"] in selected_ids]
        return sorted(rows, key=lambda x: x["image_id"])

    if sample_size is None or sample_size >= len(merged_rows):
        selected_rows = merged_rows
    else:
        rng = random.Random(random_seed)
        selected_rows = rng.sample(merged_rows, sample_size)

    selected_rows = sorted(selected_rows, key=lambda x: x["image_id"])

    if sample_ids_path is not None:
        sample_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_ids_path.open("w", encoding="utf-8") as f:
            for row in selected_rows:
                f.write(f"{row['image_id']}\n")

    return selected_rows


def get_test_samples_by_image_ids(handler: ROCOv2DataHandler, image_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if handler.test_data is None:
        raise ValueError("Dataset not loaded. Call load_dataset() first.")

    ids_in_split = list(handler.test_data["image_id"])
    id_to_idx = {img_id: i for i, img_id in enumerate(ids_in_split)}

    samples = {}
    missing = 0

    for img_id in image_ids:
        idx = id_to_idx.get(img_id)
        if idx is None:
            missing += 1
            continue
        sample = handler.test_data[idx]
        samples[img_id] = {
            "image": sample["image"],
            "caption": sample["caption"],
            "image_id": sample["image_id"],
        }

    if missing:
        print(f"Warning: {missing} image_id(s) not found in test split.")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one LLM judge (synthesis) over 3 RAG JSONLs and evaluate automatically."
    )
    parser.add_argument("--gpt-jsonl", type=str, required=True)
    parser.add_argument("--gemini-jsonl", type=str, required=True)
    parser.add_argument("--llama-jsonl", type=str, required=True)
    parser.add_argument("--prompt-path", type=str, default="prompts/judge_synthesize_medical_caption.txt")
    parser.add_argument("--provider", type=str, required=True, choices=["openai", "google", "together", "anthropic"])
    parser.add_argument("--judge-model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)

    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-ids-path", type=str, default=None)
    parser.add_argument("--load-sample-ids-path", type=str, default=None)

    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--save-merged-input", type=str, default=None)
    parser.add_argument("--evaluation-output-dir", type=str, required=True)

    args = parser.parse_args()

    gpt_rows = load_jsonl(Path(args.gpt_jsonl))
    gemini_rows = load_jsonl(Path(args.gemini_jsonl))
    llama_rows = load_jsonl(Path(args.llama_jsonl))

    merged_rows = build_merged_triplets(gpt_rows, gemini_rows, llama_rows)

    merged_rows = sample_rows(
        merged_rows=merged_rows,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        sample_ids_path=Path(args.sample_ids_path) if args.sample_ids_path else None,
        load_sample_ids_path=Path(args.load_sample_ids_path) if args.load_sample_ids_path else None,
    )

    if args.save_merged_input:
        write_jsonl(Path(args.save_merged_input), merged_rows)

    image_ids = [row["image_id"] for row in merged_rows]

    print("=" * 80)
    print("LOADING ROCOv2 TEST SPLIT FROM HUGGINGFACE")
    print("=" * 80)
    handler = ROCOv2DataHandler()
    handler.load_dataset()
    test_samples_by_id = get_test_samples_by_image_ids(handler, image_ids)

    synthesizer = JudgeSynthesizer(
        provider=args.provider,
        model_name=args.judge_model,
        prompt_path=Path(args.prompt_path),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    results: List[Dict[str, Any]] = []
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # evita append sobre arquivo existente
    if output_path.exists():
        output_path.unlink()

    total_cost = 0.0
    total = len(merged_rows)

    print("=" * 80)
    print("RUNNING JUDGE SYNTHESIZE RAG3")
    print("=" * 80)
    print(f"Samples to process: {total}")
    print(f"Judge model: {args.provider}/{args.judge_model}")
    print(f"Output JSONL: {output_path}")
    print(f"Evaluation output dir: {args.evaluation_output_dir}")

    for idx, row in enumerate(merged_rows, 1):
        image_id = row["image_id"]
        gt = row["ground_truth_caption"]
        sample = test_samples_by_id.get(image_id)

        print(f"\n[{idx}/{total}] {image_id}")

        if sample is None:
            error_result = {
                "image_id": image_id,
                "ground_truth_caption": gt,
                "generated_caption": "",
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
                "rag_examples": [],
                "timestamp": datetime.now().isoformat(),
                "error": f"Image not found in ROCOv2 test split for image_id={image_id}",
                "judge_metadata": {
                    "judge_mode": "synthesize_rag3",
                    "judge_provider": args.provider,
                    "judge_model": args.judge_model,
                    "candidate_captions": row["candidate_captions"],
                },
            }
            results.append(error_result)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
            continue

        try:
            final_caption, token_usage, raw_response = synthesizer.synthesize(
                image_input=sample["image"],
                caption_a=row["candidate_captions"]["gpt4o_rag"],
                caption_b=row["candidate_captions"]["gemini_rag"],
                caption_c=row["candidate_captions"]["llama_rag"],
            )

            total_cost += token_usage.get("cost_usd", 0.0)

            result = {
                "image_id": image_id,
                "ground_truth_caption": gt,
                "generated_caption": final_caption,
                "token_usage": token_usage,
                "rag_examples": [],
                "timestamp": datetime.now().isoformat(),
                "judge_metadata": {
                    "judge_mode": "synthesize_rag3",
                    "judge_provider": args.provider,
                    "judge_model": args.judge_model,
                    "candidate_captions": row["candidate_captions"],
                    "raw_judge_response": raw_response,
                },
            }

            results.append(result)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"Caption: {final_caption[:140]}...")
            print(f"Tokens: {token_usage}")
            print(f"Cost: ${token_usage.get('cost_usd', 0.0):.6f}")

        except Exception as e:
            error_result = {
                "image_id": image_id,
                "ground_truth_caption": gt,
                "generated_caption": "",
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
                "rag_examples": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "judge_metadata": {
                    "judge_mode": "synthesize_rag3",
                    "judge_provider": args.provider,
                    "judge_model": args.judge_model,
                    "candidate_captions": row["candidate_captions"],
                },
            }
            results.append(error_result)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
            print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("JUDGE SYNTHESIS COMPLETE")
    print("=" * 80)
    print(f"Processed: {len(results)}")
    print(f"Successful: {len([r for r in results if 'error' not in r])}")
    print(f"Failed: {len([r for r in results if 'error' in r])}")
    print(f"Total cost: ${total_cost:.6f}")
    if results:
        print(f"Average cost/sample: ${total_cost / len(results):.6f}")
    print(f"Saved JSONL to: {output_path}")

    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    evaluator = EvaluationVisualizer()
    evaluator.run_full_evaluation(str(output_path), args.evaluation_output_dir)

    print(f"Evaluation completed. Results saved to: {args.evaluation_output_dir}")


if __name__ == "__main__":
    main()