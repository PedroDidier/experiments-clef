import os
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

class MedicalImageCaptioner:
    def __init__(self, provider: str = "huggingface", model_name: str = "google/medgemma-1.5-4b-it", max_tokens: int = 512):
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        if self.provider == "huggingface" or "medgemma" in self.model_name.lower():
            print(f"Loading local model: {self.model_name} with 4-bit quantization...")
            
            # Configuração de quantização para GPUs com 24GB-40GB de VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            # Mantém suas implementações anteriores (OpenAI / Google Gemini)
            pass

    def generate_caption(self, image: Image.Image) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Geração direta sem exemplos RAG."""
        if self.provider == "huggingface":
            prompt_text = "Generate a concise, clinically accurate caption describing this medical image."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            return self._run_hf_inference(messages, [image])
        
        # Fallback para outras APIs...

    def generate_caption_with_rag(self, image: Image.Image, similar_examples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        """Geração contextualizada com exemplos similares do ROCOv2."""
        if self.provider == "huggingface":
            # Constrói o contexto RAG em texto
            rag_context = "Here are similar reference cases with their clinical captions:\n"
            for idx, ex in enumerate(similar_examples, 1):
                rag_context += f"Case {idx}: {ex.get('caption', '')}\n"
            
            prompt_text = (
                f"{rag_context}\n"
                "Based on the visual evidence and the reference cases above, generate a clinical caption for this target medical image."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            caption_data, token_usage = self._run_hf_inference(messages, [image])
            return caption_data, token_usage, similar_examples
        
        # Fallback para outras APIs...

    def _run_hf_inference(self, messages: list, images: list) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executa a geração local via PyTorch / Hugging Face."""
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False  # Geração determinística (greedy) para experimentos consistentes
            )
            
            # Decodifica apenas a resposta gerada, ignorando os tokens do prompt
            input_length = inputs["input_ids"].shape[1]
            caption = self.processor.batch_decode(
                generated_ids[:, input_length:], 
                skip_special_tokens=True
            )[0].strip()

        token_usage = {
            "input_tokens": int(input_length),
            "output_tokens": int(generated_ids.shape[1] - input_length),
            "total_tokens": int(generated_ids.shape[1]),
            "cost_usd": 0.0  # Execução local tem custo zero de API
        }

        return {"caption": caption}, token_usage
