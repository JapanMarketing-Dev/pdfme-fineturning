"""
Hugging Face Inference Endpoints - Custom Handler
PDFフォームから担当者が記入するフィールドを検出
"""

import os
import json
import base64
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# 4bit量子化のサポート確認
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes
    HAS_BNBITS = True
except ImportError:
    HAS_BNBITS = False
    BitsAndBytesConfig = None

SYSTEM_PROMPT = """あなたは日本の書類を分析するエキスパートです。
担当者（申請者）が記入する欄を検出してください。
職員が記入する欄は除外してください。"""

USER_PROMPT = """この画像から、担当者が記入する入力フィールドの位置をすべて検出してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


class EndpointHandler:
    def __init__(self, path: str = ""):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        use_lora = os.environ.get("USE_LORA", "true").lower() == "true"
        use_4bit = os.environ.get("USE_4BIT", "true").lower() == "true" and HAS_BNBITS
        
        print(f"Loading: {base_model}, LoRA: {use_lora}, 4bit: {use_4bit}")
        
        # プロセッサ
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        
        # モデル
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # LoRAアダプター
        if use_lora and path:
            print(f"Loading LoRA: {path}")
            self.model = PeftModel.from_pretrained(self.model, path)
        
        self.model.eval()
        print("Model loaded!")
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        inputs = data.get("inputs", data.get("image", ""))
        parameters = data.get("parameters", {})
        
        # 画像読み込み
        image = self._load_image(inputs)
        if image is None:
            return {"error": "Invalid image"}
        
        prompt = parameters.get("prompt", USER_PROMPT)
        max_tokens = parameters.get("max_new_tokens", 2048)
        
        # 推論
        return self._predict(image, prompt, max_tokens)
    
    def _load_image(self, img_input: str) -> Image.Image | None:
        try:
            if img_input.startswith("data:image"):
                _, data = img_input.split(",", 1)
                img_data = base64.b64decode(data)
            elif img_input.startswith("http"):
                import requests
                img_data = requests.get(img_input, timeout=30).content
            else:
                img_data = base64.b64decode(img_input)
            return Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            print(f"Image load error: {e}")
            return None
    
    def _predict(self, image: Image.Image, prompt: str, max_tokens: int) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
            ]}
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        
        result = {
            "raw_output": response,
            "image_size": {"width": image.size[0], "height": image.size[1]}
        }
        
        # JSONパース
        try:
            parsed = json.loads(response)
            result["parsed"] = parsed
            # ピクセル座標変換
            if "applicant_fields" in parsed:
                result["pixel_bboxes"] = [
                    self._to_pixels(f["bbox"], image.size)
                    for f in parsed["applicant_fields"]
                ]
        except json.JSONDecodeError:
            result["parsed"] = None
        
        return result
    
    def _to_pixels(self, bbox: list, size: tuple) -> list:
        w, h = size
        x1, y1, x2, y2 = bbox
        return [int(x1/1000*w), int(y1/1000*h), int(x2/1000*w), int(y2/1000*h)]
