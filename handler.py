"""
Hugging Face Inference Endpoints - Custom Handler
PDFフォームから担当者が記入するフィールドを検出
"""

import os
import json
import base64
import re
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
        
        # 環境変数から設定を取得
        base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        # LoRAアダプターのHugging Face リポジトリID
        lora_adapter = os.environ.get("LORA_ADAPTER", "takumi123xxx/pdfme-form-field-detector-lora")
        use_4bit = os.environ.get("USE_4BIT", "true").lower() == "true" and HAS_BNBITS
        
        print(f"Loading base model: {base_model}")
        print(f"LoRA adapter: {lora_adapter}")
        print(f"4bit quantization: {use_4bit}")
        print(f"Device: {self.device}")
        
        # プロセッサ
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        
        # モデル
        if use_4bit:
            print("Loading with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            print("Loading with bfloat16...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # LoRAアダプターをHugging Faceからロード
        if lora_adapter:
            print(f"Loading LoRA adapter from HF Hub: {lora_adapter}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            inputs = data.get("inputs", data.get("image", ""))
            parameters = data.get("parameters", {})
            
            # 画像読み込み
            image = self._load_image(inputs)
            if image is None:
                return {"error": "Invalid image input"}
            
            # 画像サイズ制限（メモリ節約）
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
            
            prompt = parameters.get("prompt", USER_PROMPT)
            max_tokens = parameters.get("max_new_tokens", 1024)
            
            # 推論
            return self._predict(image, prompt, max_tokens)
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _load_image(self, img_input: str) -> Image.Image | None:
        try:
            if not img_input:
                return None
            
            if img_input.startswith("data:image"):
                _, data = img_input.split(",", 1)
                img_data = base64.b64decode(data)
            elif img_input.startswith("http"):
                import requests
                img_data = requests.get(img_input, timeout=30).content
            else:
                # Base64文字列として扱う
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
        
        # bboxのパース
        bboxes = self._parse_bboxes(response, image.size)
        result["bboxes"] = bboxes
        result["count"] = len(bboxes)
        
        return result
    
    def _parse_bboxes(self, text: str, size: tuple) -> list:
        """様々な出力形式からbboxを抽出"""
        w, h = size
        bboxes = []
        
        # JSONパース試行
        try:
            json_match = re.search(r'[\[\{].*[\]\}]', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "applicant_fields" in data:
                    for f in data["applicant_fields"]:
                        if "bbox" in f and len(f["bbox"]) >= 4:
                            b = f["bbox"]
                            bboxes.append({
                                "bbox_normalized": b,
                                "bbox_pixel": [int(b[0]/1000*w), int(b[1]/1000*h), int(b[2]/1000*w), int(b[3]/1000*h)]
                            })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for key in ["bbox", "bbox_2d", "bbox_0100", "bbox_01000", "bbox_0_to_1000"]:
                                if key in item and isinstance(item[key], list) and len(item[key]) >= 4:
                                    b = item[key]
                                    bboxes.append({
                                        "bbox_normalized": b,
                                        "bbox_pixel": [int(b[0]/1000*w), int(b[1]/1000*h), int(b[2]/1000*w), int(b[3]/1000*h)]
                                    })
                                    break
        except:
            pass
        
        # Fallback: 正規表現で座標抽出
        if not bboxes:
            for match in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
                b = [int(x) for x in match]
                bboxes.append({
                    "bbox_normalized": b,
                    "bbox_pixel": [int(b[0]/1000*w), int(b[1]/1000*h), int(b[2]/1000*w), int(b[3]/1000*h)]
                })
        
        return bboxes
