"""
Hugging Face Inference Endpoints - Custom Handler
=================================================
PDFフォームから担当者（申請者）が記入するフィールドの位置を検出します。
"""

import os
import json
import base64
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from peft import PeftModel


# プロンプトテンプレート
SYSTEM_PROMPT = """あなたは日本の書類（申請書、届出書など）を分析するエキスパートです。
書類には2種類の入力欄があります：
1. 担当者（申請者・顧客）が記入する欄 → 検出対象
2. 職員（役所・会社の担当者）が記入する欄 → 対象外

担当者が記入する欄の例：氏名、住所、電話番号、生年月日、メールアドレス、署名欄など
職員が記入する欄の例：受付番号、処理日、担当印、審査欄、決裁欄など"""

USER_PROMPT = """この画像から、担当者（申請者・顧客）が記入する入力フィールドの位置をすべて検出してください。
職員が記入する欄は除外してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


class EndpointHandler:
    """Hugging Face Inference Endpoints用のカスタムハンドラー"""
    
    def __init__(self, path: str = ""):
        """
        モデルの初期化
        
        Args:
            path: モデルのパス（HFリポジトリまたはローカルパス）
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 環境変数から設定を取得
        base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking")
        use_lora = os.environ.get("USE_LORA", "false").lower() == "true"
        use_4bit = os.environ.get("USE_4BIT", "true").lower() == "true"
        
        print(f"Loading model from: {path or base_model}")
        print(f"Device: {self.device}")
        print(f"Use LoRA: {use_lora}")
        print(f"Use 4bit: {use_4bit}")
        
        # プロセッサをロード
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        
        # モデルをロード
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # LoRAアダプターをロード（指定された場合）
        if use_lora and path:
            print(f"Loading LoRA adapter from: {path}")
            self.model = PeftModel.from_pretrained(self.model, path)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        推論を実行
        
        Args:
            data: リクエストデータ
                - inputs: Base64エンコードされた画像、または画像URL
                - parameters: オプションのパラメータ
                    - custom_prompt: カスタムプロンプト
                    - max_new_tokens: 最大生成トークン数
        
        Returns:
            検出結果（JSON形式）
        """
        # 入力を取得
        inputs = data.get("inputs", data.get("image", ""))
        parameters = data.get("parameters", {})
        
        # 画像を読み込み
        image = self._load_image(inputs)
        if image is None:
            return {"error": "Invalid image input"}
        
        # パラメータを取得
        custom_prompt = parameters.get("custom_prompt", USER_PROMPT)
        max_new_tokens = parameters.get("max_new_tokens", 2048)
        
        # 推論を実行
        result = self._predict(image, custom_prompt, max_new_tokens)
        
        return result
    
    def _load_image(self, image_input: str) -> Image.Image | None:
        """画像を読み込み"""
        try:
            if image_input.startswith("data:image"):
                # Data URLの場合
                header, data = image_input.split(",", 1)
                image_data = base64.b64decode(data)
                image = Image.open(BytesIO(image_data))
            elif image_input.startswith("http://") or image_input.startswith("https://"):
                # URLの場合
                import requests
                response = requests.get(image_input, timeout=30)
                image = Image.open(BytesIO(response.content))
            else:
                # Base64の場合
                image_data = base64.b64decode(image_input)
                image = Image.open(BytesIO(image_data))
            
            return image.convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _predict(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int
    ) -> dict[str, Any]:
        """推論を実行"""
        
        # メッセージを構築
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # テキスト準備
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 入力準備
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # デコード
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # 結果を構築
        result = {
            "raw_output": response,
            "image_size": {
                "width": image.size[0],
                "height": image.size[1]
            }
        }
        
        # JSONパースを試行
        try:
            parsed = json.loads(response)
            result["parsed"] = parsed
            
            # ピクセル座標に変換
            if "applicant_fields" in parsed:
                result["pixel_bboxes"] = [
                    {
                        "bbox": self._denormalize_bbox(
                            field["bbox"],
                            image.size[0],
                            image.size[1]
                        )
                    }
                    for field in parsed["applicant_fields"]
                ]
        except json.JSONDecodeError:
            result["parsed"] = None
        
        return result
    
    def _denormalize_bbox(
        self,
        bbox: list[int],
        img_width: int,
        img_height: int
    ) -> list[int]:
        """0-1000正規化座標をピクセル座標に変換"""
        x1, y1, x2, y2 = bbox
        return [
            int((x1 / 1000) * img_width),
            int((y1 / 1000) * img_height),
            int((x2 / 1000) * img_width),
            int((y2 / 1000) * img_height),
        ]

