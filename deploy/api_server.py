"""
Qwen3-VL + LoRA API Server
AWS/GCP/Azure対応のREST API
"""

import os
import json
import base64
import re
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# モデル設定（環境変数で上書き可能）
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", "takumi123xxx/pdfme-form-field-detector-lora")
USE_4BIT = os.environ.get("USE_4BIT", "true").lower() == "true"

# グローバル変数
model = None
processor = None

app = FastAPI(
    title="PDFme Form Field Detector API",
    description="日本の書類から申請者記入欄を検出するAPI",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """あなたは日本の書類を分析するエキスパートです。
担当者（申請者）が記入する欄を検出してください。
職員が記入する欄は除外してください。"""

USER_PROMPT = """この画像から、担当者が記入する入力フィールドの位置をすべて検出してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


class PredictRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None
    max_tokens: Optional[int] = 1024


class PredictResponse(BaseModel):
    bboxes: list
    count: int
    raw_output: str
    image_size: dict


def load_model():
    """モデルをロード"""
    global model, processor
    
    if model is not None:
        return
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    
    print(f"Loading model: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_ADAPTER}")
    print(f"4-bit quantization: {USE_4BIT}")
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        base_model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # LoRAアダプターをロード
    if LORA_ADAPTER:
        print(f"Loading LoRA: {LORA_ADAPTER}")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    else:
        model = base_model
    
    model.eval()
    print("Model loaded!")


def parse_bboxes(text: str, width: int, height: int) -> list:
    """出力テキストからbboxを抽出"""
    bboxes = []
    
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
                            "bbox_pixel": [
                                int(b[0]/1000*width),
                                int(b[1]/1000*height),
                                int(b[2]/1000*width),
                                int(b[3]/1000*height)
                            ]
                        })
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key in ["bbox", "bbox_2d", "bbox_0100", "bbox_01000", "bbox_0_to_1000"]:
                            if key in item and isinstance(item[key], list) and len(item[key]) >= 4:
                                b = item[key]
                                bboxes.append({
                                    "bbox_normalized": b,
                                    "bbox_pixel": [
                                        int(b[0]/1000*width),
                                        int(b[1]/1000*height),
                                        int(b[2]/1000*width),
                                        int(b[3]/1000*height)
                                    ]
                                })
                                break
    except:
        pass
    
    # Fallback: 正規表現
    if not bboxes:
        for match in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
            b = [int(x) for x in match]
            bboxes.append({
                "bbox_normalized": b,
                "bbox_pixel": [
                    int(b[0]/1000*width),
                    int(b[1]/1000*height),
                    int(b[2]/1000*width),
                    int(b[3]/1000*height)
                ]
            })
    
    return bboxes


def predict_image(image: Image.Image, prompt: str, max_tokens: int) -> dict:
    """画像から予測"""
    load_model()
    
    # 画像サイズ制限
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    width, height = image.size
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    
    bboxes = parse_bboxes(response, width, height)
    
    return {
        "bboxes": bboxes,
        "count": len(bboxes),
        "raw_output": response,
        "image_size": {"width": width, "height": height}
    }


@app.on_event("startup")
async def startup_event():
    """起動時にモデルをプリロード"""
    print("Preloading model...")
    load_model()
    print("Ready!")


@app.get("/health")
async def health():
    """ヘルスチェック"""
    return {"status": "healthy", "model": BASE_MODEL, "lora": LORA_ADAPTER}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Base64画像から予測"""
    try:
        # Base64デコード
        if request.image_base64.startswith("data:image"):
            _, data = request.image_base64.split(",", 1)
            img_data = base64.b64decode(data)
        else:
            img_data = base64.b64decode(request.image_base64)
        
        image = Image.open(BytesIO(img_data)).convert("RGB")
        prompt = request.prompt or USER_PROMPT
        
        result = predict_image(image, prompt, request.max_tokens or 1024)
        return PredictResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(1024)
):
    """ファイルアップロードから予測"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        prompt = prompt or USER_PROMPT
        
        result = predict_image(image, prompt, max_tokens)
        return PredictResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

