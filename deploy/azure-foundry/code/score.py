"""
Azure ML スコアリングスクリプト
Qwen3-VL + LoRA
"""

import os
import json
import base64
import re
from io import BytesIO

import torch
from PIL import Image

model = None
processor = None

SYSTEM_PROMPT = """あなたは日本の書類を分析するエキスパートです。
担当者（申請者）が記入する欄を検出してください。
職員が記入する欄は除外してください。"""

USER_PROMPT = """この画像から、担当者が記入する入力フィールドの位置をすべて検出してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


def init():
    """モデルを初期化"""
    global model, processor
    
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    
    base_model_id = os.environ.get("BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    lora_adapter = os.environ.get("LORA_ADAPTER", "takumi123xxx/pdfme-form-field-detector-lora")
    use_4bit = os.environ.get("USE_4BIT", "true").lower() == "true"
    
    print(f"Loading model: {base_model_id}")
    print(f"LoRA adapter: {lora_adapter}")
    
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        base = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model = PeftModel.from_pretrained(base, lora_adapter)
    model.eval()
    print("Model loaded!")


def run(raw_data):
    """推論を実行"""
    global model, processor
    
    data = json.loads(raw_data)
    image_base64 = data.get("image_base64", "")
    
    # 画像をデコード
    if image_base64.startswith("data:image"):
        _, img_data = image_base64.split(",", 1)
        img_bytes = base64.b64decode(img_data)
    else:
        img_bytes = base64.b64decode(image_base64)
    
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    # 画像サイズ制限
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    width, height = image.size
    
    prompt = data.get("prompt", USER_PROMPT)
    max_tokens = data.get("max_new_tokens", 1024)
    
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


def parse_bboxes(text, width, height):
    """出力からbboxを抽出"""
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
                            "bbox_pixel": [int(b[0]/1000*width), int(b[1]/1000*height),
                                          int(b[2]/1000*width), int(b[3]/1000*height)]
                        })
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key in ["bbox", "bbox_2d", "bbox_0100", "bbox_01000"]:
                            if key in item and len(item[key]) >= 4:
                                b = item[key]
                                bboxes.append({
                                    "bbox_normalized": b,
                                    "bbox_pixel": [int(b[0]/1000*width), int(b[1]/1000*height),
                                                  int(b[2]/1000*width), int(b[3]/1000*height)]
                                })
                                break
    except:
        pass
    
    if not bboxes:
        for match in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
            b = [int(x) for x in match]
            bboxes.append({
                "bbox_normalized": b,
                "bbox_pixel": [int(b[0]/1000*width), int(b[1]/1000*height),
                              int(b[2]/1000*width), int(b[3]/1000*height)]
            })
    
    return bboxes

