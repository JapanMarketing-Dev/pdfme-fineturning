"""
HuggingFace Spaces - Gradio App (ZeroGPU対応)
PDFフォームから担当者が記入するフィールドを検出
"""

import os
import json
import spaces
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

# 4bit量子化のサポート確認
try:
    from transformers import BitsAndBytesConfig
    HAS_BNBITS = True
except ImportError:
    HAS_BNBITS = False
    BitsAndBytesConfig = None

SYSTEM_PROMPT = """あなたは日本の書類を分析するエキスパートです。
担当者（申請者）が記入する欄を検出してください。
職員が記入する欄は除外してください。"""

USER_PROMPT = """この画像から、担当者が記入する入力フィールドの位置をすべて検出してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""

# モデル設定
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER = "takumi123xxx/pdfme-form-field-detector-lora"

# グローバル変数
model = None
processor = None


def load_model():
    """モデルをロード"""
    global model, processor
    
    if model is not None:
        return
    
    print("Loading model...")
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # 4bit量子化が使える場合は使用
    if HAS_BNBITS:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # LoRAアダプターをロード
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
    model.eval()
    print("Model loaded!")


@spaces.GPU(duration=120)
def detect_fields(image: Image.Image, custom_prompt: str = None) -> str:
    """画像から入力フィールドを検出"""
    load_model()
    
    if image is None:
        return json.dumps({"error": "画像が提供されていません"}, ensure_ascii=False)
    
    image = image.convert("RGB")
    prompt = custom_prompt if custom_prompt else USER_PROMPT
    
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
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    
    result = {
        "raw_output": response,
        "image_size": {"width": image.size[0], "height": image.size[1]}
    }
    
    try:
        parsed = json.loads(response)
        result["parsed"] = parsed
        if "applicant_fields" in parsed:
            w, h = image.size
            result["pixel_bboxes"] = [
                [int(f["bbox"][0]/1000*w), int(f["bbox"][1]/1000*h),
                 int(f["bbox"][2]/1000*w), int(f["bbox"][3]/1000*h)]
                for f in parsed["applicant_fields"]
            ]
    except json.JSONDecodeError:
        result["parsed"] = None
    
    return json.dumps(result, ensure_ascii=False, indent=2)


# Gradio UI
with gr.Blocks(title="PDFフォームフィールド検出") as demo:
    gr.Markdown("""
    # PDFフォームフィールド検出
    
    日本の書類画像から「申請者が記入すべきフォーム欄」を自動検出します。
    
    **モデル**: Qwen3-VL-8B + LoRA (Recall: 18.08%)
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="書類画像")
            custom_prompt = gr.Textbox(
                label="カスタムプロンプト（オプション）",
                placeholder="空欄でデフォルトプロンプト使用",
                lines=2
            )
            submit_btn = gr.Button("検出実行", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="検出結果（JSON）", lines=20, show_copy_button=True)
    
    submit_btn.click(fn=detect_fields, inputs=[image_input, custom_prompt], outputs=output)

if __name__ == "__main__":
    demo.launch()
