"""
HuggingFace Spaces - Gradio App
===============================
PDFフォームから担当者が記入するフィールドを検出するAPI
"""

import os
import json
import base64
from io import BytesIO

import gradio as gr
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
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


# グローバル変数
model = None
processor = None


def load_model():
    """モデルを遅延ロード"""
    global model, processor
    
    if model is not None:
        return
    
    print("Loading model...")
    
    base_model = "Qwen/Qwen3-VL-8B-Instruct"
    lora_adapter = "takumi123xxx/pdfme-form-field-detector-lora"
    
    # プロセッサをロード
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    
    # 4bit量子化設定
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # ベースモデルをロード
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRAアダプターをロード
    model = PeftModel.from_pretrained(model, lora_adapter)
    model.eval()
    
    print("Model loaded successfully!")


def denormalize_bbox(bbox: list[int], img_width: int, img_height: int) -> list[int]:
    """0-1000正規化座標をピクセル座標に変換"""
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / 1000) * img_width),
        int((y1 / 1000) * img_height),
        int((x2 / 1000) * img_width),
        int((y2 / 1000) * img_height),
    ]


def detect_fields(image: Image.Image, custom_prompt: str = None) -> dict:
    """
    画像から入力フィールドを検出
    
    Args:
        image: 入力画像
        custom_prompt: カスタムプロンプト（オプション）
    
    Returns:
        検出結果（JSON形式）
    """
    load_model()
    
    if image is None:
        return {"error": "画像が提供されていません"}
    
    # RGBに変換
    image = image.convert("RGB")
    
    # プロンプトを設定
    prompt = custom_prompt if custom_prompt else USER_PROMPT
    
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
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 入力準備
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
    
    # デコード
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
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
                    "bbox": denormalize_bbox(
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


def process_image(image, custom_prompt):
    """Gradioインターフェース用のラッパー"""
    result = detect_fields(image, custom_prompt if custom_prompt else None)
    return json.dumps(result, ensure_ascii=False, indent=2)


# Gradioインターフェース
with gr.Blocks(title="PDFフォームフィールド検出") as demo:
    gr.Markdown("""
    # PDFフォームフィールド検出 API
    
    日本の書類画像から「申請者が記入すべきフォーム欄」を自動検出します。
    
    ## 使い方
    1. 書類の画像をアップロード
    2. 「検出実行」をクリック
    3. 検出結果がJSON形式で表示されます
    
    ## API利用
    このSpaceはAPIとしても利用できます。
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="書類画像")
            custom_prompt = gr.Textbox(
                label="カスタムプロンプト（オプション）",
                placeholder="デフォルトプロンプトを使用する場合は空欄のまま",
                lines=3
            )
            submit_btn = gr.Button("検出実行", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="検出結果（JSON）",
                lines=20,
                show_copy_button=True
            )
    
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, custom_prompt],
        outputs=output
    )
    
    gr.Markdown("""
    ## 出力形式
    
    ```json
    {
      "raw_output": "モデルの生の出力",
      "image_size": {"width": 1000, "height": 1414},
      "parsed": {
        "applicant_fields": [
          {"bbox": [x1, y1, x2, y2], "label": "氏名"}
        ]
      },
      "pixel_bboxes": [
        {"bbox": [100, 200, 300, 250]}
      ]
    }
    ```
    """)


if __name__ == "__main__":
    demo.launch()

