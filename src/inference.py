#!/usr/bin/env python3
"""
PDFme Form Field Detection - 推論スクリプト
==========================================
ファインチューニングしたモデルで推論を実行します。
担当者（申請者）が記入するフィールドのみを検出。
"""

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from peft import PeftModel


# プロンプトテンプレート（finetune.pyと同じ）
SYSTEM_PROMPT = """あなたは日本の書類（申請書、届出書など）を分析するエキスパートです。
書類には2種類の入力欄があります：
1. 担当者（申請者・顧客）が記入する欄 → 検出対象
2. 職員（役所・会社の担当者）が記入する欄 → 対象外

担当者が記入する欄の例：氏名、住所、電話番号、生年月日、メールアドレス、署名欄など
職員が記入する欄の例：受付番号、処理日、担当印、審査欄、決裁欄など"""

USER_PROMPT = """この画像から、担当者（申請者・顧客）が記入する入力フィールドの位置をすべて検出してください。
職員が記入する欄は除外してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


def load_model(model_path, base_model=None, use_4bit=False):
    """モデルをロード（Qwen3-VL MoEモデル対応）"""
    from transformers import BitsAndBytesConfig
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # デフォルトベースモデル
    if base_model is None:
        base_model = "Qwen/Qwen3-VL-30B-A3B-Thinking"
    
    # LoRAアダプターの場合
    if model_path != base_model:
        print(f"ベースモデル: {base_model}")
        print(f"LoRAアダプター: {model_path}")
        
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )
        else:
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True,
            )
        
        model = PeftModel.from_pretrained(model, model_path)
        processor = AutoProcessor.from_pretrained(base_model, token=hf_token, trust_remote_code=True)
    else:
        # フルモデルの場合
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, token=hf_token, trust_remote_code=True)
    
    return model, processor


def predict(model, processor, image_path, custom_prompt=None):
    """画像から担当者用フォームフィールドを検出"""
    
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    
    # メッセージ構築
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": custom_prompt or USER_PROMPT},
            ],
        }
    ]
    
    # テキスト準備
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
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
    
    return response, image.size


def denormalize_bbox(bbox, img_width, img_height):
    """0-1000正規化座標をピクセル座標に変換"""
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / 1000) * img_width),
        int((y1 / 1000) * img_height),
        int((x2 / 1000) * img_width),
        int((y2 / 1000) * img_height),
    ]


def main():
    parser = argparse.ArgumentParser(description="PDFme Form Field 推論")
    parser.add_argument("--model", required=True, help="モデルパス（LoRAアダプターまたはフルモデル）")
    parser.add_argument("--base-model", default=None, help="ベースモデル（LoRA使用時）")
    parser.add_argument("--image", required=True, help="入力画像パス")
    parser.add_argument("--use-4bit", action="store_true", help="4bit量子化を使用")
    parser.add_argument("--output-json", default=None, help="結果をJSONファイルに保存")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDFme Form Field Detection - 推論")
    print("タスク: 担当者（申請者）が記入するフィールドの検出")
    print("=" * 60)
    
    # モデルロード
    print("\nモデルをロード中...")
    model, processor = load_model(args.model, args.base_model, args.use_4bit)
    
    # 推論実行
    print(f"\n画像: {args.image}")
    print("\n推論中...")
    
    result, img_size = predict(model, processor, args.image)
    
    print("\n=== 生の出力 ===")
    print(result)
    
    # JSONパース試行
    try:
        parsed = json.loads(result)
        print("\n=== パース済み結果 ===")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        
        # ピクセル座標に変換
        if "applicant_fields" in parsed:
            print("\n=== ピクセル座標 ===")
            for i, field in enumerate(parsed["applicant_fields"]):
                bbox_norm = field["bbox"]
                bbox_pixel = denormalize_bbox(bbox_norm, img_size[0], img_size[1])
                print(f"  フィールド {i+1}: {bbox_pixel}")
        
        # JSON保存
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"\n結果を保存: {args.output_json}")
            
    except json.JSONDecodeError:
        print("\n（JSON形式ではありません）")


if __name__ == "__main__":
    main()

