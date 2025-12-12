#!/usr/bin/env python3
"""
PDFme Form Field Detection - ファインチューニングスクリプト
==========================================================
Qwen3-VL-8B-InstructをQLoRAでファインチューニングします。
PDFフォームの「担当者（申請者・顧客）が入力するフィールド」の位置を検出するタスク。
職員が入力する欄は対象外。
"""

import os
import json
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# モデル設定（Qwen3-VL-8B-Instruct: 8Bパラメータ、PEFTと完全互換）
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# プロンプトテンプレート（日本語の書類向け）
SYSTEM_PROMPT = """あなたは日本の書類（申請書、届出書など）を分析するエキスパートです。
書類には2種類の入力欄があります：
1. 担当者（申請者・顧客）が記入する欄 → 検出対象
2. 職員（役所・会社の担当者）が記入する欄 → 対象外

担当者が記入する欄の例：氏名、住所、電話番号、生年月日、メールアドレス、署名欄など
職員が記入する欄の例：受付番号、処理日、担当印、審査欄、決裁欄など"""

USER_PROMPT = """この画像から、担当者（申請者・顧客）が記入する入力フィールドの位置をすべて検出してください。
職員が記入する欄は除外してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""


def normalize_bbox_to_1000(bbox, img_width, img_height):
    """
    ピクセル座標を0-1000正規化座標に変換
    入力: [x1, y1, x2, y2] (ピクセル座標)
    出力: [x1, y1, x2, y2] (0-1000正規化)
    """
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / img_width) * 1000),
        int((y1 / img_height) * 1000),
        int((x2 / img_width) * 1000),
        int((y2 / img_height) * 1000),
    ]


def prepare_training_data(ds):
    """
    HuggingFaceデータセットをファインチューニング用に変換
    bboxesは「担当者が記入するフィールド」のみを含む
    """
    samples = []
    
    for i, item in enumerate(tqdm(ds["train"], desc="データ準備")):
        img = item["image"]
        bboxes = item["bboxes"]
        img_width, img_height = img.size
        
        # bboxesを正規化
        normalized_bboxes = [
            normalize_bbox_to_1000(bbox, img_width, img_height)
            for bbox in bboxes
        ]
        
        # 回答フォーマット: JSON形式でbboxリストを返す
        answer = json.dumps({
            "applicant_fields": [
                {"bbox": bbox}
                for bbox in normalized_bboxes
            ],
            "count": len(normalized_bboxes)
        }, ensure_ascii=False)
        
        samples.append({
            "image": img,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT,
            "answer": answer,
            "image_width": img_width,
            "image_height": img_height,
        })
    
    return samples


class FormFieldDataCollator:
    """データコレーター"""
    
    def __init__(self, processor, max_length=2048):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch):
        images = []
        texts = []
        
        for item in batch:
            image = item["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image)
            
            # チャット形式のテキスト構築（システムプロンプト付き）
            messages = [
                {
                    "role": "system",
                    "content": item["system_prompt"],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item["user_prompt"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": item["answer"],
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False)
            texts.append(text)
        
        # プロセッサで処理（truncationを無効化して画像トークンの不整合を防ぐ）
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # ラベル設定
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDFme Form Field ファインチューニング")
    parser.add_argument("--model", default=MODEL_NAME, help="使用するモデル")
    parser.add_argument("--epochs", type=int, default=3, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=2e-4, help="学習率")
    parser.add_argument("--output-dir", default="outputs/pdfme_lora", help="出力ディレクトリ")
    parser.add_argument("--use-4bit", action="store_true", default=True, help="4bit量子化を使用")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PDFme Form Field Detection - ファインチューニング")
    print("=" * 60)
    print("タスク: 担当者（申請者）が記入するフィールドの検出")
    print("       ※職員が記入する欄は対象外")
    print("=" * 60)
    print(f"モデル: {args.model}")
    print(f"エポック: {args.epochs}")
    print(f"バッチサイズ: {args.batch_size}")
    print(f"学習率: {args.lr}")
    print(f"4bit量子化: {args.use_4bit}")
    print("=" * 60)
    
    # HuggingFace Token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("警告: HF_TOKENが設定されていません")
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n警告: GPUが見つかりません。CPUで実行します（非常に遅くなります）")
    
    # データセットをダウンロード
    print("\nデータセットをダウンロード中...")
    ds = load_dataset("hand-dot/pdfme-form-field-dataset")
    print(f"データ数: {len(ds['train'])}件")
    
    # データ準備
    print("\nデータを準備中...")
    samples = prepare_training_data(ds)
    dataset = Dataset.from_list(samples)
    print(f"準備完了: {len(dataset)}件")
    
    # プロセッサをロード
    print("\nプロセッサをロード中...")
    processor = AutoProcessor.from_pretrained(
        args.model,
        token=hf_token,
        trust_remote_code=True,
    )
    
    # モデルをロード
    print("\nモデルをロード中...")
    
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
    
    # LoRA準備（Qwen3-VLはprepare_model_for_kbit_trainingと互換性がないため直接設定）
    print("\nLoRA設定中...")
    
    # gradient_checkpointingを有効化
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # トレーニング設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )
    
    # Trainer作成
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=FormFieldDataCollator(processor),
    )
    
    # トレーニング実行
    print("\nトレーニング開始...")
    trainer.train()
    
    # 保存
    print("\nモデルを保存中...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # Hugging Faceにアップロード
    hf_repo = "takumi123xxx/pdfme-form-field-detector-lora"
    print(f"\nHugging Faceにアップロード中... ({hf_repo})")
    try:
        from huggingface_hub import HfApi, create_repo
        
        api = HfApi()
        
        # リポジトリを作成（存在しない場合）
        try:
            create_repo(hf_repo, token=hf_token, exist_ok=True, repo_type="model")
        except Exception as e:
            print(f"リポジトリ作成スキップ: {e}")
        
        # モデルをアップロード
        api.upload_folder(
            folder_path=output_dir,
            repo_id=hf_repo,
            token=hf_token,
            commit_message=f"Upload finetuned LoRA adapter - {timestamp}",
        )
        print(f"アップロード完了: https://huggingface.co/{hf_repo}")
    except Exception as e:
        print(f"アップロードエラー: {e}")
        print("ローカルに保存されたモデルを手動でアップロードしてください")
    
    print("\n" + "=" * 60)
    print("完了!")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"Hugging Face: https://huggingface.co/{hf_repo}")
    print("=" * 60)


if __name__ == "__main__":
    main()

