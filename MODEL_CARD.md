---
license: apache-2.0
language:
  - ja
  - en
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - vision
  - vlm
  - qwen
  - lora
  - document-understanding
  - form-detection
  - japanese
base_model: Qwen/Qwen3-VL-8B-Instruct
datasets:
  - hand-dot/pdfme-form-field-dataset
---

# PDFme Form Field Detector

**Detects form fields that applicants need to fill in Japanese documents.**

This model is fine-tuned from [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) using QLoRA to detect input fields in Japanese application forms, registration documents, and other official paperwork.

## What This Model Does

Given an image of a Japanese document, this model identifies the bounding boxes of form fields that **applicants/customers** should fill in, while **excluding fields meant for staff/officials**.

### Example Use Cases

- Automating form digitization
- Building PDF form generators
- Creating accessibility tools for document processing

## Model Details

| Item | Value |
|------|-------|
| Base Model | Qwen/Qwen3-VL-8B-Instruct |
| Fine-tuning Method | QLoRA (4-bit quantization + LoRA) |
| Training Data | [hand-dot/pdfme-form-field-dataset](https://huggingface.co/datasets/hand-dot/pdfme-form-field-dataset) |
| Output Format | JSON with normalized bbox coordinates (0-1000) |

## Quick Start

### Installation

```bash
pip install transformers peft torch accelerate bitsandbytes
```

### Inference

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# Load model
base_model = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "takumi123xxx/pdfme-form-field-detector")
processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

# Prepare prompt
system_prompt = """You are an expert at analyzing Japanese documents.
There are two types of input fields:
1. Fields for applicants/customers to fill → Target for detection
2. Fields for staff/officials to fill → Exclude from detection"""

user_prompt = """Detect all input fields that applicants should fill in this image.
Exclude fields for staff.
Return JSON with bbox coordinates (0-1000 normalized)."""

# Load image
image = Image.open("your_document.png").convert("RGB")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": user_prompt},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=2048)
result = processor.decode(output[0], skip_special_tokens=True)
print(result)
```

### Output Format

```json
{
  "applicant_fields": [
    {"bbox": [100, 200, 500, 250]},
    {"bbox": [100, 300, 500, 350]}
  ],
  "count": 2
}
```

- `bbox`: `[x1, y1, x2, y2]` normalized to 0-1000 scale
- To convert to pixels: `pixel_x = bbox_x / 1000 * image_width`

## Limitations

- Trained on a small dataset (10 samples) - may not generalize well to all document types
- Best suited for Japanese administrative/application forms
- May require additional fine-tuning for specific document formats

## Training Details

- **Epochs**: 3
- **Batch Size**: 1 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

## License

Apache 2.0

---

# PDFme フォームフィールド検出モデル

**日本の書類から、申請者が記入すべきフォーム欄を自動検出するモデル**

[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)をQLoRAでファインチューニングし、申請書や届出書などの入力欄を検出します。

## このモデルでできること

書類の画像を入力すると、**申請者（顧客）が記入すべき欄**の位置（bbox）を検出します。
**職員が記入する欄**（受付番号、処理日など）は除外されます。

### 活用例

- フォームのデジタル化自動化
- PDF帳票生成システム
- 書類処理のアクセシビリティ向上

## モデル情報

| 項目 | 内容 |
|------|------|
| ベースモデル | Qwen/Qwen3-VL-8B-Instruct |
| 学習手法 | QLoRA（4bit量子化 + LoRA） |
| 学習データ | [hand-dot/pdfme-form-field-dataset](https://huggingface.co/datasets/hand-dot/pdfme-form-field-dataset) |
| 出力形式 | JSON（0-1000正規化されたbbox座標） |

## 使い方

### インストール

```bash
pip install transformers peft torch accelerate bitsandbytes
```

### 推論コード

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# モデル読み込み
base_model = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "takumi123xxx/pdfme-form-field-detector")
processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

# プロンプト準備
system_prompt = """あなたは日本の書類を分析するエキスパートです。
書類には2種類の入力欄があります：
1. 担当者（申請者・顧客）が記入する欄 → 検出対象
2. 職員（役所・会社の担当者）が記入する欄 → 対象外"""

user_prompt = """この画像から、担当者が記入する入力フィールドの位置をすべて検出してください。
職員が記入する欄は除外してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。"""

# 画像読み込み
image = Image.open("書類.png").convert("RGB")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": user_prompt},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=2048)
result = processor.decode(output[0], skip_special_tokens=True)
print(result)
```

### 出力例

```json
{
  "applicant_fields": [
    {"bbox": [100, 200, 500, 250]},
    {"bbox": [100, 300, 500, 350]}
  ],
  "count": 2
}
```

- `bbox`: `[x1, y1, x2, y2]` は0-1000に正規化された座標
- ピクセル座標への変換: `ピクセルX = bbox_x / 1000 * 画像幅`

## 制限事項

- 学習データが10件と少量のため、すべての書類タイプに対応できない可能性があります
- 日本の行政書類・申請書に最適化されています
- 特定の書類形式には追加のファインチューニングが必要な場合があります

## 学習詳細

- **エポック数**: 3
- **バッチサイズ**: 1（勾配累積: 4）
- **学習率**: 2e-4
- **LoRAランク**: 16
- **LoRAアルファ**: 32

## ライセンス

Apache 2.0

