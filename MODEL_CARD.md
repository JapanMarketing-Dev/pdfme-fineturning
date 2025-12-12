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

## Performance

### Training Progress

| Epoch | Loss | Learning Rate |
|-------|------|---------------|
| 0.4 | 20.74 | 0 |
| 1.0 | 20.84 | 0.000175 |
| 2.0 | 15.05 | 0.0001 |
| 2.8 | 12.24 | 0.00005 |

**Loss improved: 20.74 → 12.24 (41% reduction)**

### Current Limitations

1. **Small training dataset (10 samples)** - Limited generalization
2. **Bbox coordinate precision** - Depends on image size due to normalization
3. **Complex layouts** - May miss fields in multi-column or complex documents

### Metrics (To be measured)

- **Recall**: Percentage of ground truth fields detected
- **Precision**: Percentage of detected fields that are correct
- **IoU**: Overlap between predicted and ground truth bboxes

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
model = PeftModel.from_pretrained(model, "takumi123xxx/pdfme-form-field-detector-lora")
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

## Future Improvements

### Short-term

1. **Data augmentation** - Rotation, scaling, noise to expand training data
2. **Hyperparameter tuning** - Increase epochs, adjust learning rate
3. **Evaluation pipeline** - Automated IoU, Precision, Recall calculation

### Mid-term

4. **Field type classification** - Identify field types (name, address, date, etc.)
5. **Multi-turn dialogue** - Support conditional detection ("only detect name fields")
6. **Model ensemble** - Train multiple LoRA adapters and vote

### Long-term

7. **Large-scale dataset** - 1000+ annotated samples across document types
8. **Larger models** - Qwen3-VL-72B when PEFT compatible
9. **Active learning** - Human review → feedback → continuous improvement

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

## 性能評価

### 学習の進捗

| Epoch | Loss | 学習率 |
|-------|------|--------|
| 0.4 | 20.74 | 0 |
| 1.0 | 20.84 | 0.000175 |
| 2.0 | 15.05 | 0.0001 |
| 2.8 | 12.24 | 0.00005 |

**Loss改善: 20.74 → 12.24（41%減少）**

### 現在の制限事項

1. **学習データが少量（10件）** - 汎化性能に限界
2. **bbox座標の精度** - 正規化のため画像サイズに依存
3. **複雑なレイアウト** - 多段組みや複雑な書類では検出漏れの可能性

### 評価指標（今後測定予定）

- **検出率（Recall）**: 正解フィールドのうち、検出できた割合
- **適合率（Precision）**: 検出したフィールドのうち、正解だった割合
- **IoU**: bbox座標の重なり具合

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
model = PeftModel.from_pretrained(model, "takumi123xxx/pdfme-form-field-detector-lora")
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

## 今後の改善案

### 短期的改善（すぐに実施可能）

1. **データ拡張** - 回転、スケール変換、ノイズ追加で学習データを増やす
2. **ハイパーパラメータ調整** - エポック数増加、学習率最適化
3. **評価パイプライン構築** - IoU、Precision、Recallの自動計算

### 中期的改善（1-2週間）

4. **フィールド種類の分類** - 氏名、住所、日付などの種類を識別
5. **Multi-turn対話対応** - 「氏名欄だけ検出して」などの条件付き検出
6. **モデルアンサンブル** - 複数のLoRAアダプターを学習し、投票で決定

### 長期的改善（1ヶ月以上）

7. **大規模データセット構築** - 1000件以上のアノテーション済みデータ
8. **より大きなモデル** - Qwen3-VL-72B等、PEFT対応後に試行
9. **Active Learning** - 人間のレビュー→フィードバック→継続的改善

## 学習詳細

- **エポック数**: 3
- **バッチサイズ**: 1（勾配累積: 4）
- **学習率**: 2e-4
- **LoRAランク**: 16
- **LoRAアルファ**: 32

## ライセンス

Apache 2.0
