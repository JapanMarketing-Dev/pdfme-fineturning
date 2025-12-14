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
base_model: Qwen/Qwen3-VL-32B-Instruct
datasets:
  - hand-dot/pdfme-form-field-dataset
---

# PDFme Form Field Detector (32B)

**Detects form fields that applicants need to fill in Japanese documents.**

This model is fine-tuned from [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) using QLoRA to detect input fields in Japanese application forms, registration documents, and other official paperwork.

## What This Model Does

Given an image of a Japanese document, this model identifies the bounding boxes of form fields that **applicants/customers** should fill in, while **excluding fields meant for staff/officials**.

### Example Use Cases

- Automating form digitization
- Building PDF form generators
- Creating accessibility tools for document processing

## Model Details

| Item | Value |
|------|-------|
| Base Model | Qwen/Qwen3-VL-32B-Instruct |
| Fine-tuning Method | QLoRA (4-bit quantization + LoRA) |
| Training Data | [hand-dot/pdfme-form-field-dataset](https://huggingface.co/datasets/hand-dot/pdfme-form-field-dataset) (90 samples, augmented) |
| Output Format | JSON with normalized bbox coordinates (0-1000) |

## Performance

### Evaluation Results (IoU ≥ 0.5)

| Metric | 32B Model | 8B Model | Description |
|--------|-----------|----------|-------------|
| **Recall** | **13.56%** | 18.08% | Ground truth fields detected |
| **Precision** | **5.24%** | 7.90% | Correct predictions |
| **Average IoU** | **0.2163** | 0.2209 | Overlap between predicted and ground truth |
| Matches | 24/177 | 32/177 | Matched predictions |
| Predictions | 458 | 405 | Total predictions |

### Per-Sample Results (Best performers)

| Sample | Recall | Precision | IoU | Evaluation |
|--------|--------|-----------|-----|------------|
| **#2** | **60.00%** | **69.23%** | **0.507** | ⭐ Excellent |
| **#7** | 33.33% | 25.00% | 0.380 | Good |
| **#9** | 18.18% | 7.69% | 0.313 | Improved |

### Training Progress

| Epoch | Loss | Notes |
|-------|------|-------|
| Start | 18.74 | - |
| 0.5 | 11.13 | Rapid decrease |
| 1.0 | 6.72 | Stabilizing |
| 2.0 | 5.75 | Converging |
| 3.0 | **5.59** | Final |

**Loss improved: 18.74 → 5.59 (70% reduction)**

### Key Finding

Despite being 4x larger than the 8B model, the 32B model achieved similar accuracy. **The dataset (10 original samples) is the bottleneck**, not model capacity.

### Current Limitations

1. **Small training dataset** - 10 original samples, augmented to 90
2. **Over-detection tendency** - 458 predictions vs 177 ground truth (2.6x)
3. **Location precision** - Average IoU of 0.22 indicates room for improvement

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

# Load model (32B)
base_model = "Qwen/Qwen3-VL-32B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "takumi123xxx/pdfme-form-field-detector-lora-32b")
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

## Demo

Try the model on Hugging Face Spaces:
[takumi123xxx/pdfme-form-field-detector](https://huggingface.co/spaces/takumi123xxx/pdfme-form-field-detector)

## Cloud Deployment

### AWS SageMaker

```python
import boto3
import json
import base64

runtime = boto3.client("sagemaker-runtime", region_name="ap-northeast-1")

with open("document.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = runtime.invoke_endpoint(
    EndpointName="pdfme-form-detector-xxxxx",
    ContentType="application/json",
    Body=json.dumps({"inputs": image_base64})
)

result = json.loads(response["Body"].read().decode())
print(result)
```

### GCP Vertex AI

```python
from google.cloud import aiplatform
import base64

aiplatform.init(project="your-project-id", location="asia-northeast1")
endpoint = aiplatform.Endpoint("projects/xxx/locations/xxx/endpoints/xxx")

with open("document.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = endpoint.predict(instances=[{"image_base64": image_base64}])
print(response.predictions)
```

### Azure AI Foundry

```python
import requests
import base64

endpoint_url = "https://pdfme-detector-xxxxx.japaneast.inference.ml.azure.com/score"
api_key = "your-api-key"

with open("document.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

response = requests.post(
    endpoint_url,
    headers=headers,
    json={"image_base64": image_base64}
)
print(response.json())
```

### Recommended Instances

| Service | Instance | GPU | VRAM | Cost/hour |
|---------|----------|-----|------|-----------|
| **AWS SageMaker** | ml.g5.xlarge | A10G | 24GB | ~$1.20 |
| **GCP Vertex AI** | n1-standard-8 + L4 | L4 | 24GB | ~$1.20 |
| **Azure AI Foundry** | Standard_NC4as_T4_v3 | T4 | 16GB | ~$1.10 |

For detailed deployment instructions, see the [GitHub repository](https://github.com/JapanMarketing-Dev/pdfme-fineturning/tree/main/deploy).

## Training Details

- **Base Model**: Qwen/Qwen3-VL-32B-Instruct
- **Epochs**: 3
- **Batch Size**: 1 (with gradient accumulation of 8)
- **Learning Rate**: 2e-4
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Quantization**: 4-bit NF4
- **Training Time**: ~2 hours on RTX PRO 6000 (95GB VRAM)

## Comparison: 8B vs 32B

| Aspect | 8B Model | 32B Model |
|--------|----------|-----------|
| Parameters | 8B | 32B (4x larger) |
| Final Loss | 5.60 | 5.59 |
| Recall | 18.08% | 13.56% |
| VRAM (4-bit) | ~20GB | ~40GB |
| Inference Speed | Faster | Slower |

**Conclusion**: With only 90 training samples, both models perform similarly. **Data quantity and diversity are the bottleneck**, not model size.

## Future Improvements

### Short-term

1. **Expand original dataset** - 100+ diverse document samples
2. **Reduce epochs** - 1-2 epochs may be sufficient for 32B
3. **Separate test set** - Evaluate on unseen documents

### Mid-term

4. **Field type classification** - Identify field types (name, address, date, etc.)
5. **Multi-turn dialogue** - Support conditional detection ("only detect name fields")

### Long-term

6. **Large-scale dataset** - 1000+ annotated samples across document types
7. **Active learning** - Human review → feedback → continuous improvement

## License

Apache 2.0

---

# PDFme フォームフィールド検出モデル（32B）

**日本の書類から、申請者が記入すべきフォーム欄を自動検出するモデル**

[Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)をQLoRAでファインチューニングし、申請書や届出書などの入力欄を検出します。

## このモデルでできること

書類の画像を入力すると、**申請者（顧客）が記入すべき欄**の位置（bbox）を検出します。
**職員が記入する欄**（受付番号、処理日など）は除外されます。

## モデル情報

| 項目 | 内容 |
|------|------|
| ベースモデル | Qwen/Qwen3-VL-32B-Instruct |
| 学習手法 | QLoRA（4bit量子化 + LoRA） |
| 学習データ | 90件（拡張データ） |
| 出力形式 | JSON（0-1000正規化されたbbox座標） |

## 性能評価

### 評価結果（IoU ≥ 0.5）

| 指標 | 32Bモデル | 8Bモデル | 説明 |
|------|-----------|----------|------|
| **Recall** | **13.56%** | 18.08% | 正解フィールドの検出率 |
| **Precision** | **5.24%** | 7.90% | 予測の正解率 |
| **平均IoU** | **0.2163** | 0.2209 | 予測と正解の重なり |
| マッチ数 | 24/177 | 32/177 | マッチした予測数 |
| 予測数 | 458 | 405 | 総予測数 |

### 学習曲線

| Epoch | Loss | 備考 |
|-------|------|------|
| 開始 | 18.74 | - |
| 0.5 | 11.13 | 急速に減少 |
| 1.0 | 6.72 | 安定化 |
| 2.0 | 5.75 | 収束傾向 |
| 3.0 | **5.59** | 最終 |

**Loss改善: 18.74 → 5.59（70%減少）**

### 重要な発見

32Bモデルは8Bモデルと同等の精度でした。**データセット（元10件）がボトルネック**であり、モデルサイズではありません。

## デモ

Hugging Face Spacesでお試しください：
[takumi123xxx/pdfme-form-field-detector](https://huggingface.co/spaces/takumi123xxx/pdfme-form-field-detector)

## クラウドデプロイ

### 推奨インスタンス

| サービス | インスタンス | GPU | VRAM | 料金/時間 |
|----------|-------------|-----|------|----------|
| **AWS SageMaker** | ml.g5.xlarge | A10G | 24GB | ~$1.20 |
| **GCP Vertex AI** | n1-standard-8 + L4 | L4 | 24GB | ~$1.20 |
| **Azure AI Foundry** | Standard_NC4as_T4_v3 | T4 | 16GB | ~$1.10 |

詳細なデプロイ手順は[GitHubリポジトリ](https://github.com/JapanMarketing-Dev/pdfme-fineturning/tree/main/deploy)を参照してください。

## 学習詳細

- **ベースモデル**: Qwen/Qwen3-VL-32B-Instruct
- **エポック数**: 3
- **バッチサイズ**: 1（勾配累積: 8）
- **学習率**: 2e-4
- **LoRAランク**: 16
- **LoRAアルファ**: 32
- **量子化**: 4bit NF4
- **学習時間**: RTX PRO 6000（95GB VRAM）で約2時間

## ライセンス

Apache 2.0
