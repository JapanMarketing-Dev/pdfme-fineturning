---
language:
  - ja
license: mit
library_name: peft
base_model: Qwen/Qwen3-VL-32B-Instruct
tags:
  - vision-language-model
  - document-understanding
  - form-field-detection
  - lora
  - qwen3-vl
  - japanese
pipeline_tag: image-text-to-text
---

# PDFme Form Field Detection

日本の書類画像から「申請者が記入すべきフォーム欄」を自動検出するAIモデルのファインチューニングプロジェクト。

## 背景と目的

### 解決したい課題

日本の行政書類や申請書には、多くの入力欄があります。しかし、それらは大きく2種類に分かれます：

1. **申請者（担当者・顧客）が記入する欄** - 氏名、住所、電話番号など
2. **職員（役所・会社側）が記入する欄** - 受付番号、処理日、担当印など

書類を自動処理するシステムでは、「どこが申請者の入力欄か」を正確に判別する必要があります。

### このプロジェクトの目的

Vision-Language Model（VLM）をファインチューニングし、**初見の書類でも申請者の入力欄だけを検出できる汎用的なモデル**を作成します。

### 出力イメージ

```
入力: 書類の画像
出力: 申請者が記入すべきフィールドのbbox座標（JSON形式）
```

## モデル情報

| 項目 | 8Bモデル | 32Bモデル（最新） |
|------|----------|------------------|
| ベースモデル | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| 学習手法 | QLoRA（4bit量子化 + LoRA） | QLoRA（4bit量子化 + LoRA） |
| 学習データ | 90件（拡張データ） | 90件（拡張データ） |
| 公開先 | [takumi123xxx/pdfme-form-field-detector-lora](https://huggingface.co/takumi123xxx/pdfme-form-field-detector-lora) | [takumi123xxx/pdfme-form-field-detector-lora-32b](https://huggingface.co/takumi123xxx/pdfme-form-field-detector-lora-32b) |

## 性能評価

### 最新結果（32Bモデル）

90件の拡張データ（元データ10件 × 8種類の拡張 + 元データ）で学習した結果：

| 指標 | 32Bモデル | 8Bモデル | 説明 |
|------|-----------|----------|------|
| **Recall** | **13.56%** | 18.08% | 正解フィールドのうち検出できた割合 |
| **Precision** | **5.24%** | 7.90% | 検出したフィールドのうち正解だった割合 |
| **平均IoU** | **0.2163** | 0.2209 | 予測と正解の重なり具合 |
| IoU閾値 | 0.5 | 0.5 | 厳密な評価基準 |
| マッチ数 | 24/177 | 32/177 | 正解とマッチした予測数 |
| 予測数 | 458 | 405 | モデルが出力した総予測数 |

### サンプル別結果（32Bモデル）

| サンプル | Recall | Precision | IoU | 評価 |
|----------|--------|-----------|-----|------|
| **#2** | **60.00%** | **69.23%** | **0.507** | ⭐ 優秀 |
| **#7** | 33.33% | 25.00% | 0.380 | 良好 |
| **#9** | 18.18% | 7.69% | 0.313 | 改善 |
| #0 | 10.00% | 7.32% | 0.191 | - |
| #5 | 16.67% | 6.67% | 0.227 | - |
| #8 | 13.04% | 7.89% | 0.178 | - |
| その他 | 0-5% | 0-1% | 0.1-0.2 | 要改善 |

### 学習曲線（32Bモデル）

| Epoch | Loss | 備考 |
|-------|------|------|
| 開始 | 18.74 | - |
| 0.5 | 11.13 | 急速に減少 |
| 1.0 | 6.72 | 安定化 |
| 2.0 | 5.75 | 収束傾向 |
| 3.0 | **5.59** | 最終 |

**Loss改善: 18.74 → 5.59（70%減少）**

### モデルサイズ比較

| 項目 | 8Bモデル | 32Bモデル | 結論 |
|------|----------|-----------|------|
| パラメータ数 | 8B | 32B | 4倍 |
| 最終Loss | 5.60 | 5.59 | ≈同等 |
| Recall | 18.08% | 13.56% | 8Bがやや上 |
| VRAM（4bit） | ~20GB | ~40GB | 32Bは2倍 |

**結論**: モデルサイズを4倍にしても精度は同等。**データセットの量と多様性がボトルネック**。

### 現在の制限事項

1. **学習データの多様性が不足** - 元データ10件の拡張のため、新しいレイアウトへの汎化が限定的
2. **過検出傾向** - 予測数（458）が正解数（177）の2.6倍
3. **位置精度** - IoUが0.22と低く、境界の精度に課題

### 評価指標の定義

- **Recall**: 正解フィールドのうち、IoU≥閾値で検出できた割合
- **Precision**: 検出したフィールドのうち、IoU≥閾値で正解だった割合
- **IoU**: 予測と正解の重なり具合（0〜1、1が完全一致）

### 評価・データ拡張スクリプト

```bash
# 評価（32Bモデル）
python src/evaluate.py --lora takumi123xxx/pdfme-form-field-detector-lora-32b --base-model Qwen/Qwen3-VL-32B-Instruct

# 評価（8Bモデル）
python src/evaluate.py --lora takumi123xxx/pdfme-form-field-detector-lora --base-model Qwen/Qwen3-VL-8B-Instruct

# データ拡張（10件→90件）
python src/augment_data.py --output-dir /tmp/pdfme_augmented

# 拡張データで学習（32B）
python src/finetune.py --model Qwen/Qwen3-VL-32B-Instruct --augmented-data /tmp/pdfme_augmented/dataset
```

## クイックスタート

### 1. 環境準備

```bash
# リポジトリをクローン
git clone https://github.com/JapanMarketing-Dev/pdfme-fineturning.git
cd pdfme-fineturning

# HuggingFace Tokenを設定
export HF_TOKEN=your_huggingface_token
```

### 2. ファインチューニングの実行

```bash
# Dockerイメージをビルド＆実行
docker compose build
docker compose run finetune
```

完了すると、モデルは自動的にHugging Face Hubにアップロードされます。

### 3. 推論（テスト）

```bash
docker compose run finetune python src/inference.py \
  --model outputs/pdfme_lora_YYYYMMDD_HHMMSS \
  --image path/to/your/image.png
```

## 使い方（推論API）

### Pythonでの利用例（32Bモデル）

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# モデル読み込み（32B）
base_model = "Qwen/Qwen3-VL-32B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "takumi123xxx/pdfme-form-field-detector-lora-32b")
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

### レスポンス形式

```json
{
    "applicant_fields": [
    {"bbox": [100, 200, 500, 250]},
    {"bbox": [100, 300, 500, 350]}
    ],
    "count": 2
}
```

- `bbox`: 0-1000の正規化座標
- ピクセル座標への変換: `ピクセルX = bbox_x / 1000 * 画像幅`

## デモ

Hugging Face Spacesでデモを公開しています：
[takumi123xxx/pdfme-form-field-detector](https://huggingface.co/spaces/takumi123xxx/pdfme-form-field-detector)

## Hugging Face Inference Endpoints

### ⚠️ 重要：インスタンス選択について

| モデル | 推奨インスタンス | VRAM | 備考 |
|--------|-----------------|------|------|
| **32Bモデル（4bit）** | `nvidia-a100` | 40GB+ | ⭐推奨 |
| **8Bモデル（4bit）** | `nvidia-l4` または `nvidia-a10g` | 24GB | コスパ重視 |

### 環境変数

| 変数名 | デフォルト | 説明 |
|--------|------------|------|
| `BASE_MODEL` | `Qwen/Qwen3-VL-32B-Instruct` | ベースモデル |
| `USE_LORA` | `true` | LoRAアダプターを使用 |
| `USE_4BIT` | `true` | 4bit量子化を使用（推奨） |

## 技術的な補足

### なぜ32Bモデルでも8Bと同等の精度なのか

1. **データセットの限界**: 元データ10件では、大きなモデルの表現力を活かしきれない
2. **過学習**: 32Bモデルは少ないデータでより早く過学習する傾向
3. **タスクの複雑さ**: フォームフィールド検出は比較的単純なタスクで、8Bでも十分な場合がある

### 精度向上のために必要なこと

1. **元データを100件以上に増やす** - 最も効果的
2. **異なる書類タイプを追加** - 汎化性能向上
3. **アノテーションの精度向上** - 一貫したラベリング

### QLoRAの設定

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### 学習パラメータ

- エポック数: 3
- バッチサイズ: 1（gradient accumulation: 8）
- 学習率: 2e-4
- 量子化: 4bit（NF4）

## 今後の改善案

### 短期的改善（すぐに実施可能）

1. **データ収集の拡充** - 元データを100件以上に増やす
2. **エポック数の調整** - 32Bモデルは1-2エポックで十分な可能性
3. **評価データの分離** - 学習データとは別のテストセット作成

### 中期的改善（1-2週間）

4. **アノテーションの拡充** - フィールドの種類（氏名、住所、日付など）のラベル追加
5. **Multi-turn対話対応** - 「氏名欄だけ検出して」などの条件付き検出

### 長期的改善（1ヶ月以上）

6. **大規模データセット構築** - 様々な書類タイプ（住民票、確定申告、各種届出）を収集
7. **Active Learning** - 推論結果を人間がレビュー → 誤りをデータセットに追加

## ファイル構成

```
pdfme-fineturning/
├── Dockerfile              # Docker環境定義
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存関係
├── handler.py              # Inference Endpoints用ハンドラー
├── app.py                  # Gradio Spaceアプリ
├── MODEL_CARD.md           # Hugging Face用モデルカード
├── src/
│   ├── finetune.py         # ファインチューニング本体
│   ├── evaluate.py         # 評価スクリプト
│   ├── augment_data.py     # データ拡張スクリプト
│   └── inference.py        # ローカル推論スクリプト
└── README.md
```

## 必要環境

- Docker + NVIDIA Container Toolkit
- NVIDIA GPU（32Bモデル: 40GB+ VRAM、8Bモデル: 24GB+ VRAM）
- HuggingFace Token（読み書き権限）

## ライセンス

MIT License
