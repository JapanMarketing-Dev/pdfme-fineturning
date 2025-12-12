# PDFme Form Field Detection - ファインチューニング

日本の書類（申請書、届出書など）から**担当者（申請者・顧客）が記入するフォームフィールド**の位置を検出するためのVision-Language Modelファインチューニング環境。

**注意**: 職員が記入する欄（受付番号、処理日、担当印など）は検出対象外。

## 概要

| 項目 | 内容 |
|------|------|
| データセット | [hand-dot/pdfme-form-field-dataset](https://huggingface.co/datasets/hand-dot/pdfme-form-field-dataset)（10件） |
| ベースモデル | [Qwen3-VL-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking)（MoE、31Bパラメータ） |
| タスク | 担当者が記入するフォームフィールドの位置（bbox）検出 |
| 手法 | QLoRA（4bit量子化 + LoRA） |

## 必要環境

- Docker + NVIDIA Container Toolkit
- NVIDIA GPU（推奨: 48GB+ VRAM、RTX PRO 6000等）
- HuggingFace Token

## セットアップ

### 1. 環境変数設定

```bash
export HF_TOKEN=your_huggingface_token
```

### 2. Docker環境でファインチューニング実行

```bash
# Dockerイメージをビルド
docker compose build

# ファインチューニング実行
docker compose run finetune
```

### 3. 推論テスト

```bash
docker compose run finetune python src/inference.py \
  --model outputs/pdfme_lora_YYYYMMDD_HHMMSS \
  --base-model Qwen/Qwen3-VL-30B-A3B-Thinking \
  --image data/images/sample_000.png \
  --use-4bit
```

## ファイル構成

```
pdfme-fineturning/
├── Dockerfile              # Docker設定
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存関係
├── src/
│   ├── download_dataset.py # データセット取得・確認
│   ├── finetune.py         # ファインチューニング本体
│   └── inference.py        # 推論スクリプト
└── README.md
```

## プロンプト設計

### システムプロンプト
```
あなたは日本の書類（申請書、届出書など）を分析するエキスパートです。
書類には2種類の入力欄があります：
1. 担当者（申請者・顧客）が記入する欄 → 検出対象
2. 職員（役所・会社の担当者）が記入する欄 → 対象外

担当者が記入する欄の例：氏名、住所、電話番号、生年月日、メールアドレス、署名欄など
職員が記入する欄の例：受付番号、処理日、担当印、審査欄、決裁欄など
```

### ユーザープロンプト
```
この画像から、担当者（申請者・顧客）が記入する入力フィールドの位置をすべて検出してください。
職員が記入する欄は除外してください。
結果はJSON形式で、各フィールドのbbox座標（0-1000正規化）を返してください。
```

## 出力フォーマット

```json
{
  "applicant_fields": [
    {"bbox": [100, 200, 500, 250]},
    {"bbox": [100, 300, 500, 350]}
  ],
  "count": 2
}
```

bbox座標は0-1000の正規化座標。

## 注意事項

- Qwen3-VL-30B-A3B-ThinkingはMoEモデル（31Bパラメータ、アクティブ3B）
- transformers最新版（4.57.0+）が必要
- データセットは10件と少量のため、過学習に注意
- 汎用性を高めるにはデータ拡張が有効

