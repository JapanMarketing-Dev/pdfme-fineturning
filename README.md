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

| 項目 | 内容 |
|------|------|
| ベースモデル | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| 学習手法 | QLoRA（4bit量子化 + LoRA） |
| 学習データ | [hand-dot/pdfme-form-field-dataset](https://huggingface.co/datasets/hand-dot/pdfme-form-field-dataset) |
| 公開先 | [takumi123xxx/pdfme-form-field-detector](https://huggingface.co/takumi123xxx/pdfme-form-field-detector) |

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

### Pythonでの利用例

```python
import requests
import base64

# 画像をBase64エンコード
with open("application_form.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Hugging Face Inference Endpointsへリクエスト
response = requests.post(
    "https://your-endpoint.endpoints.huggingface.cloud",
    headers={"Authorization": "Bearer YOUR_HF_TOKEN"},
    json={
        "inputs": image_base64,
        "parameters": {"max_new_tokens": 2048}
    }
)

result = response.json()
print(result["predictions"])
```

### レスポンス形式

```json
{
  "predictions": {
    "applicant_fields": [
      {"bbox": [100, 200, 500, 250], "bbox_pixel": [120, 320, 600, 400]},
      {"bbox": [100, 300, 500, 350], "bbox_pixel": [120, 480, 600, 560]}
    ],
    "count": 2
  }
}
```

- `bbox`: 0-1000の正規化座標
- `bbox_pixel`: 元画像のピクセル座標

## プロンプト設計

モデルには以下のプロンプトで指示しています：

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

## ファイル構成

```
pdfme-fineturning/
├── Dockerfile              # Docker環境定義
├── docker-compose.yml      # Docker Compose設定
├── requirements.txt        # Python依存関係
├── handler.py              # Inference Endpoints用ハンドラー
├── src/
│   ├── download_dataset.py # データセット取得スクリプト
│   ├── finetune.py         # ファインチューニング本体
│   └── inference.py        # ローカル推論スクリプト
└── README.md
```

## 必要環境

- Docker + NVIDIA Container Toolkit
- NVIDIA GPU（推奨: 24GB+ VRAM）
- HuggingFace Token（読み書き権限）

## Hugging Face Inference Endpoints

ファインチューニング済みモデルをAPIとしてデプロイできます。

### デプロイ手順

1. Hugging Face Hubで [takumi123xxx/pdfme-form-field-detector](https://huggingface.co/takumi123xxx/pdfme-form-field-detector) にアクセス
2. 「Deploy」→「Inference Endpoints」を選択
3. GPU（A10G以上推奨）を選択してデプロイ

### 環境変数（オプション）

| 変数名 | デフォルト | 説明 |
|--------|------------|------|
| `BASE_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | ベースモデル |
| `USE_LORA` | `true` | LoRAアダプターを使用 |
| `USE_4BIT` | `true` | 4bit量子化を使用 |

## 技術的な補足

### なぜQwen3-VL-8B-Instructを選んだか

- 最初は`Qwen3-VL-30B-A3B-Thinking`（MoEモデル）を検討
- PEFTライブラリとの互換性問題があったため、安定した8Bモデルを採用
- 8Bでも十分な精度が期待できる

### QLoRAの設定

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

### 学習パラメータ

- エポック数: 3
- バッチサイズ: 1（gradient accumulation: 4）
- 学習率: 2e-4
- 量子化: 4bit（NF4）

## トラブルシューティング

### OOMエラーが出る場合

```bash
# バッチサイズを下げる
docker compose run finetune python src/finetune.py --batch-size 1

# gradient accumulation stepsを上げる（コード内で変更）
```

### モデルのロードが遅い場合

transformersを最新版にアップデート:
```bash
pip install git+https://github.com/huggingface/transformers
```

## 今後の改善案

1. **データ拡張** - 現在10件と少量なので、回転・ノイズ追加などで増やす
2. **Multi-turn対話** - フィールドの種類（氏名、住所など）も識別
3. **より大きなモデル** - PEFT互換の大型モデルが出たら試す

## ライセンス

MIT License
