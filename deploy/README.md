# クラウドMLサービス デプロイガイド

Qwen3-VL + LoRAファインチューニング済みモデルを、マネージドMLサービスでAPI化する手順です。

## 対応サービス

| クラウド | サービス | 推奨インスタンス | VRAM | 料金目安（時間） |
|----------|---------|-----------------|------|-----------------|
| **AWS** | SageMaker | ml.g5.xlarge | 24GB | ~$1.20 |
| **GCP** | Vertex AI | n1-standard-8 + T4 | 16GB | ~$1.00 |
| **Azure** | AI Foundry (Azure ML) | Standard_NC4as_T4_v3 | 16GB | ~$1.10 |

---

## AWS SageMaker

### 前提条件

```bash
pip install boto3 sagemaker
aws configure  # AWSクレデンシャル設定
```

### IAMロール作成

SageMaker用のIAMロールが必要です：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
```

必要なポリシー：
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`（モデルアーティファクト用）

### デプロイ

```bash
cd deploy/aws-sagemaker

# 環境変数設定
export AWS_REGION=ap-northeast-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole

# エンドポイント作成
python deploy.py create
```

### 推論テスト

```bash
python deploy.py test \
  --endpoint-name pdfme-form-detector-20241214-123456 \
  --image /path/to/document.png
```

### Python SDKでの利用

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

### クリーンアップ

```bash
python deploy.py delete --endpoint-name pdfme-form-detector-xxxxx
```

### 推奨インスタンス

| インスタンス | GPU | VRAM | 料金/時間 | 用途 |
|-------------|-----|------|----------|------|
| **ml.g5.xlarge** | A10G | 24GB | ~$1.20 | ⭐推奨（8Bモデル） |
| ml.g4dn.xlarge | T4 | 16GB | ~$0.75 | 低コスト |
| ml.p4d.24xlarge | A100×8 | 640GB | ~$40.00 | 32Bモデル |

---

## GCP Vertex AI

### 前提条件

```bash
pip install google-cloud-aiplatform
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### デプロイ手順

#### 1. コンテナをビルド＆プッシュ

```bash
cd deploy/gcp-vertex

# 環境変数設定
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=asia-northeast1

# コンテナをArtifact Registryにプッシュ
python deploy.py build
```

#### 2. エンドポイント作成

```bash
# ビルド時に出力されたイメージURIを設定
export CONTAINER_IMAGE=asia-northeast1-docker.pkg.dev/your-project/pdfme-models/pdfme-form-detector:xxxxx

python deploy.py create
```

### Python SDKでの利用

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

### クリーンアップ

```bash
python deploy.py delete --endpoint-id projects/xxx/locations/xxx/endpoints/xxx
```

### 推奨インスタンス

| マシンタイプ | GPU | VRAM | 料金/時間 | 用途 |
|-------------|-----|------|----------|------|
| n1-standard-8 + T4 | T4 | 16GB | ~$1.00 | 低コスト |
| **n1-standard-8 + L4** | L4 | 24GB | ~$1.20 | ⭐推奨 |
| a2-highgpu-1g | A100 | 40GB | ~$4.00 | 高性能 |

---

## Azure AI Foundry (Azure ML)

### 前提条件

```bash
pip install azure-ai-ml azure-identity
az login
```

### ワークスペース作成

Azure MLワークスペースが必要です：

```bash
# リソースグループ作成
az group create --name pdfme-rg --location japaneast

# MLワークスペース作成
az ml workspace create \
  --name pdfme-workspace \
  --resource-group pdfme-rg \
  --location japaneast
```

### デプロイ

```bash
cd deploy/azure-foundry

# 環境変数設定
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export AZURE_RESOURCE_GROUP=pdfme-rg
export AZURE_ML_WORKSPACE=pdfme-workspace

# エンドポイント作成
python deploy.py create
```

### Python SDKでの利用

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

### クリーンアップ

```bash
python deploy.py delete --endpoint-name pdfme-detector-xxxxx
```

### 推奨インスタンス

| VMサイズ | GPU | VRAM | 料金/時間 | 用途 |
|---------|-----|------|----------|------|
| **Standard_NC4as_T4_v3** | T4 | 16GB | ~$1.10 | ⭐推奨 |
| Standard_NC6s_v3 | V100 | 16GB | ~$3.00 | 高性能 |
| Standard_NC24ads_A100_v4 | A100 | 80GB | ~$5.00 | 32Bモデル |

---

## API レスポンス形式

全サービス共通のレスポンス形式：

```json
{
  "bboxes": [
    {
      "bbox_normalized": [100, 200, 500, 250],
      "bbox_pixel": [120, 320, 600, 400]
    },
    {
      "bbox_normalized": [100, 300, 500, 350],
      "bbox_pixel": [120, 480, 600, 560]
    }
  ],
  "count": 2,
  "raw_output": "[{\"bbox_0100\": [100, 200, 500, 250]}, ...]",
  "image_size": {"width": 1200, "height": 1600}
}
```

| フィールド | 説明 |
|-----------|------|
| `bboxes` | 検出されたフィールドのリスト |
| `bbox_normalized` | 0-1000正規化座標 |
| `bbox_pixel` | ピクセル座標 |
| `count` | 検出数 |
| `raw_output` | モデルの生出力 |
| `image_size` | 入力画像サイズ |

---

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `BASE_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | ベースモデル |
| `LORA_ADAPTER` | `takumi123xxx/pdfme-form-field-detector-lora` | LoRAアダプター |
| `USE_4BIT` | `true` | 4bit量子化（推奨） |

### 32Bモデルを使う場合

```bash
export BASE_MODEL=Qwen/Qwen3-VL-32B-Instruct
export LORA_ADAPTER=takumi123xxx/pdfme-form-field-detector-lora-32b
```

⚠️ 32Bモデルには**40GB以上のVRAM**が必要です

---

## コスト比較

### 月額コスト目安（24時間稼働）

| サービス | インスタンス | 月額 |
|----------|-------------|------|
| AWS SageMaker | ml.g5.xlarge | ~$864 |
| GCP Vertex AI | n1-standard-8 + L4 | ~$864 |
| Azure AI Foundry | Standard_NC4as_T4_v3 | ~$792 |

### 最小構成（オンデマンド）

| サービス | 設定 | 月額（1日2時間） |
|----------|------|-----------------|
| AWS SageMaker | Serverless | ~$72 |
| GCP Vertex AI | 自動スケーリング | ~$72 |
| Azure AI Foundry | 最小インスタンス | ~$66 |

---

## トラブルシューティング

### モデルロードエラー

```
ValueError: The checkpoint you are trying to load has model type `qwen3_vl`...
```

**原因**: transformersバージョンが古い

**解決策**: requirements.txtでGitHubソースからインストール
```
transformers @ git+https://github.com/huggingface/transformers.git
```

### OOMエラー

```
CUDA out of memory
```

**原因**: VRAMが不足

**解決策**:
1. `USE_4BIT=true` を確認
2. より大きなインスタンスを選択
3. 8Bモデルの場合、最低16GB VRAMが必要

### タイムアウトエラー

**原因**: モデルロードに時間がかかる

**解決策**:
- SageMaker: `model_data_download_timeout` を延長
- Vertex AI: `deploy_timeout` を延長
- Azure: ヘルスチェック間隔を延長

---

## ファイル構成

```
deploy/
├── Dockerfile              # 共通Dockerイメージ
├── api_server.py           # FastAPI サーバー
├── README.md               # このファイル
├── aws-sagemaker/
│   ├── deploy.py           # デプロイスクリプト
│   └── code/
│       └── inference.py    # SageMaker推論スクリプト
├── gcp-vertex/
│   └── deploy.py           # デプロイスクリプト
└── azure-foundry/
    ├── deploy.py           # デプロイスクリプト
    ├── conda.yml           # 環境定義
    └── code/
        └── score.py        # Azure MLスコアリングスクリプト
```
