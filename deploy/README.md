# クラウドデプロイガイド

Qwen3-VL + LoRAファインチューニング済みモデルをクラウドでAPI化する手順です。

## 概要

| クラウド | 推奨インスタンス | GPU | VRAM | 料金目安（時間） |
|----------|-----------------|-----|------|-----------------|
| **AWS** | g5.xlarge | A10G | 24GB | ~$1.00 |
| **GCP** | n1-standard-8 + T4 | T4 | 16GB | ~$0.80 |
| **Azure** | Standard_NC4as_T4_v3 | T4 | 16GB | ~$0.90 |

## 前提条件

- Docker環境
- Terraform（IaCデプロイの場合）
- 各クラウドのCLI設定済み

---

## AWS

### 方法1: Terraform（推奨）

```bash
cd deploy/aws/terraform

# 変数を設定
export TF_VAR_key_name="your-key-pair"

# デプロイ
terraform init
terraform plan
terraform apply
```

### 方法2: シェルスクリプト

```bash
cd deploy/aws

# 環境変数を設定
export AWS_REGION=ap-northeast-1
export KEY_PAIR_NAME=your-key-pair
export SECURITY_GROUP=sg-xxxxxxxx
export SUBNET_ID=subnet-xxxxxxxx

# ECRにプッシュのみ
./deploy.sh

# EC2も起動する場合
LAUNCH_EC2=true ./deploy.sh
```

### 推奨インスタンス

| インスタンス | GPU | VRAM | 料金/時間 | 用途 |
|-------------|-----|------|----------|------|
| **g5.xlarge** | A10G | 24GB | ~$1.00 | ⭐推奨（8Bモデル） |
| g5.2xlarge | A10G | 24GB | ~$1.50 | 高CPU処理 |
| g4dn.xlarge | T4 | 16GB | ~$0.50 | 低コスト |
| p4d.24xlarge | A100×8 | 640GB | ~$32.00 | 32Bモデル |

---

## GCP

### Terraform

```bash
cd deploy/gcp/terraform

# 変数を設定
export TF_VAR_project_id="your-project-id"

# デプロイ
terraform init
terraform plan
terraform apply
```

### 推奨インスタンス

| マシンタイプ | GPU | VRAM | 料金/時間 | 用途 |
|-------------|-----|------|----------|------|
| n1-standard-8 + T4 | T4 | 16GB | ~$0.80 | 低コスト |
| **n1-standard-8 + L4** | L4 | 24GB | ~$1.00 | ⭐推奨 |
| a2-highgpu-1g | A100 | 40GB | ~$3.00 | 高性能 |

---

## Azure

### Terraform

```bash
cd deploy/azure/terraform

# 変数を設定
export TF_VAR_admin_password="YourSecurePassword123!"

# デプロイ
terraform init
terraform plan
terraform apply
```

### 推奨インスタンス

| VMサイズ | GPU | VRAM | 料金/時間 | 用途 |
|---------|-----|------|----------|------|
| **Standard_NC4as_T4_v3** | T4 | 16GB | ~$0.90 | ⭐推奨 |
| Standard_NC6s_v3 | V100 | 16GB | ~$2.50 | 高性能 |
| Standard_NC24ads_A100_v4 | A100 | 80GB | ~$4.00 | 32Bモデル |

---

## API仕様

### エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/health` | ヘルスチェック |
| POST | `/predict` | Base64画像で予測 |
| POST | `/predict/upload` | ファイルアップロードで予測 |

### Base64画像で予測

```bash
# 画像をBase64エンコード
IMAGE_BASE64=$(base64 -w 0 document.png)

# APIリクエスト
curl -X POST "http://YOUR_IP:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_BASE64\"}"
```

### ファイルアップロードで予測

```bash
curl -X POST "http://YOUR_IP:8000/predict/upload" \
  -F "file=@document.png"
```

### レスポンス例

```json
{
  "bboxes": [
    {
      "bbox_normalized": [100, 200, 500, 250],
      "bbox_pixel": [120, 320, 600, 400]
    }
  ],
  "count": 1,
  "raw_output": "[{\"bbox_0100\": [100, 200, 500, 250]}]",
  "image_size": {"width": 1200, "height": 1600}
}
```

---

## Python SDKサンプル

```python
import requests
import base64

API_URL = "http://YOUR_IP:8000"

def detect_fields(image_path: str) -> dict:
    """画像から入力欄を検出"""
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        f"{API_URL}/predict",
        json={"image_base64": image_base64}
    )
    return response.json()

# 使用例
result = detect_fields("application_form.png")
print(f"検出数: {result['count']}")
for bbox in result['bboxes']:
    print(f"  位置: {bbox['bbox_pixel']}")
```

---

## 環境変数

| 変数名 | デフォルト | 説明 |
|--------|-----------|------|
| `BASE_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | ベースモデル |
| `LORA_ADAPTER` | `takumi123xxx/pdfme-form-field-detector-lora` | LoRAアダプター |
| `USE_4BIT` | `true` | 4bit量子化（推奨） |

### 32Bモデルを使う場合

```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -e BASE_MODEL=Qwen/Qwen3-VL-32B-Instruct \
  -e LORA_ADAPTER=takumi123xxx/pdfme-form-field-detector-lora-32b \
  -e USE_4BIT=true \
  pdfme-api:latest
```

⚠️ 32Bモデルには**40GB以上のVRAM**が必要です（A100推奨）

---

## コスト最適化

### スポット/プリエンプティブインスタンス

| クラウド | 通常料金 | スポット料金 | 割引率 |
|----------|---------|-------------|--------|
| AWS Spot | $1.00/h | ~$0.30/h | 70% |
| GCP Preemptible | $0.80/h | ~$0.24/h | 70% |
| Azure Spot | $0.90/h | ~$0.27/h | 70% |

### 自動停止

使用していないときにインスタンスを停止することで、コストを大幅に削減できます。

```bash
# AWS
aws ec2 stop-instances --instance-ids i-xxxxxxxx

# GCP
gcloud compute instances stop pdfme-form-detector-api

# Azure
az vm deallocate --resource-group pdfme-form-detector-rg --name pdfme-form-detector-vm
```

---

## トラブルシューティング

### GPUが認識されない

```bash
# NVIDIAドライバー確認
nvidia-smi

# Docker内でGPU確認
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### モデルロードが遅い

- 初回起動時はモデルダウンロードに10-15分かかります
- 2回目以降はキャッシュされるため高速です

### OOMエラー

- `USE_4BIT=true` を確認
- より大きなVRAMのインスタンスを選択
- 8Bモデルの場合、最低16GB VRAMが必要

---

## セキュリティ

### 本番環境での推奨事項

1. **認証**: API Gatewayやnginxで認証を追加
2. **HTTPS**: Let's Encryptで証明書を取得
3. **ファイアウォール**: 信頼できるIPのみ許可
4. **VPC**: プライベートサブネットでの運用

```bash
# nginx + Let's Encrypt例
sudo apt install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

