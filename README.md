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
| 公開先 | [takumi123xxx/pdfme-form-field-detector-lora](https://huggingface.co/takumi123xxx/pdfme-form-field-detector-lora) |

## 性能評価

### 学習曲線

| Epoch | Loss | 学習率 |
|-------|------|--------|
| 0.4 | 20.74 | 0 |
| 0.8 | 21.08 | 0.0002 |
| 1.0 | 20.84 | 0.000175 |
| 1.4 | 18.01 | 0.00015 |
| 1.8 | 17.05 | 0.000125 |
| 2.0 | 15.05 | 0.0001 |
| 2.4 | 12.88 | 0.000075 |
| 2.8 | 12.24 | 0.00005 |

**Loss改善: 20.74 → 12.24（41%減少）**

### 現在の制限事項

1. **学習データが少量（10件）** - 汎化性能に限界がある
2. **bbox座標の精度** - 正規化座標のため、ピクセル単位での精度は画像サイズに依存
3. **複雑なレイアウト** - 多段組みや複雑な書類では検出漏れの可能性

### 評価指標（今後測定予定）

- **検出率（Recall）**: 正解フィールドのうち、検出できた割合
- **適合率（Precision）**: 検出したフィールドのうち、正解だった割合
- **IoU（Intersection over Union）**: bbox座標の重なり具合

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
├── MODEL_CARD.md           # Hugging Face用モデルカード
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

### ⚠️ 重要：インスタンス選択について

このモデルは**8Bパラメータ**のVision-Language Modelです。デプロイ時は以下の点に注意してください：

| 条件 | 推奨インスタンス | VRAM | 備考 |
|------|-----------------|------|------|
| **4bit量子化あり** | `nvidia-l4` または `nvidia-a10g` | 24GB | ⭐推奨。コスパ最高 |
| **4bit量子化なし** | `nvidia-a10g` 以上 | 24GB+ | VRAMに余裕が必要 |

**🎯 推奨構成: `nvidia-l4` × 1（4bit量子化）**

### デプロイ手順（詳細）

1. **Hugging Face Hub**で [takumi123xxx/pdfme-form-field-detector-lora](https://huggingface.co/takumi123xxx/pdfme-form-field-detector-lora) にアクセス

2. **「Deploy」→「Inference Endpoints」**を選択

3. **基本設定**：

| 項目 | 設定値 | 説明 |
|------|--------|------|
| **Cloud Provider** | `AWS` | AWSが安定 |
| **Region** | `us-east-1` または `eu-west-1` | L4が利用可能な地域 |
| **Instance Type** | ⭐ **`nvidia-l4`** | 最も推奨 |
| **Instance Size** | `x1` | 1GPU |

4. **スケーリング設定**：

| 項目 | 設定値 | 説明 |
|------|--------|------|
| **Min Replicas** | `0` | 使わないときは課金なし |
| **Max Replicas** | `1` | 同時1インスタンス |

5. **Advanced Configuration**（重要）：

| 項目 | 設定値 | 説明 |
|------|--------|------|
| **Task** | `custom` | カスタムハンドラー使用 |
| **Container Type** | デフォルトのまま | 変更不要 |

6. **「Create Endpoint」**をクリック

### コスト目安（2025年時点）

| GPU | 1時間あたり | 月額（24時間） | 月額（Min=0、1日2時間使用） |
|-----|------------|---------------|---------------------------|
| L4 | ~$0.80 | ~$576 | ~$48 |
| A10G | ~$1.10 | ~$792 | ~$66 |
| T4 | ~$0.50 | ~$360 | ❌ VRAM不足の可能性 |

💡 **Min Replicas = 0** に設定すると、使わないときは課金されません
⚠️ Cold Start時に**1-3分**かかります（モデルロード時間）

### 環境変数（オプション）

| 変数名 | デフォルト | 説明 |
|--------|------------|------|
| `BASE_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | ベースモデル |
| `USE_LORA` | `true` | LoRAアダプターを使用 |
| `USE_4BIT` | `true` | 4bit量子化を使用（推奨） |

### トラブルシューティング

#### エラー: `PackageNotFoundError: bitsandbytes`

4bit量子化には`bitsandbytes`が必要です。handler.pyは自動的にフォールバックしますが、VRAMが足りない場合があります。

**解決策**: `nvidia-l4`または`nvidia-a10g`インスタンスを選択してください。

#### エラー: CUDA out of memory

VRAMが不足しています。

**解決策**: より大きなインスタンス（`nvidia-a10g`以上）を選択するか、`USE_4BIT=true`を確認してください。

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

### 短期的改善（すぐに実施可能）

1. **データ拡張**
   - 回転（±5度）、スケール変換、ノイズ追加で学習データを増やす
   - 10件 → 100件以上に拡張することで汎化性能向上

2. **ハイパーパラメータ調整**
   - エポック数を5-10に増加（過学習に注意）
   - 学習率スケジューラーの最適化
   - LoRAランクを32に上げて表現力向上

3. **評価パイプライン構築**
   - IoU、Precision、Recallの自動計算
   - テストデータセットの作成（学習データと別に）

### 中期的改善（1-2週間）

4. **アノテーションの拡充**
   - フィールドの種類（氏名、住所、日付など）のラベル追加
   - 職員欄のネガティブサンプル明示的に追加

5. **Multi-turn対話対応**
   - 「氏名欄だけ検出して」などの条件付き検出
   - 検出結果の修正・フィードバック機能

6. **モデルアンサンブル**
   - 複数のLoRAアダプターを学習し、投票で最終結果を決定

### 長期的改善（1ヶ月以上）

7. **大規模データセット構築**
   - 様々な書類タイプ（住民票、確定申告、各種届出）を収集
   - 1000件以上のアノテーション済みデータ

8. **より大きなモデルの活用**
   - Qwen3-VL-72B等、PEFT対応後に試行
   - 専用のObject Detectionモデルとのハイブリッド

9. **Active Learning**
   - 推論結果を人間がレビュー → 誤りをデータセットに追加
   - 継続的な精度向上サイクル

## トラブルシューティング

### HF Inference Endpointsで`qwen3_vl`が認識されないエラー

**エラー内容:**
```
ValueError: The checkpoint you are trying to load has model type `qwen3_vl` but Transformers does not recognize this architecture.
```

**原因:**
- Qwen3-VLは非常に新しいモデルで、transformersの最新版でしかサポートされていない
- HF Inference Endpointsのデフォルトのtransformersバージョンが古い

**解決策:**
`requirements_hf.txt`でtransformersをGitHubソースからインストールするように指定:
```
transformers @ git+https://github.com/huggingface/transformers.git
```

## ライセンス

MIT License
