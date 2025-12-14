"""
AWS SageMaker Endpoint デプロイスクリプト
Qwen3-VL + LoRA ファインチューニング済みモデル
"""

import os
import json
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from datetime import datetime

# 設定
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")  # 必須
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.xlarge")  # 24GB VRAM

# モデル設定
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER = "takumi123xxx/pdfme-form-field-detector-lora"

def create_endpoint():
    """SageMakerエンドポイントを作成"""
    
    if not ROLE_ARN:
        raise ValueError("SAGEMAKER_ROLE_ARN環境変数を設定してください")
    
    session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # HuggingFaceモデル設定
    hub_config = {
        "HF_MODEL_ID": LORA_ADAPTER,
        "HF_TASK": "image-text-to-text",
        "BASE_MODEL": BASE_MODEL,
        "USE_4BIT": "true",
    }
    
    # カスタム推論コードを使用
    huggingface_model = HuggingFaceModel(
        model_data=None,  # HuggingFace Hubから直接ロード
        role=ROLE_ARN,
        transformers_version="4.51.0",
        pytorch_version="2.5.0",
        py_version="py311",
        env=hub_config,
        # カスタム推論スクリプト
        entry_point="inference.py",
        source_dir="code",
    )
    
    # エンドポイント名
    endpoint_name = f"pdfme-form-detector-{timestamp}"
    
    print(f"Creating endpoint: {endpoint_name}")
    print(f"Instance type: {INSTANCE_TYPE}")
    
    # デプロイ
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=endpoint_name,
    )
    
    print(f"\n✅ Endpoint created!")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Endpoint URL: https://runtime.sagemaker.{AWS_REGION}.amazonaws.com/endpoints/{endpoint_name}/invocations")
    
    return endpoint_name


def test_endpoint(endpoint_name: str, image_path: str):
    """エンドポイントをテスト"""
    import base64
    
    runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    payload = json.dumps({
        "inputs": image_base64,
        "parameters": {"max_new_tokens": 1024}
    })
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload
    )
    
    result = json.loads(response["Body"].read().decode())
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


def delete_endpoint(endpoint_name: str):
    """エンドポイントを削除"""
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    
    # エンドポイント削除
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Deleted endpoint: {endpoint_name}")
    
    # エンドポイント設定削除
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"Deleted endpoint config: {endpoint_name}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "test", "delete"])
    parser.add_argument("--endpoint-name", help="エンドポイント名")
    parser.add_argument("--image", help="テスト画像パス")
    args = parser.parse_args()
    
    if args.action == "create":
        create_endpoint()
    elif args.action == "test":
        if not args.endpoint_name or not args.image:
            print("--endpoint-name と --image が必要です")
        else:
            test_endpoint(args.endpoint_name, args.image)
    elif args.action == "delete":
        if not args.endpoint_name:
            print("--endpoint-name が必要です")
        else:
            delete_endpoint(args.endpoint_name)

