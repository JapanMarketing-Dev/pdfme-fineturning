"""
Azure AI Foundry (Azure ML) Endpoint デプロイスクリプト
Qwen3-VL + LoRA ファインチューニング済みモデル
"""

import os
import json
from datetime import datetime
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# 設定
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "Standard_NC4as_T4_v3")  # T4 16GB

# モデル設定
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER = "takumi123xxx/pdfme-form-field-detector-lora"


def get_ml_client():
    """Azure ML クライアントを取得"""
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def create_endpoint():
    """Azure MLエンドポイントを作成"""
    
    if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME]):
        raise ValueError("Azure環境変数を設定してください")
    
    ml_client = get_ml_client()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    endpoint_name = f"pdfme-detector-{timestamp}"
    
    # エンドポイント作成
    print("Creating endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="PDFme Form Field Detector API",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # 環境定義
    print("Creating environment...")
    env = Environment(
        name=f"pdfme-env-{timestamp}",
        image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest",
        conda_file="conda.yml",
    )
    
    # モデル登録
    print("Registering model...")
    model = Model(
        name=f"pdfme-model-{timestamp}",
        path="./code",
        type="custom_model",
    )
    ml_client.models.create_or_update(model)
    
    # デプロイメント作成
    print("Creating deployment...")
    deployment = ManagedOnlineDeployment(
        name="main",
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=CodeConfiguration(
            code="./code",
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        environment_variables={
            "BASE_MODEL": BASE_MODEL,
            "LORA_ADAPTER": LORA_ADAPTER,
            "USE_4BIT": "true",
        },
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    # トラフィックを100%に設定
    endpoint.traffic = {"main": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # エンドポイント情報取得
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    
    print(f"\n✅ Deployment complete!")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"API Key: {ml_client.online_endpoints.get_keys(endpoint_name).primary_key}")
    
    return endpoint_name


def test_endpoint(endpoint_name: str, image_path: str):
    """エンドポイントをテスト"""
    import base64
    import requests
    
    ml_client = get_ml_client()
    
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    headers = {
        "Authorization": f"Bearer {keys.primary_key}",
        "Content-Type": "application/json",
    }
    
    payload = json.dumps({
        "image_base64": image_base64,
    })
    
    response = requests.post(endpoint.scoring_uri, headers=headers, data=payload)
    result = response.json()
    
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    return result


def delete_endpoint(endpoint_name: str):
    """エンドポイントを削除"""
    ml_client = get_ml_client()
    ml_client.online_endpoints.begin_delete(endpoint_name).result()
    print(f"Deleted endpoint: {endpoint_name}")


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

