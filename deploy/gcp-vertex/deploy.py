"""
GCP Vertex AI Endpoint デプロイスクリプト
Qwen3-VL + LoRA ファインチューニング済みモデル
"""

import os
import json
from datetime import datetime
from google.cloud import aiplatform

# 設定
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("GCP_REGION", "asia-northeast1")
MACHINE_TYPE = os.environ.get("MACHINE_TYPE", "n1-standard-8")
ACCELERATOR_TYPE = os.environ.get("ACCELERATOR_TYPE", "NVIDIA_TESLA_T4")
ACCELERATOR_COUNT = int(os.environ.get("ACCELERATOR_COUNT", "1"))

# コンテナイメージ（事前にビルドしてArtifact Registryにプッシュ）
CONTAINER_IMAGE = os.environ.get("CONTAINER_IMAGE")

# モデル設定
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LORA_ADAPTER = "takumi123xxx/pdfme-form-field-detector-lora"


def create_endpoint():
    """Vertex AIエンドポイントを作成"""
    
    if not PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID環境変数を設定してください")
    if not CONTAINER_IMAGE:
        raise ValueError("CONTAINER_IMAGE環境変数を設定してください")
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # モデルをアップロード
    print("Uploading model...")
    model = aiplatform.Model.upload(
        display_name=f"pdfme-form-detector-{timestamp}",
        serving_container_image_uri=CONTAINER_IMAGE,
        serving_container_environment_variables={
            "BASE_MODEL": BASE_MODEL,
            "LORA_ADAPTER": LORA_ADAPTER,
            "USE_4BIT": "true",
        },
        serving_container_ports=[8000],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )
    
    print(f"Model uploaded: {model.resource_name}")
    
    # エンドポイント作成
    print("Creating endpoint...")
    endpoint = aiplatform.Endpoint.create(
        display_name=f"pdfme-form-detector-endpoint-{timestamp}",
    )
    
    # モデルをデプロイ
    print("Deploying model to endpoint...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"pdfme-deployed-{timestamp}",
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        min_replica_count=1,
        max_replica_count=1,
    )
    
    print(f"\n✅ Deployment complete!")
    print(f"Endpoint: {endpoint.resource_name}")
    print(f"Endpoint ID: {endpoint.name}")
    
    return endpoint


def test_endpoint(endpoint_id: str, image_path: str):
    """エンドポイントをテスト"""
    import base64
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    instances = [{"image_base64": image_base64}]
    
    response = endpoint.predict(instances=instances)
    print(f"Result: {json.dumps(response.predictions, indent=2, ensure_ascii=False)}")
    return response


def delete_endpoint(endpoint_id: str):
    """エンドポイントを削除"""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    endpoint = aiplatform.Endpoint(endpoint_id)
    endpoint.undeploy_all()
    endpoint.delete()
    print(f"Deleted endpoint: {endpoint_id}")


def build_and_push_container():
    """コンテナをビルドしてArtifact Registryにプッシュ"""
    import subprocess
    
    if not PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID環境変数を設定してください")
    
    repo_name = "pdfme-models"
    image_name = "pdfme-form-detector"
    tag = datetime.now().strftime("%Y%m%d%H%M%S")
    
    full_image_name = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{repo_name}/{image_name}:{tag}"
    
    # Artifact Registry リポジトリ作成
    print("Creating Artifact Registry repository...")
    subprocess.run([
        "gcloud", "artifacts", "repositories", "create", repo_name,
        "--repository-format=docker",
        f"--location={REGION}",
        f"--project={PROJECT_ID}",
    ], check=False)
    
    # Docker認証
    print("Configuring Docker auth...")
    subprocess.run([
        "gcloud", "auth", "configure-docker", f"{REGION}-docker.pkg.dev",
    ], check=True)
    
    # ビルド
    print("Building Docker image...")
    subprocess.run([
        "docker", "build", "-t", full_image_name,
        "-f", "../Dockerfile", ".."
    ], check=True)
    
    # プッシュ
    print("Pushing to Artifact Registry...")
    subprocess.run(["docker", "push", full_image_name], check=True)
    
    print(f"\n✅ Image pushed: {full_image_name}")
    return full_image_name


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "create", "test", "delete"])
    parser.add_argument("--endpoint-id", help="エンドポイントID")
    parser.add_argument("--image", help="テスト画像パス")
    args = parser.parse_args()
    
    if args.action == "build":
        image = build_and_push_container()
        print(f"\nSet environment variable:")
        print(f"export CONTAINER_IMAGE={image}")
    elif args.action == "create":
        create_endpoint()
    elif args.action == "test":
        if not args.endpoint_id or not args.image:
            print("--endpoint-id と --image が必要です")
        else:
            test_endpoint(args.endpoint_id, args.image)
    elif args.action == "delete":
        if not args.endpoint_id:
            print("--endpoint-id が必要です")
        else:
            delete_endpoint(args.endpoint_id)

