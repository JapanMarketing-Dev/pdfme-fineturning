#!/bin/bash
# EC2起動時に実行されるスクリプト

set -e

# ログ出力
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting setup ==="

# Docker + NVIDIA Container Toolkitインストール
apt-get update
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# ECRログイン
AWS_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# コンテナ起動
ECR_REPO_NAME="pdfme-form-detector"
docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

docker run -d \
    --name pdfme-api \
    --gpus all \
    -p 8000:8000 \
    -e BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
    -e LORA_ADAPTER=takumi123xxx/pdfme-form-field-detector-lora \
    -e USE_4BIT=true \
    --restart unless-stopped \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

echo "=== Setup complete ==="

