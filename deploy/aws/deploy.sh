#!/bin/bash
# AWS EC2へのデプロイスクリプト
# 前提: AWS CLIが設定済み、ECRリポジトリが作成済み

set -e

# 設定（環境に合わせて変更）
AWS_REGION="${AWS_REGION:-ap-northeast-1}"
ECR_REPO_NAME="${ECR_REPO_NAME:-pdfme-form-detector}"
EC2_INSTANCE_TYPE="${EC2_INSTANCE_TYPE:-g5.xlarge}"  # 24GB VRAM
KEY_PAIR_NAME="${KEY_PAIR_NAME:-your-key-pair}"
SECURITY_GROUP="${SECURITY_GROUP:-sg-xxxxxxxx}"
SUBNET_ID="${SUBNET_ID:-subnet-xxxxxxxx}"

echo "=== AWS EC2 Deployment ==="

# 1. ECRリポジトリ作成（存在しない場合）
echo "Creating ECR repository..."
aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION 2>/dev/null || true

# 2. ECRログイン
echo "Logging into ECR..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 3. Dockerイメージビルド
echo "Building Docker image..."
cd "$(dirname "$0")/.."
docker build -t $ECR_REPO_NAME:latest -f Dockerfile .

# 4. イメージをECRにプッシュ
echo "Pushing to ECR..."
docker tag $ECR_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

echo "=== Image pushed to ECR ==="
echo "ECR URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"

# 5. EC2インスタンス起動（オプション）
if [ "$LAUNCH_EC2" = "true" ]; then
    echo "Launching EC2 instance..."
    
    # Deep Learning AMI (Ubuntu 22.04)
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text \
        --region $AWS_REGION)
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type $EC2_INSTANCE_TYPE \
        --key-name $KEY_PAIR_NAME \
        --security-group-ids $SECURITY_GROUP \
        --subnet-id $SUBNET_ID \
        --user-data file://user-data.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pdfme-form-detector}]" \
        --query 'Instances[0].InstanceId' \
        --output text \
        --region $AWS_REGION)
    
    echo "EC2 Instance ID: $INSTANCE_ID"
    echo "Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region $AWS_REGION)
    
    echo "=== Deployment Complete ==="
    echo "Public IP: $PUBLIC_IP"
    echo "API URL: http://$PUBLIC_IP:8000"
    echo "Health Check: http://$PUBLIC_IP:8000/health"
fi

echo "Done!"

