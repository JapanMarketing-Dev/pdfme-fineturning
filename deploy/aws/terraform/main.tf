# Terraform設定 - AWS EC2 + GPU
# Qwen3-VL + LoRA APIサーバー

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "ap-northeast-1"
}

variable "instance_type" {
  default = "g5.xlarge"  # 24GB VRAM, ~$1.00/hour
}

variable "key_name" {
  description = "EC2 Key Pair名"
}

# VPC（既存のものを使う場合はdata sourceで取得）
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# セキュリティグループ
resource "aws_security_group" "pdfme_api" {
  name        = "pdfme-api-sg"
  description = "Security group for PDFme API"
  vpc_id      = data.aws_vpc.default.id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # API
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "pdfme-api-sg"
  }
}

# Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*Ubuntu 22.04*"]
  }
}

# EC2インスタンス
resource "aws_instance" "pdfme_api" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.instance_type
  key_name      = var.key_name

  vpc_security_group_ids = [aws_security_group.pdfme_api.id]
  subnet_id              = data.aws_subnets.default.ids[0]

  # ルートボリューム（モデルキャッシュ用に大きめ）
  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    
    # Dockerインストール
    apt-get update
    apt-get install -y docker.io
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ubuntu
    
    # NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    systemctl restart docker
    
    # APIサーバー起動（Dockerイメージをビルド）
    cd /home/ubuntu
    git clone https://github.com/JapanMarketing-Dev/pdfme-fineturning.git
    cd pdfme-fineturning/deploy
    docker build -t pdfme-api:latest -f Dockerfile .
    
    docker run -d \
      --name pdfme-api \
      --gpus all \
      -p 8000:8000 \
      -e BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
      -e LORA_ADAPTER=takumi123xxx/pdfme-form-field-detector-lora \
      -e USE_4BIT=true \
      --restart unless-stopped \
      pdfme-api:latest
  EOF

  tags = {
    Name = "pdfme-form-detector-api"
  }
}

# Elastic IP（固定IP）
resource "aws_eip" "pdfme_api" {
  instance = aws_instance.pdfme_api.id
  domain   = "vpc"

  tags = {
    Name = "pdfme-api-eip"
  }
}

output "api_url" {
  value = "http://${aws_eip.pdfme_api.public_ip}:8000"
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_eip.pdfme_api.public_ip}"
}

output "health_check" {
  value = "curl http://${aws_eip.pdfme_api.public_ip}:8000/health"
}

