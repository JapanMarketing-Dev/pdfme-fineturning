# Terraform設定 - GCP Compute Engine + GPU
# Qwen3-VL + LoRA APIサーバー

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

variable "project_id" {
  description = "GCPプロジェクトID"
}

variable "region" {
  default = "asia-northeast1"
}

variable "zone" {
  default = "asia-northeast1-a"
}

variable "machine_type" {
  default = "n1-standard-8"  # 8 vCPU, 30GB RAM
}

variable "gpu_type" {
  default = "nvidia-tesla-t4"  # 16GB VRAM
  # 他の選択肢: nvidia-l4 (24GB), nvidia-tesla-v100 (16GB), nvidia-a100-40gb
}

# ファイアウォールルール
resource "google_compute_firewall" "pdfme_api" {
  name    = "pdfme-api-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22", "8000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["pdfme-api"]
}

# GPUインスタンス
resource "google_compute_instance" "pdfme_api" {
  name         = "pdfme-form-detector-api"
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu"
      size  = 100
      type  = "pd-ssd"
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  tags = ["pdfme-api"]

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e
    
    # Dockerインストール（Deep Learning VMには既にインストール済みの場合あり）
    if ! command -v docker &> /dev/null; then
      apt-get update
      apt-get install -y docker.io
      systemctl start docker
      systemctl enable docker
    fi
    
    # NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    systemctl restart docker
    
    # APIサーバー起動
    cd /home
    git clone https://github.com/JapanMarketing-Dev/pdfme-fineturning.git || true
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

  labels = {
    app = "pdfme-form-detector"
  }
}

# 静的IP
resource "google_compute_address" "pdfme_api" {
  name   = "pdfme-api-ip"
  region = var.region
}

output "api_url" {
  value = "http://${google_compute_instance.pdfme_api.network_interface[0].access_config[0].nat_ip}:8000"
}

output "ssh_command" {
  value = "gcloud compute ssh pdfme-form-detector-api --zone=${var.zone}"
}

output "health_check" {
  value = "curl http://${google_compute_instance.pdfme_api.network_interface[0].access_config[0].nat_ip}:8000/health"
}

