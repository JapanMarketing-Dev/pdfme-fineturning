FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# システム依存関係
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python環境設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN python -m pip install --upgrade pip

WORKDIR /app

# 依存関係インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Transformers最新版をソースからインストール（Qwen3-VL対応）
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers

# ソースコードコピー
COPY . .

# デフォルトコマンド
CMD ["python", "src/finetune.py"]

