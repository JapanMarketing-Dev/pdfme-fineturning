# Qwen3-VL + LoRA API Server
# AWS/GCP/Azure対応

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# システム依存関係
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python環境
RUN python3.11 -m pip install --upgrade pip

# PyTorchとML依存関係
RUN pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# transformers（最新版が必要）
RUN pip install git+https://github.com/huggingface/transformers.git

# その他の依存関係
RUN pip install \
    accelerate>=1.2.0 \
    peft>=0.14.0 \
    bitsandbytes>=0.45.0 \
    qwen-vl-utils>=0.0.8 \
    pillow>=10.0.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    python-multipart>=0.0.12 \
    huggingface_hub>=0.26.0

# アプリケーションコード
WORKDIR /app
COPY api_server.py /app/

# ポート公開
EXPOSE 8000

# 環境変数
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# 起動コマンド
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

