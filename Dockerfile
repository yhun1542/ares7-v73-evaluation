FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="7.5" \
    FORCE_CUDA=1

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11을 기본 python3로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# pip 업그레이드
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging

# 프로젝트 파일 복사
COPY requirements.txt /app/

# 기본 패키지 먼저 설치
RUN pip3 install --no-cache-dir \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0

# PyTorch 설치 (CUDA 12.6 지원)
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# flash-attn 설치 (소스에서 빌드, Tesla T4는 Compute Capability 7.5)
# psutil을 먼저 설치해야 flash-attn 빌드 가능
RUN pip3 install --no-cache-dir psutil && \
    pip3 install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation failed, continuing without it"

# rotary-embedding-torch 설치
RUN pip3 install --no-cache-dir rotary-embedding-torch>=0.3.0

# 나머지 의존성 설치 (flash-attn 제외)
RUN pip3 install --no-cache-dir \
    pytorch-tabnet>=4.0 \
    catboost>=1.2 \
    xgboost>=2.0.0 \
    mojito2>=0.3.0 \
    ib_insync>=0.9.86 \
    aiohttp>=3.9.0 \
    requests>=2.31.0 \
    httpx>=0.25.0 \
    sqlalchemy>=2.0.0 \
    redis>=5.0.0 \
    prometheus-client>=0.19.0 \
    dash>=2.14.0 \
    plotly>=5.18.0 \
    python-dotenv>=1.0.0 \
    pyyaml>=6.0.0 \
    toml>=0.10.2 \
    python-dateutil>=2.8.0 \
    pytz>=2023.3 \
    tqdm>=4.66.0 \
    loguru>=0.7.0 \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    python-telegram-bot>=20.0 \
    openai>=1.0.0 \
    anthropic>=0.7.0 \
    yfinance>=0.2.0 \
    alpha_vantage>=2.3.0 \
    typing-extensions>=4.8.0

# 프로젝트 전체 복사
COPY . /app/

# 로그 및 결과 디렉토리 생성
RUN mkdir -p /app/logs /app/results /app/data

# 포트 노출
EXPOSE 8080

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); import sys; sys.exit(0 if torch.cuda.is_available() else 1)"

# 실행 명령
CMD ["python3", "main.py", "--mode", "backtest", "--symbols", "SPY,QQQ,IWM", "--capital", "1000000"]
