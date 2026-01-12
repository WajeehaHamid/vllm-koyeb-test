# Dockerfile for deploying vLLM on Koyeb
# Optimized for GPU instances with CUDA support

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (this will take a while as it compiles CUDA kernels)
RUN pip3 install --no-cache-dir vllm

# Install additional dependencies for testing and API
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    requests \
    pydantic \
    aiohttp

# Copy test script
COPY koyeb_vllm_setup.py /app/
COPY koyeb_api_server.py /app/

# Create directory for model cache
RUN mkdir -p /app/hf_cache
ENV HF_HOME=/app/hf_cache

# Expose port for API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the API server
# Can be overridden to run test script instead
CMD ["python3", "koyeb_api_server.py"]
