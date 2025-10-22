# ============================================
# Stage 1: Builder - 构建和安装依赖
# ============================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

# Proxy configuration (can be overridden at build time)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Set proxy environment variables if provided
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        git \
        curl \
        ca-certificates \
        build-essential \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv - modern Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY requirements.txt ./

# Install Python dependencies using uv
# Create a virtual environment for better isolation
RUN uv sync --frozen --no-dev

# ============================================
# Stage 2: Runtime - 最小化运行时镜像
# ============================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime

# Proxy configuration for runtime (optional, for model downloads etc.)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Set proxy environment variables if provided
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies (no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libxcb-xfixes0 \
        libxcb-shape0 \
        libsndfile1 \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy uv from builder stage
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/uv.lock ./

# Copy application code
COPY . .

# Initialize git submodules (for frontend)
RUN git submodule update --init --recursive || true

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models \
    MODELSCOPE_CACHE=/app/models

# Create necessary directories
RUN mkdir -p logs models cache avatars backgrounds characters live2d-models

# Expose port 12393 (default port)
EXPOSE 12393

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:12393/ || exit 1

# Run the server
CMD ["python3", "run_server.py"]
