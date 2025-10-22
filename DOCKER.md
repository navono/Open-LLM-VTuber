# Docker 部署指南 / Docker Deployment Guide

## 中文说明

### 前置要求

- Docker Engine 20.10+
- Docker Compose v2.0+
- NVIDIA GPU 和 NVIDIA Container Toolkit (用于 GPU 加速)

### 快速开始

1. **准备配置文件**
   ```bash
   # 复制示例配置文件
   cp config_templates/conf.default.yaml conf.yaml
   # 或使用中文配置
   cp config_templates/conf.ZH.default.yaml conf.yaml
   
   # 编辑配置文件，填入你的 API keys 等信息
   nano conf.yaml
   ```

2. **构建并启动容器**
   ```bash
   # 使用 docker-compose
   docker-compose up -d
   
   # 或者使用 docker compose (新版本)
   docker compose up -d
   ```

3. **查看日志**
   ```bash
   docker-compose logs -f
   ```

4. **访问应用**
   
   在浏览器中打开: http://localhost:12393

### 仅构建 Docker 镜像

```bash
docker build -t open-llm-vtuber:latest -f dockerfile .
```

### 手动运行容器

```bash
docker run -d \
  --name open-llm-vtuber \
  --gpus all \
  -p 12393:12393 \
  -v $(pwd)/conf.yaml:/app/conf.yaml:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/avatars:/app/avatars \
  -v $(pwd)/backgrounds:/app/backgrounds \
  -v $(pwd)/characters:/app/characters \
  -v $(pwd)/live2d-models:/app/live2d-models \
  open-llm-vtuber:latest
```

### 常用命令

```bash
# 停止容器
docker-compose down

# 重启容器
docker-compose restart

# 查看容器状态
docker-compose ps

# 进入容器 shell
docker-compose exec open-llm-vtuber bash

# 查看实时日志
docker-compose logs -f open-llm-vtuber

# 重新构建镜像
docker-compose build --no-cache

# 清理并重新启动
docker-compose down && docker-compose up -d
```

### 使用 Hugging Face 镜像

如果你在中国大陆，可以取消 `docker-compose.yml` 中的注释来使用 HF 镜像：

```yaml
environment:
  - HF_ENDPOINT=https://hf-mirror.com
```

或者在运行时指定：

```bash
docker run -d \
  --name open-llm-vtuber \
  --gpus all \
  -p 12393:12393 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v $(pwd)/conf.yaml:/app/conf.yaml:ro \
  open-llm-vtuber:latest
```

### 故障排查

1. **容器无法启动**
   - 检查配置文件是否正确: `cat conf.yaml`
   - 查看容器日志: `docker-compose logs`

2. **GPU 不可用**
   - 确认已安装 NVIDIA Container Toolkit
   - 测试 GPU: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

3. **端口冲突**
   - 修改 `docker-compose.yml` 中的端口映射，例如 `"8080:12393"`

4. **模型下载缓慢**
   - 使用 HF 镜像 (见上文)
   - 或者预先下载模型到 `./models` 目录

---

## English Instructions

### Prerequisites

- Docker Engine 20.10+
- Docker Compose v2.0+
- NVIDIA GPU and NVIDIA Container Toolkit (for GPU acceleration)

### Quick Start

1. **Prepare Configuration**
   ```bash
   # Copy example configuration
   cp config_templates/conf.default.yaml conf.yaml
   
   # Edit the configuration file with your API keys
   nano conf.yaml
   ```

2. **Build and Start Container**
   ```bash
   # Using docker-compose
   docker-compose up -d
   
   # Or using docker compose (newer version)
   docker compose up -d
   ```

3. **View Logs**
   ```bash
   docker-compose logs -f
   ```

4. **Access Application**
   
   Open in browser: http://localhost:12393

### Build Docker Image Only

```bash
docker build -t open-llm-vtuber:latest -f dockerfile .
```

### Run Container Manually

```bash
docker run -d \
  --name open-llm-vtuber \
  --gpus all \
  -p 12393:12393 \
  -v $(pwd)/conf.yaml:/app/conf.yaml:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/avatars:/app/avatars \
  -v $(pwd)/backgrounds:/app/backgrounds \
  -v $(pwd)/characters:/app/characters \
  -v $(pwd)/live2d-models:/app/live2d-models \
  open-llm-vtuber:latest
```

### Common Commands

```bash
# Stop container
docker-compose down

# Restart container
docker-compose restart

# Check container status
docker-compose ps

# Enter container shell
docker-compose exec open-llm-vtuber bash

# View live logs
docker-compose logs -f open-llm-vtuber

# Rebuild image
docker-compose build --no-cache

# Clean and restart
docker-compose down && docker-compose up -d
```

### Using Hugging Face Mirror

If you're in mainland China, uncomment this line in `docker-compose.yml`:

```yaml
environment:
  - HF_ENDPOINT=https://hf-mirror.com
```

Or specify at runtime:

```bash
docker run -d \
  --name open-llm-vtuber \
  --gpus all \
  -p 12393:12393 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v $(pwd)/conf.yaml:/app/conf.yaml:ro \
  open-llm-vtuber:latest
```

### Troubleshooting

1. **Container won't start**
   - Check configuration file: `cat conf.yaml`
   - View container logs: `docker-compose logs`

2. **GPU not available**
   - Ensure NVIDIA Container Toolkit is installed
   - Test GPU: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

3. **Port conflict**
   - Modify port mapping in `docker-compose.yml`, e.g., `"8080:12393"`

4. **Slow model downloads**
   - Use HF mirror (see above)
   - Or pre-download models to `./models` directory

### Volume Mounts Explained

- `conf.yaml`: Main configuration file (read-only)
- `models/`: Downloaded AI models cache
- `cache/`: Application cache
- `logs/`: Application logs
- `avatars/`, `backgrounds/`, `characters/`, `live2d-models/`: User customization files

### Environment Variables

- `HF_HOME`: Hugging Face cache directory
- `MODELSCOPE_CACHE`: ModelScope cache directory
- `HF_ENDPOINT`: Hugging Face mirror endpoint (optional)

### Health Check

The container includes a health check that pings the application every 30 seconds. You can check the health status with:

```bash
docker inspect --format='{{.State.Health.Status}}' open-llm-vtuber
```
