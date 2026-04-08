# ── Stage 1: 依赖安装 ──────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: 运行镜像 ──────────────────────────────────
# python:3.10-slim 官方镜像同时提供 linux/amd64 和 linux/arm64
FROM python:3.10-slim

WORKDIR /app

# 系统依赖（matplotlib 渲染 + 中文字体 + 中文 locale 支持）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        locales \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i '/zh_CN.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

# 设置中文 UTF-8 环境，防止日志乱码
ENV LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    IN_DOCKER=1

# 从 builder 阶段复制已安装的 Python 包
COPY --from=builder /install /usr/local

# 复制项目源码
COPY prepare_data.py train.py evaluate.py retrain.py run_pipeline.py log_config.py font_config.py ./
COPY requirements.txt ./

# 复制测试文件（供容器内验证）
COPY tests/ ./tests/

# 复制数据集（已有初始数据时直接打包进镜像，避免首次运行生成合成数据）
COPY dataset/ ./dataset/

# 创建运行时目录
RUN mkdir -p outputs

# 清除 matplotlib 字体缓存，确保识别新安装的中文字体
RUN python -c "import matplotlib; import shutil; shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)"

# 构建时预下载 VGG16 预训练权重（~528MB），避免运行时下载
RUN echo "" && \
    echo "============================================================" && \
    echo "  正在下载 VGG16 预训练权重 (~528MB)..." && \
    echo "  首次构建需要下载，请耐心等待" && \
    echo "  后续构建会使用 Docker 缓存，无需重复下载" && \
    echo "============================================================" && \
    echo "" && \
    python -c "from torchvision import models; models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)" && \
    echo "" && \
    echo "  ✅ VGG16 预训练权重下载完成！" && \
    echo "============================================================" && \
    echo ""

CMD ["python", "run_pipeline.py"]
