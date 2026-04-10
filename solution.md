# onestep_pi 项目文档

本文档已重构为模块化结构，详细文档请查看 `openpi/docs_f1flow/` 目录。

## 文档索引

| 文档 | 内容 |
|------|------|
| [README](openpi/docs_f1flow/README.md) | 文档总览 |
| [01_env_setup.md](openpi/docs_f1flow/01_env_setup.md) | LIBERO-Plus 环境配置（双 venv 架构、安装、常见问题） |
| [02_dataset.md](openpi/docs_f1flow/02_dataset.md) | LIBERO-Plus 数据集下载、lerobot 补丁 |
| [03_training.md](openpi/docs_f1flow/03_training.md) | 训练与评估流程（从零训练 / 从 checkpoint 微调） |
| [04_l1flow_design.md](openpi/docs_f1flow/04_l1flow_design.md) | L1 Flow 集成到 π0.5 的技术方案 |

## 快速开始

### 1. 环境配置

```bash
# 克隆 LIBERO-plus
cd openpi/third_party
git clone https://github.com/sylvestf/LIBERO-plus.git

# 修复 setup.py 问题
echo "# Python package marker" > LIBERO-plus/libero/__init__.py

# 安装到 Client 环境
source examples/libero/.venv/bin/activate
cd LIBERO-plus && uv pip install .
uv pip install wand scikit-image
```

详见 [01_env_setup.md](openpi/docs_f1flow/01_env_setup.md)

### 2. 数据集下载

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

# 使用 hf_transfer 加速，带自动重试
max_retries=20; attempt=0
while [ $attempt -lt $max_retries ]; do
    attempt=$((attempt + 1))
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    uv run huggingface-cli download \
        Sylvest/libero_plus_lerobot \
        --repo-type dataset \
        --local-dir data/libero_plus_lerobot \
        --resume-download \
    && echo "下载完成！" && break
    sleep 60
done
```

详见 [02_dataset.md](openpi/docs_f1flow/02_dataset.md)

### 3. 训练（从本地 checkpoint 微调）

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

# HF_LEROBOT_HOME 让 lerobot 在 openpi/data/libero_plus_lerobot 查找数据
# （config 中 repo_id="data/libero_plus_lerobot"，解析为 HF_LEROBOT_HOME/data/libero_plus_lerobot）

# 安装 FFmpeg（LIBERO-Plus 视频解码必须，仅需执行一次）
uv pip install av

# 创建标准 soname 符号链接（仅需执行一次）
cd .venv/lib/python3.11/site-packages/av.libs/
for f in *.so.*; do
    soname=$(echo "$f" | sed -E 's/-[0-9a-f]{8}(\.so\.[0-9]+).*/\1/')
    if [ "$soname" != "$f" ] && [ ! -e "$soname" ]; then ln -sf "$f" "$soname"; fi
done
cd "$OPENPI_DIR"

# 计算 norm stats
HF_LEROBOT_HOME="$OPENPI_DIR" \
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow_from_ckpt

# 训练
LD_LIBRARY_PATH="$OPENPI_DIR/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$OPENPI_DIR" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 --overwrite
```

详见 [03_training.md](openpi/docs_f1flow/03_training.md)

### 4. 评估

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

export LIBERO_CONFIG_PATH=/tmp/libero && mkdir -p /tmp/libero
export PYTHONPATH="$OPENPI_DIR/third_party/LIBERO-plus:$OPENPI_DIR/packages/openpi-client/src:$OPENPI_DIR"

examples/libero/.venv/bin/python examples/libero/main.py \
    --policy_model pi05_libero_plus_l1_flow_from_ckpt \
    --ckpt_path checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/<step>/params \
    --benchmark_name libero_spatial \
    --num_trials_per_task 1
```

详见 [03_training.md](openpi/docs_f1flow/03_training.md)

## 技术方案

L1 Flow 如何集成到 π0.5：预测目标、损失函数、推理调度、参数冻结策略等，详见 [04_l1flow_design.md](openpi/docs_f1flow/04_l1flow_design.md)
