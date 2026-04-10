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
# 设置镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 多线程下载
source .venv/bin/activate
huggingface-cli download Sylvest/libero_plus_lerobot \
    --repo-type dataset \
    --local-dir /root/.cache/huggingface/lerobot/Sylvest/libero_plus_lerobot \
    --local-dir-use-symlinks False
```

详见 [02_dataset.md](openpi/docs_f1flow/02_dataset.md)

### 3. 训练（从本地 checkpoint 微调）

```bash
cd openpi
source .venv/bin/activate

# 计算 norm stats
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow_from_ckpt

# 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 --overwrite
```

详见 [03_training.md](openpi/docs_f1flow/03_training.md)

### 4. 评估

```bash
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/libero && mkdir -p /tmp/libero

python examples/libero/main.py \
    --policy_model pi05_libero_plus_l1_flow_from_ckpt \
    --ckpt_path checkpoints/libero_plus_from_ckpt29999/<step>/params \
    --benchmark_name libero_spatial \
    --num_trials_per_task 1
```

详见 [03_training.md](openpi/docs_f1flow/03_training.md)

## 技术方案

L1 Flow 如何集成到 π0.5：预测目标、损失函数、推理调度、参数冻结策略等，详见 [04_l1flow_design.md](openpi/docs_f1flow/04_l1flow_design.md)
