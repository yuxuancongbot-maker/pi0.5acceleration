# L1 Flow 训练指南

本文档记录在 openpi 项目中使用 L1 Flow 方法微调 LIBERO 数据集的训练流程。

## 架构说明

openpi 项目包含两个主要组件：

1. **训练环境**：运行模型训练（JAX 或 PyTorch 后端）
2. **推理环境**：运行模型推理（client/server 架构）

### L1 Flow 核心改动

L1 Flow 不改变网络结构，仅改变三个方面：

| 方面 | 原始 Flow Matching | L1 Flow |
|------|-------------------|---------|
| 预测目标 | velocity (`noise - actions`) | sample (`actions`) |
| 损失函数 | MSE | L1 / MAE |
| 推理步数 | 多步 ODE (默认 10 步) | 2 步 (NFE=2) |

通过 `l1_flow=True` 配置标志统一切换，无需手动改代码。

### 参数冻结策略

openpi 中参数路径命名约定：
- **VLM backbone (PaliGemma)**：参数路径含 `llm`（不含 `_1`）
- **Action Expert**：参数路径含 `llm_1`（带 `_1` 后缀）

冻结 VLM、只训练 action expert 的正确写法：

```python
# 正确：精确匹配 action expert
freeze_filter = nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))
# 效果：冻结一切不匹配 "llm_1" 的参数 → 只有 action expert 可训练
```

> **注意**：不要使用 `nnx.All(nnx.Not(nnx_utils.PathRegex(".*llm.*")))`，这个 regex `".*llm.*"` 会同时匹配 VLM 和 action expert，导致两者都可训练，违背冻结 VLM 的意图。

## 一、环境安装

### 1.1 克隆仓库（含子模块）

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 如果已克隆，需要更新子模块
git submodule update --init --recursive
```

### 1.2 安装依赖

项目使用 `uv` 管理 Python 依赖。基础安装：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 是为了将 LeRobot 作为依赖拉取。

### 1.3 Python 版本

- 训练推荐使用 Python 3.11（如果使用 RLDS 数据）
- LIBERO LeRobot 格式训练可用 Python 3.8/3.10

### 1.4 验证安装

```bash
uv run python -c "import openpi; print('openpi imported successfully')"
```

## 二、数据准备

### 2.1 下载 LIBERO 数据集（LeRobot 格式）

openpi 使用 LeRobot 格式的 LIBERO 数据，通常在训练时由 `LeRobotLiberoDataConfig` 自动从 HuggingFace 拉取：

```python
# 配置中指定 repo_id，训练时自动下载
data=LeRobotLiberoDataConfig(
    repo_id="physical-intelligence/libero",
    base_config=DataConfig(prompt_from_task=True),
    extra_delta_transform=False,
)
```

如需手动预下载，可使用：

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='physical-intelligence/libero', repo_type='dataset', local_dir='./data/libero')
"
```

### 2.2 配置 LIBERO 环境变量（评估时需要）

训练本身不需要 LIBERO 仿真环境，但评估时需要：

```bash
export LIBERO_CONFIG_PATH=/tmp/libero
mkdir -p /tmp/libero

# 替换下面的路径为你的实际路径
OPENPI_DIR=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

cat > /tmp/libero/config.yaml << EOF
benchmark_root: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero
bddl_files: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero/bddl_files
init_states: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero/init_files
datasets: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/datasets
assets: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero/assets
EOF
```

### 2.3 计算归一化统计量

```bash
# 注意：norm stats 与数据集绑定，pi05_libero 和 pi05_libero_l1_flow 共用同一份
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

## 三、启动训练

### 3.1 JAX 训练命令

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_l1_flow \
    --exp-name=l1_flow_test \
    --overwrite
```
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_l1_flow \
    --exp-name=l1_flow_test \
    --overwrite \
    --fsdp-devices=4
```

```bash
export OPENPI_DATA_HOME=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi_cache
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_l1_flow \
    --exp-name=l1_flow_test \
    --overwrite
```

### 3.2 PyTorch 训练命令

```bash
# 单 GPU 训练
uv run scripts/train_pytorch.py pi05_libero_l1_flow \
    --exp_name=l1_flow_test \
    --save_interval 1000

# 多 GPU 训练
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> \
    scripts/train_pytorch.py pi05_libero_l1_flow \
    --exp_name=l1_flow_test
```

### 3.3 训练参数说明

| 参数 | 说明 |
|-----|------|
| `--exp-name` | 实验名称，用于区分不同运行 |
| `--overwrite` | 覆盖相同配置下的已有 checkpoint |
| `--resume` | 从最新 checkpoint 恢复训练 |

### 3.4 显存优化

```bash
# 允许 JAX 使用更多 GPU 显存（默认 75%，可设置到 90%）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

或使用 FSDP（多 GPU）：
```bash
--fsdp-devices <num_gpus>
```

## 四、训练配置详解

### 4.1 完整配置（`src/openpi/training/config.py:769-795`）

```python
TrainConfig(
    name="pi05_libero_l1_flow",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        l1_flow=True,  # 启用 L1 Flow: 预测 x1, 使用 L1 loss, 2-step 推理
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,  # LIBERO 动作已是 delta 格式
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,  # 与 peak_lr 相同 → warmup 后恒定学习率
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=30_000,
    # 冻结除 action expert (llm_1) 以外的所有参数
    freeze_filter=nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
)
```

### 4.2 关键配置解读

| 配置项 | 值 | 说明 |
|--------|---|------|
| `l1_flow` | `True` | 切换预测目标(→x1)、损失(→L1)、推理(→2步) |
| `freeze_filter` | `Not(PathRegex(".*llm.*_1.*"))` | 只训练 action expert，冻结 VLM 和其他参数 |
| `extra_delta_transform` | `False` | LIBERO 动作本身就是 delta 格式，不需额外转换 |
| `weight_loader` | `CheckpointWeightLoader(gs://...)` | 从 GCS 加载 pi0.5 预训练权重 |
| `decay_lr = peak_lr` | `5e-5` | warmup 后保持恒定学习率 |

### 4.3 freeze_filter 原理

```
trainable_filter = All(Param, Not(freeze_filter))

freeze_filter = Not(PathRegex(".*llm.*_1.*"))
             → 匹配所有路径中不含 "llm_1" 的参数（VLM + 其他）→ 冻结

trainable = Not(freeze_filter) 
          = Not(Not(PathRegex(".*llm.*_1.*")))
          = PathRegex(".*llm.*_1.*")
          → 只有 action expert 可训练
```

参考：`pi0_config.py:95` 中 `action_expert_params_filter = PathRegex(".*llm.*_1.*")`。

## 五、监控训练

### 5.1 查看训练进度

训练日志会输出到终端，同时保存到 `checkpoints` 目录。

### 5.2 wandb 监控

训练自动记录到 Weights & Biases，访问：

```
https://wandb.ai/yuxuancong-bot-dalian-university-of-technology/openpi
```

## 六、模型推理

训练完成后，启动 Policy Server 进行推理：

### 6.1 启动 Policy Server

```bash
# 使用训练好的 checkpoint（假设迭代 20000）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero_l1_flow \
    --policy.dir=checkpoints/pi05_libero_l1_flow/l1_flow_test/20000
```

### 6.2 运行评估 Client

启动 server 后，在另一个终端运行：

```bash
# LIBERO 评估
uv run examples/libero/main.py --env LIBERO
```

### 6.3 Docker 方式

```bash
# 授权 X11 访问
sudo xhost +local:docker

# Server 和 Client 分别配置
# compose.yml 中 server 使用 policy:checkpoint 参数
# client 使用 --env LIBERO 参数
docker compose -f examples/libero/compose.yml up --build
```

## 七、关键代码位置

| 组件 | JAX | PyTorch |
|------|-----|---------|
| `l1_flow` 配置标志 | `pi0_config.py:33` | 同左 |
| MixedTimestepSampler | `pi0.py:19-49` | `pi0_pytorch.py:52-81` |
| 预测目标切换 | `pi0.py:241` | `pi0_pytorch.py:371` |
| L1 vs MSE 损失 | `pi0.py:256-258` | `pi0_pytorch.py:417-419` |
| 2-step 推理 | `pi0.py:330-369` | `pi0_pytorch.py:471-496` |
| 训练配置 | `config.py:769-795` | 同左 |
| freeze_filter 工具 | `nnx_utils.py:46-63` | N/A |

## 八、常见问题

### 8.1 `uv sync` 依赖冲突

```bash
rm -rf .venv && uv sync
```

### 8.2 训练显存不足

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

或使用 FSDP（多 GPU）：
```bash
--fsdp-devices <num_gpus>
```

### 8.3 缺少 norm stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 8.4 推理连接错误

检查 server 是否在正确端口运行，网络连接和防火墙设置。

### 8.5 VLM 意外参与训练

如果发现显存占用远超预期或训练速度过慢，检查 `freeze_filter` 是否正确使用了 `".*llm.*_1.*"` regex 精确匹配 action expert。使用 `".*llm.*"` 会导致 VLM 和 action expert 同时训练。

## 九、停止训练

```bash
# Ctrl+C

# 或强制停止
pkill -f "scripts/train.py"
```
