# L1 Flow on LIBERO-Plus 训练指南

本文档记录在 openpi 项目中使用 L1 Flow 方法在 **LIBERO-Plus** 数据集上微调的训练流程。
从本地已训练好的 `pi05_libero_l1_flow` checkpoint (step 29999) 出发，在 LIBERO-Plus 扰动数据集上继续微调。

## 与标准 LIBERO 训练的核心区别

| 项目 | LIBERO | LIBERO-Plus |
|------|--------|-------------|
| 数据集 repo_id | `physical-intelligence/libero` | `data/libero_plus_lerobot` |
| 初始权重 | GCS `gs://openpi-assets/...` | 本地 checkpoint `step 29999` |
| 训练配置名 | `pi05_libero_l1_flow` | `pi05_libero_plus_l1_flow_from_ckpt` |
| 必须设置的环境变量 | 无 | `HF_LEROBOT_HOME` |
| norm_stats | 自动从 GCS 下载 | 本地计算或直接复用 |

`HF_LEROBOT_HOME` 的作用：lerobot 在 `HF_LEROBOT_HOME/data/libero_plus_lerobot` 下查找数据，
设为 `openpi/` 目录后，`repo_id="data/libero_plus_lerobot"` 即对应 `openpi/data/libero_plus_lerobot`。

## 一、环境安装

环境与标准 LIBERO 训练完全相同，已安装可跳过。

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

**额外依赖（LIBERO-Plus 必须）**：LIBERO-Plus 数据集以 mp4 视频存储，需要 FFmpeg 解码库。
通过安装 `av`（PyAV）将 FFmpeg 打包进 `.venv`，无需系统级安装：

```bash
uv pip install av
```

`av` 的 wheel 里 FFmpeg 库文件名带 hash（如 `libavutil-34cda749.so.59.39.100`），
而 torchcodec 按标准名（`libavutil.so.59`）查找，需要手动创建符号链接：

```bash
cd .venv/lib/python3.11/site-packages/av.libs/
for f in *.so.*; do
    soname=$(echo "$f" | sed -E 's/-[0-9a-f]{8}(\.so\.[0-9]+).*/\1/')
    if [ "$soname" != "$f" ] && [ ! -e "$soname" ]; then
        ln -sf "$f" "$soname"
    fi
done
cd -
```

验证：

```bash
uv run python -c "import openpi; print('openpi imported successfully')"
uv run python -c "import av; print(f'av {av.__version__} ok')"
ls .venv/lib/python3.11/site-packages/av.libs/libavutil.so.59  # 应存在
```

## 二、数据准备

### 2.1 使用本地已有数据集（推荐，免下载）

服务器上 `research_lijinhao/pi0.5_deploy/libero_plus_lerobot` 已有完整数据集（19GB，28694 个视频），
直接软链接，无需从 HuggingFace 下载：

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
SRC="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_lijinhao/pi0.5_deploy/libero_plus_lerobot"

# 软链接数据集（不占额外磁盘空间）
ln -s "$SRC" "$OPENPI_DIR/data/libero_plus_lerobot"
```

### 2.2 复用已有 norm_stats（免重新计算）

源目录里已有预先计算好的 `norm_stats.json`，直接复制到训练代码期望的位置：

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
SRC="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_lijinhao/pi0.5_deploy/libero_plus_lerobot"

ASSETS_DIR="$OPENPI_DIR/assets/pi05_libero_plus_l1_flow_from_ckpt/data/libero_plus_lerobot"
mkdir -p "$ASSETS_DIR"
cp "$SRC/norm_stats.json" "$ASSETS_DIR/norm_stats.json"
```

norm_stats 保存位置：`assets/pi05_libero_plus_l1_flow_from_ckpt/data/libero_plus_lerobot/norm_stats.json`

### 2.3 验证数据集

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

HF_LEROBOT_HOME="$OPENPI_DIR" uv run python3 -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('data/libero_plus_lerobot')
print(f'total_episodes: {meta.total_episodes}')   # 预期: 14347
print(f'total_frames:   {meta.total_frames}')     # 预期: 2238036
print(f'root:           {meta.root}')
print('验证通过！')
"
```

### 2.4 （可选）从头计算 norm_stats

若需要重新计算（如数据有变动）：

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

HF_LEROBOT_HOME="$OPENPI_DIR" \
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow_from_ckpt
```

### 2.5 配置 LIBERO 环境变量（评估时需要，训练时可跳过）

```bash
export LIBERO_CONFIG_PATH=/tmp/libero
mkdir -p /tmp/libero

OPENPI_DIR=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

cat > /tmp/libero/config.yaml << EOF
benchmark_root: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero
bddl_files: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero/bddl_files
init_states: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero/init_files
datasets: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/datasets
assets: ${OPENPI_DIR}/third_party/LIBERO-plus/libero/libero/assets
EOF
```

## 三、启动训练

### 3.1 JAX 训练命令（标准）

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 \
    --overwrite
```

### 3.2 JAX 训练命令（关闭 wandb）

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=disabled \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 \
    --overwrite
```

### 3.3 JAX 训练命令（多 GPU FSDP）

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 \
    --overwrite \
    --fsdp-devices=4
```

### 3.4 恢复训练

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 \
    --resume
```

### 3.5 基于 checkpoint 29999 微调，保留原始权重不被覆盖

原始 checkpoint `pi05_libero_l1_flow/l1_flow_changed/29999` 仅作为初始化权重被**读取**，
新的 checkpoint 写入独立目录，原始文件完全不受影响。

权重流向示意：

```
checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/params   ← 只读，不修改
        ↓ weight_loader 加载
        训练（LIBERO-Plus 数据）
        ↓ 写入新目录
checkpoints/pi05_libero_plus_l1_flow_from_ckpt/<exp-name>/<step>/
```

使用不同 `--exp-name` 区分每次实验，去掉 `--overwrite` 以防止误删已有的微调结果：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=disabled \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_run1
# 不加 --overwrite：若同名 exp 已存在则报错退出，原始 29999 checkpoint 永远不受影响
```

若确认要覆盖**本次实验**的已有结果（不影响 29999）：

```bash
LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=disabled \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_run1 \
    --overwrite
# --overwrite 只删除 checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_run1/
# 不会触及 checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/
```

### 3.6 训练参数说明

| 参数 | 说明 |
|-----|------|
| `--exp-name` | 实验名称，用于区分不同运行 |
| `--overwrite` | 仅覆盖**当前 exp** 目录，不影响初始化用的 29999 checkpoint |
| `--resume` | 从最新 checkpoint 恢复训练 |
| `HF_LEROBOT_HOME` | 指定 lerobot 数据集根目录（必须设置） |

### 3.6 显存优化

```bash
# 允许 JAX 使用更多 GPU 显存（默认 75%，可设置到 90%）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

或使用 FSDP（多 GPU）：
```bash
--fsdp-devices <num_gpus>
```

## 四、训练配置详解

### 4.1 完整配置（`src/openpi/training/config.py`）

```python
TrainConfig(
    name="pi05_libero_plus_l1_flow_from_ckpt",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        l1_flow=True,  # 启用 L1 Flow: 预测 x1, 使用 L1 loss, 2-step 推理
    ),
    data=LeRobotLiberoPlusDataConfig(
        repo_id="data/libero_plus_lerobot",   # 配合 HF_LEROBOT_HOME 使用本地数据
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
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
    # 从本地 pi05_libero_l1_flow checkpoint (step 29999) 初始化
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/params"
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
| `repo_id` | `data/libero_plus_lerobot` | 配合 `HF_LEROBOT_HOME=$(pwd)` 指向本地数据 |
| `freeze_filter` | `Not(PathRegex(".*llm.*_1.*"))` | 只训练 action expert，冻结 VLM |
| `weight_loader` | 本地路径 `checkpoints/.../29999/params` | 从已训练的 L1 Flow checkpoint 初始化 |
| `decay_lr = peak_lr` | `5e-5` | warmup 后保持恒定学习率 |

### 4.3 初始化权重位置

```
checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/
├── params/          ← weight_loader 加载这里
├── train_state/
├── assets/
└── _CHECKPOINT_METADATA
```

## 五、监控训练

### 5.1 查看训练进度

训练日志输出到终端，checkpoint 保存到：

```
checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/<step>/
```

### 5.2 wandb 监控

```
https://wandb.ai/yuxuancong-bot-dalian-university-of-technology/openpi
```

## 六、模型推理（评估）

### 6.1 启动 Policy Server

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

# 替换 <step> 为实际的 checkpoint 步数
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero_plus_l1_flow_from_ckpt \
    --policy.dir=checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/<step>
```

### 6.2 运行评估 Client

Server 启动后，在另一个终端运行：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

export LIBERO_CONFIG_PATH=/tmp/libero && mkdir -p /tmp/libero
export PYTHONPATH="$(pwd)/third_party/LIBERO-plus:$(pwd)/packages/openpi-client/src:$(pwd)"

# 使用 Client 环境的 Python
examples/libero/.venv/bin/python examples/libero/main.py \
    --policy_model pi05_libero_plus_l1_flow_from_ckpt \
    --ckpt_path checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/<step>/params \
    --benchmark_name libero_spatial \
    --num_trials_per_task 1
```

### 6.3 可用的 benchmark

| benchmark_name | 说明 |
|---|---|
| `libero_spatial` | 空间关系任务 |
| `libero_object` | 物体操作任务 |
| `libero_goal` | 目标导向任务 |
| `libero_90` | 90 个任务子集 |
| `libero_10` | 10 个任务子集 |

> **注意**：LIBERO-Plus 评估中 `num_trials_per_task=1`（论文设定），标准 LIBERO 为 50。

## 七、关键代码位置

| 组件 | 位置 |
|------|------|
| 训练配置 | `src/openpi/training/config.py`（`pi05_libero_plus_l1_flow_from_ckpt`） |
| `l1_flow` 标志 | `src/openpi/models/pi0_config.py:33` |
| 预测目标切换 | `src/openpi/models/pi0.py:241` |
| L1 vs MSE 损失 | `src/openpi/models/pi0.py:256-258` |
| 2-step 推理 | `src/openpi/models/pi0.py:330-369` |
| freeze_filter | `src/openpi/training/nnx_utils.py:46-63` |

## 八、常见问题

### 8.1 `HF_LEROBOT_HOME` 未设置导致找不到数据集

```
FileNotFoundError: ... data/libero_plus_lerobot ...
```

解决：所有训练/norm_stats 命令都需加 `HF_LEROBOT_HOME="$(pwd)"`。

### 8.2 训练显存不足

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# 或多卡
--fsdp-devices <num_gpus>
```

### 8.3 norm_stats 路径不对

训练代码期望 norm_stats 在：
```
assets/pi05_libero_plus_l1_flow_from_ckpt/data/libero_plus_lerobot/norm_stats.json
```

若缺失，重新执行第 2.2 节的复制命令，或运行 `compute_norm_stats.py`。

### 8.4 weight_loader 找不到 checkpoint

相对路径 `checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/params` 需从 `openpi/` 目录运行，确保 `cd` 到正确目录。

### 8.5 推理连接错误

检查 Policy Server 是否在正确端口运行，以及 Client 的 server 地址配置。

### 8.6 FFmpeg 库缺失导致视频解码失败

```
libtorchcodec: libavutil.so.59: cannot open shared object file: No such file or directory
```

LIBERO-Plus 数据集以 mp4 视频存储，解码需要 FFmpeg。通过 `av`（PyAV）将 FFmpeg 打包进 `.venv`：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
uv pip install av
```

`av` wheel 中 FFmpeg 库文件名带 hash，torchcodec 按标准 soname 查找会失败，需创建符号链接：

```bash
cd .venv/lib/python3.11/site-packages/av.libs/
for f in *.so.*; do
    soname=$(echo "$f" | sed -E 's/-[0-9a-f]{8}(\.so\.[0-9]+).*/\1/')
    if [ "$soname" != "$f" ] && [ ! -e "$soname" ]; then
        ln -sf "$f" "$soname"
    fi
done
cd -
```

安装和符号链接创建后，所有训练命令需加 `LD_LIBRARY_PATH` 指向 av 的 FFmpeg 库：

```bash
LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
HF_LEROBOT_HOME="$(pwd)" \
...
```

验证库已存在：

```bash
find .venv -name "libavutil.so.59"
# 预期输出: .venv/lib/python3.11/site-packages/av.libs/libavutil.so.59
```

## 九、停止训练

```bash
# Ctrl+C

# 或强制停止
pkill -f "scripts/train.py"
```

## 十、Checkpoint 位置速查

| 路径 | 说明 |
|------|------|
| `checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/` | 初始化权重（已训练好的 L1 Flow） |
| `checkpoints/pi05_libero_plus_l1_flow_from_ckpt/<exp_name>/<step>/` | 微调后的 checkpoint |
| `assets/pi05_libero_plus_l1_flow_from_ckpt/data/libero_plus_lerobot/norm_stats.json` | 归一化统计量 |
| `data/libero_plus_lerobot -> research_lijinhao/pi0.5_deploy/libero_plus_lerobot` | 数据集软链接 |
