# LIBERO-Plus 训练与评估

## 背景

LIBERO-Plus（arXiv:2510.13626）是 LIBERO 的扰动增强版本，包含 7 个扰动维度：
1. Objects Layout - 目标物体位移
2. Camera Viewpoints - 相机位置、朝向、视野变化
3. Robot Initial States - 机械臂初始姿态变化
4. Language Instructions - LLM 重写的指令
5. Light Conditions - 光照强度、方向、颜色、阴影
6. Background Textures - 场景和表面外观变化
7. Sensor Noise - 图像退化

**观测/动作格式与标准 LIBERO 完全一致**（8D state, 7D action, 相同图像 key），因此 `LiberoInputs`/`LiberoOutputs` transforms 无需修改。

## 训练配置

### 配置 1：`pi05_libero_plus_l1_flow`

从 `pi05_libero`（标准 LIBERO 微调过的 pi0.5）初始化。

**位置**：`src/openpi/training/config.py`

**与 `pi05_libero_l1_flow` 的唯一区别**：`repo_id` 从 `physical-intelligence/libero` 改为 `Sylvest/libero_plus_lerobot`。

### 配置 2：`pi05_libero_plus_l1_flow_from_ckpt`（推荐）

从本地 `pi05_libero_l1_flow` checkpoint 初始化（已在标准 LIBERO 上用 L1 Flow 训练过）。

**优势**：
- 权重已在标准 LIBERO 上用 L1 Flow 训练过，action expert 已适配 LIBERO 的动作空间
- 相比从 `pi05_libero`（未用 L1 Flow）初始化，收敛更快
- 可以用更少的训练步数达到更好的性能

**weight_loader**：
```python
weight_loader=weight_loaders.CheckpointWeightLoader(
    "checkpoints/pi05_libero_l1_flow/l1_flow_changed/29999/params"
)
```

## 训练流程

### 方案 A：从 pi05_libero 初始化

```bash
# ===== Server 环境 (openpi/.venv) =====
cd openpi
source .venv/bin/activate

# 1. 验证数据集可用（首次自动从 HuggingFace 下载）
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('Sylvest/libero_plus_lerobot')
print(f'Size: {len(ds)}, Keys: {list(ds[0].keys())}')
"

# 2. 计算 norm stats（必须，分布可能不同）
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow

# 3. 训练（JAX）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_plus_l1_flow \
    --exp-name=libero_plus_l1flow --overwrite

# 3. 训练（PyTorch）
uv run scripts/train_pytorch.py --config-name pi05_libero_plus_l1_flow
```

### 方案 B：从本地 checkpoint 微调（推荐）

```bash
# ===== Server 环境 =====
cd openpi
source .venv/bin/activate

# 1. 计算 LIBERO-plus 的 norm stats
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow_from_ckpt

# 2. 从 checkpoint 29999 开始微调
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 --overwrite
```

## 评估流程

```bash
# ===== Client 环境 (examples/libero/.venv) =====
cd openpi
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/libero && mkdir -p /tmp/libero

# 评估（LIBERO-Plus 环境，num_trials_per_task=1）
python examples/libero/main.py \
    --policy_model pi05_libero_plus_l1_flow_from_ckpt \
    --ckpt_path checkpoints/libero_plus_from_ckpt29999/<step>/params \
    --benchmark_name libero_spatial \
    --num_trials_per_task 1
```

## 可用的 benchmark

- `libero_spatial` - 空间关系任务（2402 tasks）
- `libero_object` - 物体操作任务
- `libero_goal` - 目标导向任务
- `libero_90` - 90 个任务子集
- `libero_10` - 10 个任务子集
- `libero_100` - 100 个任务子集
- `libero_mix` - 混合任务

## 注意事项

- **网络不通时**: 手动下载 `Sylvest/libero_plus_lerobot` 到本地后，将 `repo_id` 改为本地路径
- **混合训练**: 如需同时用标准 LIBERO + LIBERO-Plus 数据，需先合并为一个 LeRobot 数据集或修改 data_loader 支持多数据源
- **评估时 num_trials_per_task**: LIBERO-plus 论文中从 50 改为 1（加速评估）
