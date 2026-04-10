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

## 关键环境变量

训练配置中 `repo_id="data/libero_plus_lerobot"`，lerobot 查找路径为 `HF_LEROBOT_HOME/data/libero_plus_lerobot`。**必须**设置：

```bash
export HF_LEROBOT_HOME="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
# 这样 lerobot 会在 openpi/data/libero_plus_lerobot 下查找数据
```

LIBERO-Plus 数据集以 mp4 视频存储，需要 FFmpeg 解码。通过安装 `av` 将 FFmpeg 打包进 `.venv`（**首次使用前执行一次**）：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
uv pip install av
```

`av` wheel 中 FFmpeg 库文件名带 hash，torchcodec 按标准 soname 查找会失败，需创建符号链接（**仅需执行一次**）：

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

所有训练命令需通过 `LD_LIBRARY_PATH` 让 torchcodec 找到 FFmpeg 库：

```bash
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH"
```

## 训练流程

所有命令均在 `openpi/` 目录下执行（`uv run` 自动使用 `.venv`）。

### 方案 A：从 pi05_libero 初始化

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

# 1. 验证数据集已下载
HF_LEROBOT_HOME="$OPENPI_DIR" uv run python3 -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('data/libero_plus_lerobot')
print(f'total_episodes: {meta.total_episodes}')
"

# 2. 计算 norm stats（必须，LIBERO-plus 分布与标准 LIBERO 不同）
HF_LEROBOT_HOME="$OPENPI_DIR" \
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow

# 3. 训练（JAX）
HF_LEROBOT_HOME="$OPENPI_DIR" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
LD_LIBRARY_PATH="$OPENPI_DIR/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
uv run scripts/train.py pi05_libero_plus_l1_flow \
    --exp-name=libero_plus_l1flow --overwrite
```

### 方案 B：从本地 checkpoint 微调（推荐）

从 `pi05_libero_l1_flow` step 29999 初始化，action expert 已在标准 LIBERO 上适配 L1 Flow。

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

# 1. 计算 LIBERO-plus 的 norm stats
HF_LEROBOT_HOME="$OPENPI_DIR" \
uv run scripts/compute_norm_stats.py --config-name pi05_libero_plus_l1_flow_from_ckpt

# norm stats 保存位置：assets/pi05_libero_plus_l1_flow_from_ckpt/data/libero_plus_lerobot/norm_stats.json

# 2. 从 checkpoint 29999 开始微调
HF_LEROBOT_HOME="$OPENPI_DIR" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
LD_LIBRARY_PATH="$OPENPI_DIR/.venv/lib/python3.11/site-packages/av.libs:$LD_LIBRARY_PATH" \
uv run scripts/train.py pi05_libero_plus_l1_flow_from_ckpt \
    --exp-name=libero_plus_from_ckpt29999 --overwrite

# checkpoint 保存位置：checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/<step>/
```

## 评估流程

```bash
# ===== Client 环境 (examples/libero/.venv) =====
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
cd "$OPENPI_DIR"

export LIBERO_CONFIG_PATH=/tmp/libero && mkdir -p /tmp/libero
export PYTHONPATH="$OPENPI_DIR/third_party/LIBERO-plus:$OPENPI_DIR/packages/openpi-client/src:$OPENPI_DIR"

# 评估（LIBERO-Plus 环境，num_trials_per_task=1）
examples/libero/.venv/bin/python examples/libero/main.py \
    --policy_model pi05_libero_plus_l1_flow_from_ckpt \
    --ckpt_path checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/<step>/params \
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

- **FFmpeg 库缺失**: LIBERO-Plus 用视频存储，必须先 `uv pip install av` 并创建 soname 符号链接，训练命令中设置 `LD_LIBRARY_PATH`（见上方关键环境变量章节），否则报 `libavutil.so.59: cannot open shared object file`
- **RepackTransform 键名方向**: `RepackTransform({目标键: 源键})`，目标键必须与 `LiberoInputs` 期望的格式一致（`observation/image`），源键来自数据集（`observation.images.front`）。写反或目标键格式错误分别导致 `KeyError: 'actions'` 或 `KeyError: 'observation/image'`
- **网络不通时**: 手动下载 `Sylvest/libero_plus_lerobot` 到本地后，将 `repo_id` 改为本地路径
- **混合训练**: 如需同时用标准 LIBERO + LIBERO-Plus 数据，需先合并为一个 LeRobot 数据集或修改 data_loader 支持多数据源
- **评估时 num_trials_per_task**: LIBERO-plus 论文中从 50 改为 1（加速评估）
