# LIBERO-Plus 数据集下载

## 数据集存储位置

训练配置 `pi05_libero_plus_l1_flow_from_ckpt` 使用 `repo_id="data/libero_plus_lerobot"`，lerobot 会在 `HF_LEROBOT_HOME/data/libero_plus_lerobot` 下查找数据。

**约定**：将 `HF_LEROBOT_HOME` 设为 `openpi/` 目录，数据存放在：

```
openpi/data/libero_plus_lerobot/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   └── episodes.jsonl
├── data/
│   ├── chunk-000/      # parquet 文件（state, action, timestamp...）
│   ├── chunk-001/
│   └── ...             # 共 15 个 chunks，14347 个 parquet 文件
└── videos/
    ├── chunk-000/      # mp4 视频（front + wrist 双相机）
    └── ...             # 共 28694 个视频
```

## 下载数据集

数据集共 43047 个文件（含 28694 个视频），使用 `huggingface-cli` + `hf_transfer` 加速，外层套重试循环处理网络中断：

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
TARGET_DIR="$OPENPI_DIR/data/libero_plus_lerobot"

cd "$OPENPI_DIR"

# 后台下载（自动重试 20 次，失败后等 60 秒）
max_retries=20; attempt=0
while [ $attempt -lt $max_retries ]; do
    attempt=$((attempt + 1))
    echo "[$(date '+%H:%M:%S')] Attempt $attempt/$max_retries"

    HF_HUB_ENABLE_HF_TRANSFER=1 \
    uv run huggingface-cli download \
        Sylvest/libero_plus_lerobot \
        --repo-type dataset \
        --local-dir "$TARGET_DIR" \
        --resume-download \
    && echo "下载完成！" && break

    echo "失败，等待 60s 后重试..."
    sleep 60
done
```

**说明**：
- `HF_HUB_ENABLE_HF_TRANSFER=1`：启用 Rust 实现的 hf_transfer，更快更稳定
- `--resume-download`：支持断点续传
- 数据集使用 HuggingFace **Xet Storage**（新一代 LFS），每个文件单独下载，中断后可续传
- 不需要设置 `HF_ENDPOINT` 镜像（镜像不代理 Xet Storage 的实际文件）

## 检查下载进度

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
DATA_DIR="$OPENPI_DIR/data/libero_plus_lerobot"

# 已下载大小
du -sh "$DATA_DIR"

# parquet 文件数（完整应为 14347）
find "$DATA_DIR/data" -name "*.parquet" | wc -l

# 视频文件数（完整应为 28694）
find "$DATA_DIR/videos" -name "*.mp4" 2>/dev/null | wc -l

# 用 lerobot 验证元数据
cd "$OPENPI_DIR"
HF_LEROBOT_HOME="$OPENPI_DIR" uv run python3 -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('data/libero_plus_lerobot')
print(f'total_episodes: {meta.total_episodes}')   # 预期: 14347
print(f'total_frames:   {meta.total_frames}')     # 预期: 2238036
print(f'root:           {meta.root}')
"
```

## 数据集信息

| 字段 | 值 |
|------|------|
| repo_id | `Sylvest/libero_plus_lerobot` |
| codebase_version | v2.1 |
| total_episodes | 14347 |
| total_frames | 2238036 |
| total_tasks | 40 |
| fps | 20 |
| observation.state shape | [8] |
| action shape | [7] |
| 图像 | front (256x256) + wrist (256x256)，av1 编码 |

## 使用本地已有数据集（推荐，免下载）

服务器上 `research_lijinhao/pi0.5_deploy/libero_plus_lerobot` 已有完整数据集（19GB），可直接软链接，无需重新下载：

```bash
OPENPI_DIR="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi"
SRC="/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_lijinhao/pi0.5_deploy/libero_plus_lerobot"

# 1. 软链接数据集（不占额外磁盘空间）
ln -s "$SRC" "$OPENPI_DIR/data/libero_plus_lerobot"

# 2. 复制 norm_stats（源目录里已有现成的，无需重新计算）
ASSETS_DIR="$OPENPI_DIR/assets/pi05_libero_plus_l1_flow_from_ckpt/data/libero_plus_lerobot"
mkdir -p "$ASSETS_DIR"
cp "$SRC/norm_stats.json" "$ASSETS_DIR/norm_stats.json"

# 3. 验证
HF_LEROBOT_HOME="$OPENPI_DIR" uv run python3 -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata('data/libero_plus_lerobot')
print(f'total_episodes: {meta.total_episodes}')   # 预期: 14347
print(f'total_frames:   {meta.total_frames}')     # 预期: 2238036
"
```

完成后可直接跳过 `compute_norm_stats.py`，直接开始训练。
