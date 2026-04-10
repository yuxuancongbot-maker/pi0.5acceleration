# LIBERO-Plus 数据集下载与 lerobot 补丁

## 问题：数据集缺少 version tag

`Sylvest/libero_plus_lerobot` 数据集在 HuggingFace 上没有 version tag，导致 lerobot 加载时报错：
```
RevisionNotFoundError: Your dataset must be tagged with a codebase version.
```

## 解决方案：修改 lerobot 的 `get_safe_version` 函数

编辑 `.venv/lib/python3.11/site-packages/lerobot/common/datasets/utils.py`，修改 `get_safe_version` 函数（约 319 行）：

```python
def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:
    """
    Returns the version if available on repo or the latest compatible one.
    Otherwise, will throw a `CompatibilityError`.
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        # PATCH: Return None instead of raising error when no tags exist
        # This allows loading datasets without version tags (e.g., Sylvest/libero_plus_lerobot)
        logging.warning(f"No version tags found for {repo_id}, using main branch")
        return None  # 改这里：原来是 raise RevisionNotFoundError(...)

    # ... 其余代码不变
```

## 加速数据集下载

LIBERO-plus 数据集很大（43048 个文件，约 14347 episodes），使用国内镜像 + 多线程下载：

```bash
# 1. 设置 HuggingFace 镜像（加速）
export HF_ENDPOINT=https://hf-mirror.com

# 2. 使用 huggingface-cli 多线程下载（比 LeRobotDataset 自动下载快很多）
source .venv/bin/activate
huggingface-cli download Sylvest/libero_plus_lerobot \
    --repo-type dataset \
    --local-dir /root/.cache/huggingface/lerobot/Sylvest/libero_plus_lerobot \
    --local-dir-use-symlinks False

# 3. 检查下载进度
find /root/.cache/huggingface/lerobot/Sylvest/libero_plus_lerobot/data -name "*.parquet" | wc -l
# 预期：14347 个 parquet 文件
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
