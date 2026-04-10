# 如何推送到 GitHub

## 仓库结构

```
onestep_pi/                          ← 外层 git 仓库（pi0.5acceleration）
├── solution.md
├── docs_f1flow/
├── L1_FLOW_TRAINING_COMMANDS-*.md
└── openpi/                          ← 内层 git 仓库（openpi），非 submodule
    ├── .git/
    └── src/openpi/training/config.py
```

`openpi/` 目录本身是一个独立的 git 仓库，但外层仓库**直接追踪**其中的文件（不是 submodule）。
这是因为外层仓库在 `openpi/.git` 存在的情况下仍然可以 `git add` 其中的文件。

## 推送流程

所有操作在 `onestep_pi/` 目录下执行：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi

# 1. 查看改动
git status

# 2. 按需 add 文件（精确指定，避免误提交无关文件）
git add docs_f1flow/03_training.md \
        solution.md \
        L1_FLOW_TRAINING_COMMANDS-liberoplus.md \
        openpi/src/openpi/training/config.py \
        openpi/L1_FLOW_TRAINING_COMMANDS-liberoplus.md

# 3. 提交
git commit -m "描述本次改动"

# 4. 推送
git push origin main
```

## remote 地址

```
origin  https://github.com/yuxuancongbot-maker/pi0.5acceleration.git
```

## 注意事项

- **精确 add 文件**，不要用 `git add .` 或 `git add openpi/`，避免把 `gpu.py`、`assets.zip`、`run_libero_plus_finetune.sh` 等无关文件带进去
- `openpi/third_party/LIBERO-plus` 是 openpi 内部的 submodule，不要 add
- `.mcp.json` 是本地工具配置，不要 add
- openpi 自己的 git 历史（`openpi/.git`）不受影响，两个仓库互相独立
