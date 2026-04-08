# LIBERO-plus 环境配置指南

本文档记录在 openpi 项目中配置 LIBERO-plus 环境的完整步骤。

## 概述

**LIBERO-plus** 是 LIBERO 的扩展基准，用于深度分析视觉-语言-动作（VLA）模型的鲁棒性。
- GitHub: https://github.com/sylvestf/LIBERO-plus
- 论文: arXiv:2510.13626
- 7 个扰动维度：Camera, Robot, Language, Light, Background, Noise, Layout

## 前提条件

- Python 3.10
- 已配置好的 `examples/libero/.venv` 虚拟环境（参见 [LOCAL_SETUP.md](LOCAL_SETUP.md)）
- 系统依赖库（见 LOCAL_SETUP.md）

## 安装步骤

### 1. 克隆 LIBERO-plus 仓库

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party
git clone https://github.com/sylvestf/LIBERO-plus.git
```

### 2. 下载 assets 资产

使用 Python API 下载（避免 huggingface-cli 命令缺失问题）：

```bash
source examples/libero/.venv/bin/activate
mkdir -p third_party/libero_plus
cd third_party/libero_plus

python << 'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Sylvest/LIBERO-plus',
    filename='assets.zip',
    local_dir='.',
    repo_type='dataset'
)
EOF
```

**下载完成后**: `third_party/libero_plus/assets.zip` (约 6GB)

### 3. 解压 assets

```bash
cd third_party/libero_plus

# 创建目标目录
mkdir -p ../LIBERO-plus/libero/libero

# 解压（注意：zip 内部路径很深，必须完整提取）
unzip -o assets.zip "inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/*" -d .

# 移动到正确位置
mv inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets ../LIBERO-plus/libero/libero/

# 清理临时文件
rm -rf inspire assets.zip
```

**最终 assets 位置**: `third_party/LIBERO-plus/libero/libero/assets/`

### 4. 安装 LIBERO-plus Python 包

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
source examples/libero/.venv/bin/activate

# 卸载旧版 libero
pip uninstall libero -y || true

# 安装 LIBERO-plus
cd third_party/LIBERO-plus
uv pip install -e .
```

### 5. 安装额外依赖

```bash
# 系统依赖
apt install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

# Python 依赖
source examples/libero/.venv/bin/activate
uv pip install wand scikit-image
```

### 6. 配置 LIBERO 环境变量

```bash
mkdir -p /tmp/libero

cat > /tmp/libero/config.yaml << 'EOF'
benchmark_root: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/LIBERO-plus/libero/libero
bddl_files: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/LIBERO-plus/libero/libero/bddl_files
init_states: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/LIBERO-plus/libero/libero/init_files
datasets: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/LIBERO-plus/libero/datasets
assets: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/LIBERO-plus/libero/libero/assets
EOF
```

## 快速启动命令

在每次新终端会话中执行：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/LIBERO-plus:$PWD/packages/openpi-client/src:$PWD
export LIBERO_CONFIG_PATH=/tmp/libero
export PYOPENGL_PLATFORM=egl
```

## 验证安装

```bash
python -c "
from libero.libero import benchmark
bench_dict = benchmark.get_benchmark_dict()
print('Available benchmarks:')
for name in bench_dict.keys():
    print(f'  - {name}')
"
```

**输出**:
```
Available benchmarks:
  - libero_spatial
  - libero_object
  - libero_goal
  - libero_90
  - libero_10
  - libero_100
  - libero_mix
```

## 可用 Benchmark

| Benchmark | 说明 |
|-----------|------|
| libero_spatial | 标准空间关系任务（30个） |
| libero_object | 标准物体关系任务（30个） |
| libero_goal | 标准目标任务（30个） |
| libero_90 | 预训练任务（90个） |
| libero_10 | 测试任务（10个） |
| libero_100 | 100个任务 |
| libero_mix | LIBERO-plus 混合任务 |

## 评估

LIBERO-plus 评估与标准 LIBERO 类似，主要区别：

1. **num_trials_per_task 设为 1**（而非默认的 50）
2. 任务分类信息在 `third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json`

## 训练数据集下载

如需训练数据，从 HuggingFace 下载：

| 数据类型 | HuggingFace 链接 |
|---------|------------------|
| RLDS 格式 | https://huggingface.co/datasets/Sylvest/libero_plus_rlds |
| LeRobot 格式 | https://huggingface.co/datasets/Sylvest/libero_plus_lerobot |
| 各 suite 数据 | https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite |

```bash
# 下载示例
python << 'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Sylvest/libero_plus_rlds', filename='...', local_dir='./data', repo_type='dataset')
EOF
```

## 目录结构

```
openpi/
├── third_party/
│   ├── LIBERO-plus/              # LIBERO-plus 代码
│   │   └── libero/libero/
│   │       ├── assets/           # 3D 模型、纹理等资源 (15GB)
│   │       ├── bddl_files/      # BDDL 任务定义
│   │       ├── init_files/      # 初始化状态
│   │       └── benchmark/        # benchmark 定义
│   │           └── task_classification.json
│   └── libero/                   # 原版 LIBERO（已不常用）
└── examples/libero/
    └── .venv/                    # Python 虚拟环境
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'libero'`

确保 `PYTHONPATH` 包含 `third_party/LIBERO-plus`：
```bash
export PYTHONPATH=$PWD/third_party/LIBERO-plus:$PWD/packages/openpi-client/src:$PWD
```

### 2. `Do you want to specify a custom path...` 卡住

必须设置 `LIBERO_CONFIG_PATH` 环境变量：
```bash
export LIBERO_CONFIG_PATH=/tmp/libero
```

### 3. `ImportError: libGL.so.1` 或 EGL 错误

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 4. assets 下载速度慢

HuggingFace 有速率限制，可设置 `HF_TOKEN` 环境变量加速：
```bash
export HF_TOKEN=your_token_here
```

## 参考

- [LIBERO-plus GitHub](https://github.com/sylvestf/LIBERO-plus)
- [LIBERO-plus HuggingFace Assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus)
- [LIBERO-plus 论文](https://arxiv.org/pdf/2510.13626)
- [原版 LIBERO](../libero/LOCAL_SETUP.md)
