# LIBERO-Plus 环境配置

## 环境架构

openpi 项目使用 **两个独立的 uv 虚拟环境**：

| 环境 | 路径 | 用途 | 包管理 |
|------|------|------|--------|
| **Server** | `openpi/.venv` | 训练环境（JAX/PyTorch，运行 `scripts/train.py`） | `uv pip install` |
| **Client** | `openpi/examples/libero/.venv` | 评估环境（LIBERO-plus，运行 `examples/libero/main.py`） | `uv pip install` |

**关键点**：
- 两个环境互不干扰，LIBERO-plus 只需安装在 **Client 环境**
- 使用 `uv pip` 而非 `python -m pip` 进行包管理
- Server 环境用于训练，Client 环境用于评估

## 目录结构

```
onestep_pi/openpi/
├── .venv/                          # Server 环境 (uv 创建)
├── third_party/
│   └── LIBERO-plus/                # git clone https://github.com/sylvestf/LIBERO-plus.git
│       └── libero/libero/
│           ├── assets/             # 从 HuggingFace 下载解压
│           ├── benchmark/
│           ├── envs/
│           └── bddl_files/
└── examples/libero/
    └── .venv/                      # Client 环境 (uv 创建)
```

## 安装步骤

```bash
# 1. 克隆仓库（在 openpi 目录下）
cd openpi/third_party
git clone https://github.com/sylvestf/LIBERO-plus.git

# 2. 下载并解压 assets（HuggingFace）
#    注意：需在 Server 环境中执行（huggingface_hub 在 Server 环境中）
cd openpi
source .venv/bin/activate
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Sylvest/LIBERO-plus', filename='assets.zip', local_dir='.', repo_type='dataset')
"
unzip -o assets.zip
cp -r inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/* \
    third_party/LIBERO-plus/libero/libero/assets/
rm -rf inspire/  # 清理解压的临时目录

# 3. 修复 LIBERO-plus 的 setup.py 问题
#    问题：find_packages() 找不到 libero.libero 包（缺少 libero/__init__.py）
#    解决：创建 libero/__init__.py 使其成为合法的 Python 包
echo "# This file makes the libero/ directory a proper Python package" > third_party/LIBERO-plus/libero/__init__.py

# 4. 在 Client 环境中安装 LIBERO-plus
source examples/libero/.venv/bin/activate
cd third_party/LIBERO-plus && uv pip install .

# 5. 安装额外依赖（wand、scikit-image 等）
uv pip install wand scikit-image
# 系统依赖（需 root 权限）
sudo apt install libexpat1 libfontconfig1-dev libmagickwand-dev
```

## 必须设置的环境变量

```bash
# LIBERO_CONFIG_PATH: 不设置会导致 import 时卡在交互式输入
export LIBERO_CONFIG_PATH=/tmp/libero
mkdir -p /tmp/libero

# PYTHONPATH: 确保 Client 环境能找到 LIBERO-plus
export PYTHONPATH=$PWD/third_party/LIBERO-plus:$PWD/packages/openpi-client/src:$PWD
```

## 常见问题排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `EOFError: EOF when reading a line` | `LIBERO_CONFIG_PATH` 未设置 | `export LIBERO_CONFIG_PATH=/tmp/libero && mkdir -p /tmp/libero` |
| `No module named 'wand'` | Client 环境缺少 wand | `uv pip install wand` + 安装 libmagickwand-dev |
| `No module named 'libero.libero'` | `find_packages()` 找不到包（缺 `libero/__init__.py`） | 创建 `third_party/LIBERO-plus/libero/__init__.py` 后重新安装 |
| `No module named pip` | venv 中 pip 丢失 | `python -m ensurepip --upgrade`（uv 环境一般不需要） |
| robosuite WARNING: No private macro file | 非致命，可忽略 | 可选运行 `python robosuite/scripts/setup_macros.py` |
| assets 解压后结构不对 | zip 内路径深度嵌套 | 不能用 `unzip -j`，需保留目录结构后手动 cp |

## 验证安装（Client 环境）

```bash
cd openpi
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=/tmp/libero
python -c "
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
bm = get_benchmark('libero_spatial')()
print(f'libero_spatial tasks: {bm.n_tasks}')  # 预期: 2402
"
```
