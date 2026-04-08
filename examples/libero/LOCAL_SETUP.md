# LIBERO 环境本地配置指南

本文档记录在本地（非 Docker）配置 LIBERO 环境的完整步骤。

## 环境要求

- Python 3.10.12
- uv (Python 包管理器)
- 系统依赖库

## 系统依赖安装

```bash
# 安装 OpenGL 相关库
apt-get update
apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3.10-dev \
    python3-dev
```

## 创建虚拟环境

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi

# 创建 Python 3.10 虚拟环境
uv venv --python 3.10 examples/libero/.venv
source examples/libero/.venv/bin/activate
```

## 安装 PyTorch (CUDA 11.3)

```bash
# 先安装 PyTorch（需要指定 CUDA 版本）
uv pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

# 锁定 numpy 版本（兼容性问题）
uv pip install 'numpy<2.0.0'
```

## 安装核心依赖

```bash
uv pip install imageio[ffmpeg] tqdm tyro PyYaml opencv-python==4.6.0.66 robosuite==1.4.1 matplotlib==3.5.3
```

## 设置 robosuite 宏文件

```bash
python /path/to/.venv/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
```

## 安装 bddl 和其他依赖

```bash
uv pip install bddl==1.0.1 future easydict cloudpickle gym
```

## 安装 openpi-client 和 libero

```bash
# 安装 openpi-client
uv pip install -e packages/openpi-client

# 安装 libero
uv pip install -e third_party/libero
```

## 配置 LIBERO 环境变量

```bash
# 创建配置目录
mkdir -p /tmp/libero

# 写入配置
cat > /tmp/libero/config.yaml << 'EOF'
benchmark_root: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/libero/libero/libero
bddl_files: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/libero/libero/libero/bddl_files
init_states: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/libero/libero/libero/init_files
datasets: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/libero/libero/datasets
assets: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/third_party/libero/libero/libero/assets
EOF
```

## 快速启动命令汇总

在每次新终端会话中，需要设置以下环境变量：

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/third_party/libero:$PWD/packages/openpi-client/src
export LIBERO_CONFIG_PATH=/tmp/libero
export PYOPENGL_PLATFORM=egl
```

## 运行

### 终端 1: 运行仿真客户端

```bash
python examples/libero/main.py
```

可选参数：
- `--args.task-suite-name libero_spatial` - Libero Spatial 任务集（默认）
- `--args.task-suite-name libero_object` - Libero Object 任务集
- `--args.task-suite-name libero_goal` - Libero Goal 任务集
- `--args.task-suite-name libero_10` - Libero 10 任务集
- `--args.num-trials-per-task 50` - 每个任务的试验次数

### 终端 2: 运行策略服务器

```bash
uv run scripts/serve_policy.py --env LIBERO
```

可选参数：
- `--policy.config pi05_libero` - 使用 pi05_libero 配置
- `--policy.dir ./my_custom_checkpoint` - 自定义检查点目录

## 已知问题与解决方案

### 1. llvmlite 安装失败

**问题**: `RuntimeError: Cannot install on Python version 3.10.12; only versions >=3.6,<3.10 are supported`

**解决**: 先安装最新版本的 llvmlite，再安装其他依赖：
```bash
uv pip install llvmlite
uv pip install numba
```

### 2. evdev 编译失败

**问题**: `fatal error: Python.h: No such file or directory`

**解决**: 安装 Python 开发头文件：
```bash
apt-get install -y python3.10-dev python3-dev
```

### 3. libGL.so.1 无法找到

**问题**: `ImportError: libGL.so.1: cannot open shared object file`

**解决**: 安装 OpenGL 库：
```bash
apt-get install -y libgl1-mesa-glx libgl1-mesa-dri libegl1
```

### 4. Gym 版本警告

**警告**: `Gym has been unmaintained since 2022 and does not support NumPy 2.0`

**说明**: 这是警告信息，不影响功能。LIBERO 仍在使用旧版 gym。

## 验证安装

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/third_party/libero:$PWD/packages/openpi-client/src
export LIBERO_CONFIG_PATH=/tmp/libero
export PYOPENGL_PLATFORM=egl

python -c "import robosuite; import mujoco; import libero; print('All imports successful!')"
```

## 使用 EGL 加速（无头渲染）

如果遇到 Mesa GL 或 GLX 错误，可以设置使用 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 参考

- [LIBERO GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [robosuite 文档](https://robosuite.farama.org/)
- [Mujoco 文档](https://mujoco.readthedocs.io/)
