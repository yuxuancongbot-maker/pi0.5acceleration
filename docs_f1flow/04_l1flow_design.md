# L1 Flow 集成到 π0.5 的技术方案

## 整体思路

π0.5 的 action expert 本质上是一个 flow matching 模块，L1 Flow 不改变网络结构，只改变：
- **预测目标**：velocity → sample
- **损失函数**：MSE → L1
- **推理调度**：多步 ODE → 2步

因此可以直接加载原始权重后做轻量 fine-tune。

---

## 第一步：理解 π0.5 的 Action Expert 结构

π0.5（openpi 仓库）的 action expert 是一个条件 flow matching 头，核心调用链如下：

```
观测特征 (image + language token) 
    → 主干 PaliGemma/VLM (参数路径含 "llm")
    → action_expert (参数路径含 "llm_1")
        → 输入: noisy_action(xt), timestep(t), conditioning
        → 输出: predicted_velocity (x1 - x0)  ← 我们要改这里
    → 多步 ODE 积分 → 最终动作
```

> **关键命名约定**：openpi 中 VLM backbone 的参数路径含 `llm`，action expert 的参数路径含 `llm_1`（带后缀 `_1`）。这是冻结/训练分离的基础。

---

## 第二步：修改预测目标（核心改动）

原始 π0.5 flow matching 训练目标是速度预测：

```python
# 原始 pi0.5: 预测速度
target = noise - actions   # velocity (在 openpi 中的实际写法)
loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # MSE
```

改为 L1 Flow 的样本预测：

```python
# L1 Flow: 直接预测 x1 样本
target = actions   # terminal sample (x1)
loss = jnp.mean(jnp.abs(v_t - u_t), axis=-1)  # L1 / MAE
```

**实际代码位置**：
- JAX: `src/openpi/models/pi0.py:241,256-258`
- PyTorch: `src/openpi/models_pytorch/pi0_pytorch.py:371,417-419`

通过 `self.l1_flow` 标志自动切换，无需手动修改代码。

---

## 第三步：完整实现（已在 openpi 中完成）

### 3.1 修改 timestep 采样分布

混合 LogisticNormal + Uniform 分布，强调中间 timestep，同时保证边界概率非零。

**实际代码位置**：
- JAX: `src/openpi/models/pi0.py:19-49` (`MixedTimestepSampler`)
- PyTorch: `src/openpi/models_pytorch/pi0_pytorch.py:52-81`

```python
class MixedTimestepSampler:
    """
    混合 LogisticNormal + Uniform 分布
    alpha=0.99: 99% 概率使用 LogisticNormal（集中在 0.3~0.7），1% 使用 Uniform
    """
    def __init__(self, alpha=0.99, device='cuda'):
        self.alpha = alpha
        self.device = device
    
    def sample(self, batch_size):
        t = torch.zeros(batch_size, device=self.device)
        use_logistic = torch.rand(batch_size) < self.alpha
        
        # LogisticNormal: sigmoid(N(0,1)) ≈ 集中在 0.3~0.7
        n_logistic = use_logistic.sum().item()
        if n_logistic > 0:
            x = torch.randn(n_logistic)
            t[use_logistic] = torch.sigmoid(x).to(self.device)
        
        # Uniform 部分
        n_uniform = (~use_logistic).sum().item()
        if n_uniform > 0:
            t[~use_logistic] = torch.rand(n_uniform).to(self.device)
        
        return t
```

### 3.2 训练目标切换逻辑

在模型前向中，通过 `l1_flow` 标志切换预测目标和损失函数：

```python
# 预测目标
u_t = actions if self.l1_flow else noise - actions

# 损失函数
if self.l1_flow:
    loss = jnp.mean(jnp.abs(v_t - u_t), axis=-1)   # L1
else:
    loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # MSE
```

### 3.3 修改推理调度（2-step inference, NFE=2）

**实际代码位置**：
- JAX: `src/openpi/models/pi0.py:330-369` (`_l1_flow_sample`)
- PyTorch: `src/openpi/models_pytorch/pi0_pytorch.py:471-496`

```python
@torch.no_grad()
def _l1_flow_sample(self, observations, action_shape):
    """
    2-step inference (NFE=2):
    Step 1: 从纯噪声 x0 出发，模型预测 x1，用 Euler 步进到中点 t=0.5
    Step 2: 从中点直接预测最终 x1
    """
    conditioning = self.encode_observations(observations)
    
    # Step 1: 纯噪声 → 中点
    x0 = torch.randn(action_shape, device=self.device)
    t0 = torch.zeros(action_shape[0], device=self.device)
    
    x1_pred_coarse = self.denoise(x0, timesteps=t0, conditioning=conditioning)
    v_t0 = x1_pred_coarse - x0  # 从 sample prediction 还原速度
    x_mid = x0 + 0.5 * v_t0     # Euler step 到 t=0.5
    
    # Step 2: 中点 → 最终预测
    t_mid = torch.full((action_shape[0],), 0.5, device=self.device)
    x1_final = self.denoise(x_mid, timesteps=t_mid, conditioning=conditioning)
    
    return x1_final
```

---

## 第四步：参数冻结策略（重要修正）

目标：**冻结 VLM backbone，只训练 action expert**。

openpi 中参数路径的命名约定：
- VLM (PaliGemma) 参数路径包含 `llm`（不含 `_1`）
- Action Expert 参数路径包含 `llm_1`（带 `_1` 后缀）

### 正确的 freeze_filter

```python
# 正确：冻结除 action expert (llm_1) 以外的所有参数
freeze_filter = nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))
```

逻辑推导：
- `PathRegex(".*llm.*_1.*")` 匹配 action expert 参数
- `Not(...)` 取反 → 匹配 action expert 以外的所有参数
- `freeze_filter` 定义冻结哪些 → action expert 以外全部冻结
- `trainable_filter = Not(freeze_filter)` → 只有 action expert 可训练

### 错误写法（需避免）

```python
# 错误！这会让 VLM 和 action expert 都可训练
freeze_filter = nnx.All(nnx.Not(nnx_utils.PathRegex(".*llm.*")))
```

原因：`".*llm.*"` 同时匹配 VLM (`llm`) 和 action expert (`llm_1`)，`Not` 之后冻结的是不含 "llm" 的参数，导致两者都不被冻结。

### 参考

`get_freeze_filter()` 方法 (`pi0_config.py:90-119`) 使用相同的 regex 模式 `".*llm.*_1.*"` 作为 `action_expert_params_filter`，这是 openpi 项目的标准做法。

---

## 第五步：配置总览

完整的 `pi05_libero_l1_flow` 训练配置（`src/openpi/training/config.py:769-795`）：

```python
TrainConfig(
    name="pi05_libero_l1_flow",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        l1_flow=True,  # 启用 L1 Flow
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
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
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=30_000,
    # 冻结除 action expert (llm_1) 以外的所有参数
    freeze_filter=nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")),
)
```

---

## 方案总结

| 阶段 | 改动 | 预期效果 |
|------|------|----------|
| 仅换推理（无 fine-tune） | 2-step 替代多步 ODE | 推理速度提升 5-10×，性能略降 |
| Fine-tune action expert | L1 Flow 目标 + 冻结 VLM | 最佳性价比，仅训练 action expert 参数 |
| 完整训练 | 从头以 L1 Flow 目标训练 | 最佳性能 + 2 NFE 效率 |

关键点：
1. **网络权重形状完全兼容**，预训练权重可直接加载
2. 通过 `l1_flow=True` 一个标志切换训练目标、损失函数和推理策略
3. 冻结 VLM 使用 `".*llm.*_1.*"` regex 精确匹配 action expert，而非 `".*llm.*"`
