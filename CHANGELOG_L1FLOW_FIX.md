# L1 Flow Code Fix — 2026-04-07

## Background

We have a training run in progress that uses the **old (pre-fix) code**. Once that model finishes training, we need to **revert to the old code** to test its performance. The new training with the fixed code will also proceed in parallel. Both versions must be preserved.

---

## Changes Made

### 3 files modified, 3 issues fixed.

---

### Fix 1: Train-Inference Time Convention Mismatch

**Files**: `src/openpi/models_pytorch/pi0_pytorch.py` (line ~368), `src/openpi/models/pi0.py` (line ~242)

**Problem**: Training used `x_t = t * noise + (1-t) * actions` (t=1 is noise), but inference used t=0 and t=0.5 following the original L1Flow convention (t=0 is noise). The model learned "t=0 means clean input" during training, but received pure noise at t=0 during inference.

**Old code (training)**:
```python
x_t = time_expanded * noise + (1 - time_expanded) * actions
```

**New code (training)**:
```python
if self.l1_flow:
    x_t = time_expanded * actions + (1 - time_expanded) * noise
else:
    x_t = time_expanded * noise + (1 - time_expanded) * actions
```

**Inference code** (`_l1_flow_sample`): unchanged — it was already correct for original L1Flow convention.

**Why**: Aligns with original L1Flow (https://github.com/THyanNK/L1Flow). In their convention, t=0 is noise, t=1 is clean sample. Our inference code was written for this convention but training wasn't.

---

### Fix 2: Remove Time Step Clamping for L1 Flow

**Files**: `src/openpi/models_pytorch/pi0_pytorch.py` (`sample_time`, line ~219), `src/openpi/models/pi0.py` (`compute_loss`, line ~238)

**Problem**: L1 Flow training mapped timesteps via `t * 0.999 + 0.001` to avoid boundary issues. The original L1Flow doesn't do this — LogisticNormal already samples from open interval (0,1), and sample-space L1 loss has no `1/(1-t)` division, so no numerical issue at boundaries.

**Old code**:
```python
t = self.timestep_sampler.sample(bsize)
t = t * 0.999 + 0.001  # Map to [0.001, 1.0]
```

**New code**:
```python
t = self.timestep_sampler.sample(bsize)
# No clamping — LogisticNormal is already in (0,1), sample-space L1 loss is numerically stable
```

---

### Fix 3: Pretrained Checkpoint Changed

**File**: `src/openpi/training/config.py` (line ~791)

**Old**:
```python
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
```

**New**:
```python
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_libero/params"),
```

**Why**: `pi05_base` is the pretrained base model. `pi05_libero` is already fine-tuned on LIBERO. Starting from a LIBERO-adapted checkpoint should give better initialization for L1 Flow fine-tuning. Architecture is identical (l1_flow only changes loss/inference, not model structure), so all weight keys match perfectly.

---

## How to Revert for Testing Old Model

To test the model trained with old code, revert these 3 changes:

```bash
# Option 1: git stash (if no other uncommitted changes)
git stash

# Option 2: revert specific files
git checkout HEAD -- \
  src/openpi/models_pytorch/pi0_pytorch.py \
  src/openpi/models/pi0.py \
  src/openpi/training/config.py
```

After testing, restore the fixed code:

```bash
# Option 1
git stash pop

# Option 2: re-apply the fixes manually or from this document
```

---

## Summary Table

| Item | Old (pre-fix) | New (fixed) | Impact |
|------|--------------|-------------|--------|
| Training interpolation (L1Flow) | `t*noise + (1-t)*actions` | `t*actions + (1-t)*noise` | Fixes train-inference t mismatch |
| Timestep range (L1Flow) | `[0.001, 1.0]` | `(0, 1)` raw | Matches original L1Flow |
| Pretrained checkpoint | `pi05_base` | `pi05_libero` | Better initialization |
| Inference code | unchanged | unchanged | Already matched original |
