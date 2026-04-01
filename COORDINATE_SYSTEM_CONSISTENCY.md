# Coordinate System Consistency Across stEVE Training

This document verifies that all components use consistent coordinate transformations and rotations for reliable RL training.

## 🎯 Why This Matters

For RL to work correctly across multiple meshes, **all models must use identical coordinate system transformations**:
- Same `rotation_yzx_deg` → Consistent 3D geometry and physics
- Same `fluoroscopy_rot_zx` → Consistent 2D observations (camera angle)

If different models have different transformations, the RL agent receives inconsistent observations and **cannot learn effectively**.

---

## ✅ Verified Settings Across All Components

### Standard Settings (Used Everywhere)

```python
rotation_yzx_deg = [90, -90, 0]   # 3D mesh rotation to match SOFA coordinates
fluoroscopy_rot_zx = [20, 5]      # 2D camera angle for observations
```

---

## 📍 Where These Settings Are Used

### 1. **Preprocessing** (`vmr_processing_tools/create_dualdevicenav_format.py`)

**Purpose:** Process VMR raw data (VTP/PTH) → stEVE-compatible OBJ/JSON

**Settings:**
```python
ROTATION_YZX_DEG = [-90, 0, 90]  # Line 41 - INVERSE of stEVE rotation
SCALING_FACTOR = 10.0            # Line 42 - cm to mm
```

**Why inverse rotation?**
- Preprocessing applies `[-90, 0, 90]` to data files
- stEVE then applies `[90, -90, 0]` when loading
- These cancel out to produce the final correct orientation
- This approach matches the original DualDeviceNav pipeline

**Files affected:**
- `{model}_collision.obj` - collision mesh
- `{model}_visual.obj` - visual mesh
- `Centrelines/*.mrk.json` - centerline JSON files

---

### 2. **DualDeviceNav Class** (`eve_bench/eve_bench/dualdevicenav.py`)

#### Original DualDeviceNav (Model 0105)

```python
# Line 36
rotation_yzx_deg=[90, -90, 0]

# Line 82
image_rot_zx=[20, 5]
```

**Hardcoded in the class definition** ✅

#### DualDeviceNavCustom (Custom Models)

```python
# Line 257-258 (default if not provided)
if rotation_yzx_deg is None:
    rotation_yzx_deg = [90, -90, 0]

# Line 312-313 (default if not provided)
if fluoroscopy_rot_zx is None:
    fluoroscopy_rot_zx = [20, 5]
```

**Defaults match the original** ✅

---

### 3. **Human Play Scripts** (`eve_bench/example/dual_human_play_0011.py`)

```python
# Line 65-66
rotation_yzx_deg=[90, -90, 0],
fluoroscopy_rot_zx=[20, 5]
```

**Explicitly specified** ✅

---

### 4. **RL Training Scripts**

#### Training on Model 0011 (`training_scripts/DualDeviceNav_train_0011.py`)

```python
# Lines 183-188 (UPDATED)
intervention = DualDeviceNavCustom(
    mesh_folder=model_path,
    model_name="0011_H_AO_H",
    rotation_yzx_deg=[90, -90, 0],  # EXPLICIT ✅
    fluoroscopy_rot_zx=[20, 5]      # EXPLICIT ✅
)
```

#### Training on Model 0105 (`training_scripts/DualDeviceNav_train.py`)

```python
# Line 139
intervention = DualDeviceNav()  # Uses hardcoded defaults ✅
```

**All training scripts use correct settings** ✅

---

## 🔍 How to Verify Consistency

### Before Training on Multiple Models

Run these checks to ensure all models use the same settings:

#### 1. Visual Check - All Models Should Look Similar

```bash
# Visualize all processed models
cd vmr_processing_tools
visualize_all_models_as_steve.bat

# All models should appear:
# - Vertical orientation
# - Insertion point at bottom
# - Target points in upper branches
# - Similar overall appearance
```

**If any model looks different (horizontal, upside down, etc.)** → Reprocess that model

#### 2. Test Human Play - Random Sampling

Test a few models with human play to verify they behave consistently:

```bash
# Test model 0011
python eve_bench/example/dual_human_play_0011.py

# Create similar scripts for other models
# All should have:
# - rotation_yzx_deg=[90, -90, 0]
# - fluoroscopy_rot_zx=[20, 5]
```

#### 3. Check Saved Environment Configs

After training starts, check the saved configs:

```bash
# Check training config
cat results/.../config/env_train.yml

# Verify these lines:
#   rotation_yzx_deg: [90, -90, 0]
#   fluoroscopy_rot_zx: [20, 5]
```

---

## 🚨 Common Mistakes to Avoid

### ❌ DON'T: Use different rotations for different models

```python
# BAD - Model-specific rotations
model_0011 = DualDeviceNavCustom(..., rotation_yzx_deg=[90, -90, 0])
model_0012 = DualDeviceNavCustom(..., rotation_yzx_deg=[0, 0, 0])  # WRONG!
```

### ❌ DON'T: Reprocess models with different rotation settings

```python
# BAD - Different preprocessing rotations
ROTATION_YZX_DEG = [-90, 0, 90]   # For model 0011
ROTATION_YZX_DEG = [0, 0, 0]      # For model 0012 - WRONG!
```

### ❌ DON'T: Use different fluoroscopy angles

```python
# BAD - Different camera angles
fluoroscopy_rot_zx=[20, 5]    # For training
fluoroscopy_rot_zx=[30, 10]   # For testing - WRONG!
```

### ✅ DO: Use the same settings everywhere

```python
# GOOD - Consistent across all models
STANDARD_ROTATION = [90, -90, 0]
STANDARD_FLUOROSCOPY = [20, 5]

model_0011 = DualDeviceNavCustom(..., 
    rotation_yzx_deg=STANDARD_ROTATION,
    fluoroscopy_rot_zx=STANDARD_FLUOROSCOPY)
    
model_0012 = DualDeviceNavCustom(...,
    rotation_yzx_deg=STANDARD_ROTATION,
    fluoroscopy_rot_zx=STANDARD_FLUOROSCOPY)
```

---

## 📊 Settings Summary Table

| Component | `rotation_yzx_deg` | `fluoroscopy_rot_zx` | Status |
|-----------|-------------------|---------------------|---------|
| **Preprocessing** | `[-90, 0, 90]` (inverse) | N/A | ✅ Correct |
| **DualDeviceNav (0105)** | `[90, -90, 0]` | `[20, 5]` | ✅ Hardcoded |
| **DualDeviceNavCustom (default)** | `[90, -90, 0]` | `[20, 5]` | ✅ Default |
| **dual_human_play_0011.py** | `[90, -90, 0]` | `[20, 5]` | ✅ Explicit |
| **DualDeviceNav_train.py** | `[90, -90, 0]` | `[20, 5]` | ✅ Hardcoded |
| **DualDeviceNav_train_0011.py** | `[90, -90, 0]` | `[20, 5]` | ✅ Explicit |

**All components use consistent settings** ✅

---

## 🎓 Understanding the Coordinate System

### Preprocessing Stage

```
Raw VMR Data (VTP/PTH)
  ↓ (scale 10x, rotate [-90, 0, 90])
Processed OBJ/JSON files
```

### stEVE Loading Stage

```
Processed OBJ/JSON files
  ↓ (rotate [90, -90, 0])
  ↓ (apply (y, -z, -x) to centerlines)
Final 3D geometry in SOFA
```

### Observation Generation

```
Final 3D geometry
  ↓ (apply fluoroscopy_rot_zx=[20, 5])
2D camera view
  ↓
RL Agent Observations
```

**Key Point:** The agent sees the same 2D view angles regardless of the original mesh orientation, as long as all meshes use the same `rotation_yzx_deg` and `fluoroscopy_rot_zx`.

---

## ✅ Verification Checklist

Before starting RL training with multiple models:

- [ ] All models processed with `create_dualdevicenav_format.py` (same script)
- [ ] Visual check: All models appear vertical in `visualize_all_models_as_steve.bat`
- [ ] Training scripts explicitly specify `rotation_yzx_deg=[90, -90, 0]`
- [ ] Training scripts explicitly specify `fluoroscopy_rot_zx=[20, 5]`
- [ ] Human play scripts use the same settings as training scripts
- [ ] Saved `env_train.yml` and `env_eval.yml` configs match expected values

---

## 📝 For Future Reference

If you need to add a new model:

1. **Process with standard script:**
   ```bash
   vmr_processing_tools\run_dualdevicenav.bat D:\vmr\vmr\<model_name>
   ```

2. **Verify visually:**
   ```bash
   python visualize_as_steve_sees_it.py D:\vmr\vmr\<model_name>\dualdevicenav_format
   ```

3. **Test with human play:**
   ```python
   DualDeviceNavCustom(
       mesh_folder=model_path,
       model_name="<model_name>",
       rotation_yzx_deg=[90, -90, 0],  # ALWAYS use this
       fluoroscopy_rot_zx=[20, 5]      # ALWAYS use this
   )
   ```

4. **Use in training with explicit parameters**

---

## 🔗 Related Files

- `vmr_processing_tools/create_dualdevicenav_format.py` - Preprocessing
- `eve_bench/eve_bench/dualdevicenav.py` - Environment definition
- `training_scripts/DualDeviceNav_train_0011.py` - Training script
- `eve_bench/example/dual_human_play_0011.py` - Human play test

---

**Last Updated:** December 2025  
**Verified Settings:** `rotation_yzx_deg=[90, -90, 0]`, `fluoroscopy_rot_zx=[20, 5]`








