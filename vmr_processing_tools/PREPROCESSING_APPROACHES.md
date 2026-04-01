# Two Preprocessing Approaches

This folder contains **TWO** different scripts for creating DualDeviceNav-compatible data from VMR models. Choose based on your preference:

---

## 🎯 Approach 1: NO PRE-ROTATION (RECOMMENDED - Cleaner!)

**Script:** `create_dualdevicenav_format_norot.py`  
**Batch file:** `run_dualdevicenav_norot.bat`

### What it does:
- ✅ **OBJ mesh:** Scaled only (cm → mm), **NO rotation applied**
- ✅ **JSON centerlines:** Scaled only, **NO rotation applied**
- ✅ **All rotation handled by stEVE** via `rotation_yzx_deg` and `rotate_branches`/`rotate_ip`

### Output folder:
```
vmr_folder/dualdevicenav_format_norot/
```

### Usage in stEVE:
```python
intervention = DualDeviceNavCustom(
    mesh_folder="path/to/dualdevicenav_format_norot",
    model_name="0011_H_AO_H",
    insertion_point=insertion_point,
    rotation_yzx_deg=[90, -90, 0],  # Or any rotation you want
    rotate_branches=True,            # Must be True
    rotate_ip=True,                  # Must be True
    fluoroscopy_rot_zx=[20, 5]       # Standard fluoroscopy angles
)
```

### Advantages:
- ✅ **Cleaner:** Raw data is unrotated, all transformation logic in stEVE
- ✅ **Flexible:** Easy to change rotation by modifying `rotation_yzx_deg`
- ✅ **No double rotation issues:** Single source of truth for rotation
- ✅ **Standard fluoroscopy angles:** Can use normal `[20, 5]` view angles

---

## 📦 Approach 2: PRE-ROTATED (Original 0105 Style)

**Script:** `create_dualdevicenav_format.py`  
**Batch file:** `run_dualdevicenav.bat`

### What it does:
- ✅ **OBJ mesh:** Scaled + rotated by `[90, -90, 0]`
- ✅ **JSON centerlines:** Scaled + rotated by `[90, -90, 0]`
- ✅ **stEVE applies NO additional rotation** (data already in final orientation)

### Output folder:
```
vmr_folder/dualdevicenav_format/
```

### Usage in stEVE:
```python
intervention = DualDeviceNavCustom(
    mesh_folder="path/to/dualdevicenav_format",
    model_name="0011_H_AO_H",
    insertion_point=insertion_point,
    rotation_yzx_deg=[0, 0, 0],      # No rotation needed (already rotated)
    rotate_branches=False,            # Must be False
    rotate_ip=False,                  # Must be False
    fluoroscopy_rot_zx=[20, 5]        # Standard fluoroscopy angles
)
```

### Advantages:
- ✅ **Matches original 0105 approach:** Same pipeline as the paper authors
- ✅ **Pre-oriented data:** No transformation needed at runtime

### Disadvantages:
- ⚠️ **Harder to debug:** Rotation baked into data files
- ⚠️ **Less flexible:** Need to reprocess to change orientation

---

## 🤔 Which Should You Use?

### Use **NO PRE-ROTATION** (Approach 1) if:
- ✅ You want a cleaner, more flexible pipeline
- ✅ You might experiment with different rotations
- ✅ You want to understand transformation logic better
- ✅ You're creating NEW models (0011, 0012, etc.)

### Use **PRE-ROTATED** (Approach 2) if:
- ✅ You want to exactly match the original 0105 pipeline
- ✅ You're replicating published results
- ✅ You prefer data files to be pre-oriented

---

## 📊 Comparison Table

| Feature | NO PRE-ROTATION | PRE-ROTATED |
|---------|----------------|-------------|
| **OBJ mesh** | Scaled only | Scaled + rotated |
| **JSON centerlines** | Scaled only | Scaled + rotated |
| **rotation_yzx_deg** | `[90, -90, 0]` | `[0, 0, 0]` |
| **rotate_branches** | `True` | `False` |
| **rotate_ip** | `True` | `False` |
| **Fluoroscopy angles** | Standard `[20, 5]` | Standard `[20, 5]` |
| **Flexibility** | ✅ High | ⚠️ Low |
| **Matches 0105** | ❌ No | ✅ Yes |
| **Easier to debug** | ✅ Yes | ⚠️ No |

---

## 🎬 Quick Start

### For NO PRE-ROTATION (RECOMMENDED):
```bash
# Windows
cd vmr_processing_tools
run_dualdevicenav_norot.bat

# Linux/Mac
python create_dualdevicenav_format_norot.py /path/to/vmr/model
```

### For PRE-ROTATED (0105 style):
```bash
# Windows
cd vmr_processing_tools
run_dualdevicenav.bat

# Linux/Mac
python create_dualdevicenav_format.py /path/to/vmr/model
```

---

## 💡 Key Insight

The confusion in your earlier attempts came from **double rotation**:
- You applied rotation during preprocessing (`create_dualdevicenav_format.py`)
- Then used `rotate_branches=True` and `rotate_ip=True` in stEVE
- This caused the mesh to be rotated TWICE!

**Solution:**
- Use **Approach 1** (no pre-rotation) for cleaner pipeline
- Or use **Approach 2** (pre-rotated) with `rotate_branches=False` and `rotate_ip=False`

**NEVER mix the two approaches!**









