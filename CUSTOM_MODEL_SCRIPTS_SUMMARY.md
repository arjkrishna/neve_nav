# Custom VMR Model Scripts - Summary

Created scripts to use newly processed VMR models (like 0011_H_AO_H) with the stEVE environment, allowing you to train and test RL agents on new anatomies.

## What Was Created

### 1. **Core Infrastructure**

#### `eve_bench/eve_bench/dualdevicenav.py`
- Added `DualDeviceNavCustom` class
- Accepts custom model paths instead of hardcoded model 0105
- Auto-detects insertion points and target branches
- Validates required files exist

### 2. **Play/Test Scripts**

#### `eve_bench/example/dual_human_play_0011.py`
- Interactive manual control script for model 0011
- Allows you to test the processed model manually
- Same controls as original `dual_human_play.py`

### 3. **Training Scripts**

#### `training_scripts/DualDeviceNav_train_0011.py`
- Full RL training pipeline for model 0011
- Same configuration as original training script
- Results saved to `results/vmr_models/0011_H_AO_H/`

### 4. **Documentation**

#### `eve_bench/example/README_CUSTOM_MODELS.md`
- Complete guide for using custom models
- Examples for creating new scripts
- Troubleshooting tips

## Quick Start

### Step 1: Process a Model

```bash
cd D:\stEVE_training
vmr_processing_tools\run_dualdevicenav.bat
```

This creates the required files in:
```
D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format\
├── vessel_architecture_collision.obj
├── vessel_architecture_visual.obj
└── Centrelines\
    ├── Centerline curve - 0011_H_AO_H.mrk.json
    ├── Centerline curve (1).mrk.json
    └── ...
```

### Step 2: Test the Model Manually

```bash
python eve_bench\example\dual_human_play_0011.py
```

**Controls:**
- Arrow keys to control devices
- ENTER to reset
- ESC to exit

### Step 3: Train an RL Agent (Optional)

```bash
python training_scripts\DualDeviceNav_train_0011.py -nw 4 -d cuda -n my_first_run
```

## Using Other Models

To use a different model (e.g., 0015_H_AO_COA):

### 1. Process it:
```bash
python vmr_processing_tools\create_dualdevicenav_format.py D:\vmr\vmr\0015_H_AO_COA
```

### 2. Create a play script:

Copy `dual_human_play_0011.py` and change:
```python
MODEL_0011_PATH = r"D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format"
```
to:
```python
MODEL_0015_PATH = r"D:\vmr\vmr\0015_H_AO_COA\dualdevicenav_format"
```

And:
```python
intervention = DualDeviceNavCustom(
    mesh_folder=MODEL_0015_PATH,
    model_name="0015_H_AO_COA"
)
```

### 3. Or use it programmatically:

```python
from eve_bench.dualdevicenav import DualDeviceNavCustom

intervention = DualDeviceNavCustom(
    mesh_folder=r"D:\vmr\vmr\0015_H_AO_COA\dualdevicenav_format",
    model_name="0015_H_AO_COA",
    insertion_point=None  # Auto-detect
)
```

## Key Features

### `DualDeviceNavCustom` Class

**Parameters:**
- `mesh_folder` (str): Path to `dualdevicenav_format` folder
- `model_name` (str): Name for logging/display
- `insertion_point` (list, optional): Custom `[x, y, z]` or `None` for auto-detect
- `stop_device_at_tree_end` (bool): Stop at vessel tree end
- `normalize_action` (bool): Normalize actions

**Features:**
- ✅ Auto-detects insertion point from first branch
- ✅ Auto-detects target branches
- ✅ Validates all required files exist
- ✅ Same interface as original `DualDeviceNav`
- ✅ Works with any processed VMR model

## File Requirements

Each processed model needs:

```
<model_folder>/dualdevicenav_format/
├── vessel_architecture_collision.obj  # Required: collision mesh
├── vessel_architecture_visual.obj     # Required: visual mesh
└── Centrelines/                       # Required: centerlines folder
    ├── Centerline curve - <name>.mrk.json  # At least 1 required
    ├── Centerline curve (1).mrk.json       # Additional branches
    └── ...
```

Each JSON file must contain:
- Centerline coordinates in LPS coordinate system
- Radius measurements for each point

## Batch Processing

Process all 44 ready models at once:

```bash
# Check which models are ready
python vmr_processing_tools\check_vmr_files.py D:\vmr\vmr

# Process all ready models
python vmr_processing_tools\batch_create_dualdevicenav.py D:\vmr\vmr
```

## Training Results

Results from model 0011 training will be saved to:
```
results/vmr_models/0011_H_AO_H/
├── <trial_name>/
    ├── checkpoints/
    ├── configs/
    ├── logs/
    └── results.csv
```

Compare this with the original model 0105 results in:
```
results/eve_paper/neurovascular/
```

## Differences from Original DualDeviceNav

| Aspect | Original | Custom |
|--------|----------|--------|
| Model | Hardcoded 0105 | Any processed VMR model |
| Data Path | `eve_bench/data/dualdevicenav/` | User-specified path |
| Insertion Point | Hardcoded | Auto-detected or custom |
| Target Branches | Hardcoded 4 branches | Auto-detected all branches |
| Flexibility | Single model | Unlimited models |

## Common Issues

### "Model data not found"
**Solution:** Run the processing script first:
```bash
vmr_processing_tools\run_dualdevicenav.bat
```

### "No centerlines found"
**Solution:** VMTK extraction failed. Check:
- VTP file exists and is valid
- PTH files exist
- Conda environment is activated

### Simulation crashes
**Solution:**
- Verify radius data exists: `python vmr_processing_tools\verify_radius_data.py`
- Check insertion point is inside vessel
- Ensure all JSON files are valid

## Next Steps

1. ✅ **Process model 0011** using `run_dualdevicenav.bat`
2. ✅ **Test it** with `dual_human_play_0011.py`
3. ⭕ **Batch process** other models for comparison
4. ⭕ **Train agents** on multiple anatomies
5. ⭕ **Compare performance** across different models

## See Also

- `vmr_processing_tools/RADIUS_COMPLETE_GUIDE.md` - Why radius matters
- `vmr_processing_tools/EXAMPLE_WORKFLOW.md` - Processing workflow
- `eve_bench/example/README_CUSTOM_MODELS.md` - Detailed usage guide
- `vmr_file_check_results.txt` - List of all processable models

## Model Availability

From the file check, you have:
- **44 models ready** for immediate processing (have VTP + PTH)
- **1 model** needs manual endpoint selection
- **5 models** cannot be processed (missing files)

All 44 ready models can now be used for training!


