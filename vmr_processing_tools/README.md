# VMR Processing Tools for stEVE

This folder contains tools to convert VMR data (VTP/VTU/PTH files) into stEVE-compatible format (OBJ + JSON).

## 📚 **Important: Read This First!**

**See [`RADIUS_COMPLETE_GUIDE.md`](RADIUS_COMPLETE_GUIDE.md)** for comprehensive documentation on:
- ✅ **Why radius information is CRITICAL** for stEVE training (used in 6 places!)
- ✅ **How to extract radius** using VMTK Python or 3D Slicer VMTK
- ✅ **Where radius is used** in navigation logic and RL rewards
- ✅ **Agent vs Environment** distinction (agent can't see radius, but needs it!)
- ✅ **Practical recommendations** for processing your VMR models
- ⚠️ **Why PTH conversion is NOT sufficient** (missing radius data)

**Key Takeaway:** For DualDeviceNav-quality training, you MUST use VMTK to extract centerlines with radius from VTP meshes. PTH files alone are insufficient!

---

## 🚀 Quick Start: Which Tool Should I Use?

| Your Goal | Use This Tool | Command |
|-----------|---------------|---------|
| **Create DualDeviceNav-like training data** | `create_dualdevicenav_format.py` | `python create_dualdevicenav_format.py vmr_downloads/0105_0001/` |
| **Batch process 51 VMR models** | `batch_create_dualdevicenav.py` | `python batch_create_dualdevicenav.py vmr_downloads/ --models-list vmr_filtered_models.txt` |
| **Just extract centerlines with radius** | `extract_centerlines_vmtk.py` | `python extract_centerlines_vmtk.py vmr_downloads/0105_0001/` |
| **Verify radius data is present** | `verify_radius_data.py` | `python verify_radius_data.py vmr_downloads/` |
| **Convert PTH to JSON (no radius)** ⚠️ | `convert_vmr_to_steve_format.py` | ⚠️ NOT recommended for training! |

**Recommended workflow for 51 models:**
1. Run `batch_create_dualdevicenav.py` on all models
2. Check `batch_processing_summary.txt` for failures
3. Manually process failures in 3D Slicer with SlicerVMTK
4. Run `verify_radius_data.py` to confirm all have radius

---

## 📋 Summary: What Code Exists in stEVE

### ✅ Already in stEVE Core:

| Feature | Location | Function |
|---------|----------|----------|
| **VTP → OBJ conversion** | `eve/eve/intervention/vesseltree/vmr.py:163-176` | `_make_mesh_obj()` |
| **JSON loading** | `eve_bench/eve_bench/dualdevicenav.py:108-136` | `load_points_from_json()` |
| **PTH (XML) loading** | `eve/eve/intervention/vesseltree/vmr.py:46-78` | `_load_points_from_pth()` |

### ❌ Missing (Created in This Folder):

| Feature | This Script | Function |
|---------|-------------|----------|
| **PTH → JSON conversion** | `convert_vmr_to_steve_format.py` | `convert_pth_to_slicer_json()` |
| **Batch VTP → OBJ** | `convert_vmr_to_steve_format.py` | `process_vmr_model_complete()` |
| **Complete pipeline** | `convert_vmr_to_steve_format.py` | All-in-one conversion |
| **VMTK centerline extraction** ⭐ | `extract_centerlines_vmtk.py` | Extract with radius (RECOMMENDED) |
| **Radius verification** | `verify_radius_data.py` | Verify radius data is present |
| **DualDeviceNav pipeline** 🎯 | `create_dualdevicenav_format.py` | **Complete pipeline** (VTP → OBJ + JSON with radius) |
| **Batch DualDeviceNav** | `batch_create_dualdevicenav.py` | Batch process multiple models |

---

## 🎯 **RECOMMENDED: Complete DualDeviceNav Pipeline**

### `create_dualdevicenav_format.py` ⭐ **NEW!**

**This is what you want!** Replicates the EXACT DualDeviceNav processing pipeline used for model 0105:

**What it does:**
1. ✅ **VTP mesh → OBJ files** (collision + visual) with correct transformations
   - Scaling: 10x (cm → mm)
   - Rotation: [90, -90, 0] (Y, Z, X)
   - Decimation: 0.9 (keep 10% of faces)
   - Flip normals for SOFA

2. ✅ **VTP mesh → VMTK centerline extraction** with radius (MISR)
   - Automatic or manual endpoint selection
   - Resamples to 1mm resolution
   - Computes MaximumInscribedSphereRadius

3. ✅ **Centerlines → DualDeviceNav JSON format**
   - LPS coordinate system
   - Radius measurements included
   - Exact 3D Slicer markup format
   - Transformation (y, -z, -x) applied by stEVE on load

**Single model:**
```bash
# Process one VMR model
python vmr_processing_tools/create_dualdevicenav_format.py vmr_downloads/0105_0001/

# Output:
#   vmr_downloads/0105_0001/dualdevicenav_format/
#     ├── 0105_0001_collision.obj      # For SOFA physics
#     ├── 0105_0001_visual.obj         # For visualization
#     └── Centrelines/
#         └── Centerline curve - 0105_0001.mrk.json  # With radius!
```

**Batch processing:**
```bash
# Process all 51 filtered models
python vmr_processing_tools/batch_create_dualdevicenav.py vmr_downloads/ \
    --models-list vmr_download_tools/vmr_filtered_models.txt

# Or process all models in folder
python vmr_processing_tools/batch_create_dualdevicenav.py vmr_downloads/
```

**Expected results:**
- ✅ ~70-80% success rate with automatic endpoint detection
- ⚠️ ~20-30% may need manual processing in 3D Slicer (complex anatomies)
- 📊 Creates `batch_processing_summary.txt` with results

**For models that fail automatic extraction:**
1. Open VTP in 3D Slicer
2. Use SlicerVMTK → Extract Centerline (manual endpoint selection)
3. Export as `.mrk.json` to the Centrelines/ folder

---

## 🎯 What This Script Does

### `convert_vmr_to_steve_format.py`

**Purpose:** Convert downloaded VMR models (VTP/VTU/PTH) to stEVE training format (OBJ + JSON)

**Capabilities:**
1. ✅ Converts VTP surface meshes → OBJ files (collision + visual)
2. ✅ Converts PTH centerlines → 3D Slicer .mrk.json files
3. ✅ Applies scaling (cm → mm)
4. ✅ Applies rotation transformations
5. ✅ Applies coordinate transformations (e.g., DualDeviceNav format)
6. ✅ Batch processes multiple models
7. ✅ Creates stEVE-compatible file structure

---

## 🚀 Quick Start

### Process a Single VMR Model:

```bash
# Basic conversion
python vmr_processing_tools/convert_vmr_to_steve_format.py vmr_downloads/0105_0001/

# With rotation (like DualDeviceNav uses)
python vmr_processing_tools/convert_vmr_to_steve_format.py vmr_downloads/0105_0001/ --rotation 90,-90,0

# With coordinate transform for DualDeviceNav format
python vmr_processing_tools/convert_vmr_to_steve_format.py vmr_downloads/0105_0001/ --rotation 90,-90,0 --transform dual
```

### Batch Process All Downloaded Models:

```bash
python vmr_processing_tools/convert_vmr_to_steve_format.py vmr_downloads/ --batch
```

---

## 📁 Output Structure

After processing `vmr_downloads/0105_0001/`, you'll get:

```
vmr_downloads/0105_0001/
├── Meshes/                           (Original VMR files)
│   ├── 0105_0001.vtp
│   └── 0105_0001.vtu
├── Paths/                            (Original PTH centerlines)
│   ├── aorta.pth
│   ├── lcca.pth
│   └── ...
└── steve_format/                     (NEW - Generated by script)
    ├── 0105_0001_collision.obj       ← Collision mesh (90% decimated)
    ├── 0105_0001_visual.obj          ← Visual mesh (95% decimated)
    └── Centerlines/                  ← Centerlines in stEVE format
        ├── Centerline curve - aorta.mrk.json
        ├── Centerline curve - lcca.mrk.json
        └── ...
```

---

## 🔧 Detailed Usage

### 1. Convert VTP to OBJ Only

```python
from convert_vmr_to_steve_format import convert_vtp_to_obj

convert_vtp_to_obj(
    vtp_file="vmr_downloads/0105_0001/Meshes/0105_0001.vtp",
    output_obj_file="output/vessel.obj",
    scale_to_mm=True,
    rotation=(90, -90, 0),  # (y, z, x) in degrees
    decimation=0.9          # Keep 10% of triangles
)
```

### 2. Convert PTH to JSON Only

```python
from convert_vmr_to_steve_format import convert_pth_to_slicer_json, dualdevicenav_transform

convert_pth_to_slicer_json(
    pth_file="vmr_downloads/0105_0001/Paths/aorta.pth",
    output_json_file="output/aorta.mrk.json",
    scale_to_mm=True,
    coordinate_transform=dualdevicenav_transform  # Optional: (x,y,z) -> (y,-z,-x)
)
```

### 3. Complete Model Processing

```python
from convert_vmr_to_steve_format import process_vmr_model_complete

process_vmr_model_complete(
    vmr_folder="vmr_downloads/0105_0001/",
    output_folder=None,              # Default: vmr_folder/steve_format/
    obj_basename="vessel_model",     # Output OBJ names
    rotation=(90, -90, 0),           # Optional rotation
    coordinate_transform=None        # Optional coord transform
)
```

### 4. Batch Processing

```python
from convert_vmr_to_steve_format import batch_process_vmr_models

# Process all models
batch_process_vmr_models("vmr_downloads/")

# Process specific models
batch_process_vmr_models(
    "vmr_downloads/",
    model_ids=["0105_0001", "0166_0001", "0078_0001"]
)
```

---

## 🎨 Coordinate Transforms

### DualDeviceNav Transform

The DualDeviceNav benchmark uses a specific coordinate transformation:

```python
# Original VMR: (x, y, z)
# DualDeviceNav: (y, -z, -x)

def dualdevicenav_transform(x, y, z):
    return (y, -z, -x)
```

**Usage:**
```bash
python convert_vmr_to_steve_format.py vmr_downloads/0105_0001/ --transform dual
```

### Custom Transforms

Create your own transformation function:

```python
def my_custom_transform(x, y, z):
    # Example: rotate 90° around Z
    return (-y, x, z)

process_vmr_model_complete(
    vmr_folder="vmr_downloads/0105_0001/",
    coordinate_transform=my_custom_transform
)
```

---

## 📊 Comparison with Manual Workflow

| Step | Manual (3D Slicer) | This Script | DualDeviceNav Original |
|------|-------------------|-------------|----------------------|
| **VTP → OBJ** | Export from Slicer | ✅ Automated | Used Blender + manual editing |
| **Extract Centerlines** | VMTK Extension | ⚠️ Use PTH or Slicer | Used VMTK + manual refinement |
| **PTH → JSON** | N/A (Slicer exports JSON) | ✅ Automated | N/A (started with VMTK JSON) |
| **Scaling** | Manual adjustment | ✅ Automated | Manual in Blender |
| **Rotation** | Manual in Slicer | ✅ Automated | Manual in Blender |
| **Batch Processing** | ❌ One at a time | ✅ Automated | ❌ One at a time |
| **Mesh Refinement** | Blender (advanced) | Basic decimation | Extensive Blender editing |

---

## ⚠️ Important Notes

### Centerlines from PTH Files:

**PTH files contain centerlines BUT NO RADIUS information!**

If you need radius data (like DualDeviceNav has):
1. **Option A:** Extract centerlines in 3D Slicer using VMTK
   - Includes radius automatically
   - Export as `.mrk.json` directly
   
2. **Option B:** Use this script's PTH conversion
   - Converts geometry only
   - Radius info must be added separately

### Mesh Quality:

- **This script:** Basic decimation only
- **DualDeviceNav:** Extensive manual refinement in Blender
  - Smoothing
  - Remeshing
  - Bifurcation cleanup
  - Visual polish

For production use (like DualDeviceNav), consider:
1. Use this script for initial conversion
2. Import OBJ into Blender for refinement
3. Export final version

---

## 🔗 Integration with stEVE

### Use Converted Data in Training:

```python
import eve
from eve.intervention.vesseltree.util.branch import BranchWithRadii

# Load your converted data
mesh = "vmr_downloads/0105_0001/steve_format/0105_0001_collision.obj"
visu_mesh = "vmr_downloads/0105_0001/steve_format/0105_0001_visual.obj"

# Load centerlines from JSON
from eve_bench.eve_bench.dualdevicenav import load_branches
centerlines_folder = "vmr_downloads/0105_0001/steve_format/Centerlines/"
branches = load_branches(centerlines_folder)

# Create vessel tree
vessel_tree = eve.intervention.vesseltree.FromMesh(
    mesh=mesh,
    insertion_position=[65.0, -5.0, 35.0],
    insertion_direction=[-1.0, 0.0, 1.0],
    branch_list=branches,
    rotation_yzx_deg=[90, -90, 0],
    visu_mesh=visu_mesh
)
```

---

## 📝 Example: Replicate DualDeviceNav Format

To create data similar to DualDeviceNav:

```bash
# 1. Convert VMR model with DualDeviceNav settings
python vmr_processing_tools/convert_vmr_to_steve_format.py \
    vmr_downloads/0105_0001/ \
    --rotation 90,-90,0 \
    --transform dual

# 2. (Optional) Refine in Blender
# - Import steve_format/0105_0001_collision.obj
# - Apply smoothing, remeshing
# - Export as final_collision.obj

# 3. Extract centerlines with radius in 3D Slicer
# - Load VTP
# - Use SlicerVMTK "Extract Centerline"
# - Mark endpoints
# - Export each centerline as .mrk.json

# 4. Use in stEVE training!
```

---

## 🛠️ Troubleshooting

### Issue: No PTH files found
**Solution:** Extract centerlines in 3D Slicer using VMTK
- See `visualization_tools/README.md` for instructions

### Issue: JSON files missing radius data
**Solution:** PTH files don't contain radius
- Extract centerlines in 3D Slicer with VMTK instead
- VMTK automatically computes radius

### Issue: Mesh too large/slow
**Solution:** Increase decimation factor
```python
convert_vtp_to_obj(vtp_file, output_file, decimation=0.8)  # More aggressive
```

### Issue: Coordinate system mismatch
**Solution:** Check if coordinate transform is needed
- DualDeviceNav uses: `(y, -z, -x)`
- Try different transforms until it matches

---

## 🔬 Advanced: Learn CHS Statistics

After converting multiple models:

```python
import numpy as np
from pathlib import Path

# Collect centerline statistics from multiple patients
centerline_folders = [
    Path("vmr_downloads/0105_0001/steve_format/Centerlines/"),
    Path("vmr_downloads/0166_0001/steve_format/Centerlines/"),
    # ... more models
]

# Load all centerlines
from eve_bench.eve_bench.dualdevicenav import load_branches

all_aortas = []
for folder in centerline_folders:
    branches = load_branches(str(folder))
    aorta = [b for b in branches if 'aorta' in b.name.lower()]
    if aorta:
        all_aortas.append(aorta[0])

# Compute statistics
coords = np.concatenate([a.coordinates for a in all_aortas])
radii = np.concatenate([a.radii for a in all_aortas])

print("Aorta Statistics:")
print(f"  Position mean: {coords.mean(axis=0)}")
print(f"  Position std: {coords.std(axis=0)}")
print(f"  Radius mean: {radii.mean():.2f} mm")
print(f"  Radius std: {radii.std():.2f} mm")

# Use these for CHS parameter learning!
```

---

## 📚 Related Tools

- **`vmr_download_tools/`** - Download VMR models
- **`visualization_tools/`** - Visualize meshes with PyVista
- **`eve/eve/intervention/vesseltree/`** - stEVE vessel tree classes
- **`eve_bench/data/dualdevicenav/`** - Example of final format

---

## 🎓 Learning Pipeline

**Complete workflow to use your VMR data for CHS learning:**

```bash
# 1. Download models
python vmr_download_tools/download_filtered_vmr.py

# 2. Convert to stEVE format
python vmr_processing_tools/convert_vmr_to_steve_format.py vmr_downloads/ --batch

# 3. Extract centerlines with radius in 3D Slicer (VMTK)
# (For each model - see visualization_tools/README.md)

# 4. Analyze statistics
python analyze_centerline_statistics.py

# 5. Update CHS generators with learned parameters
# Edit eve/eve/intervention/vesseltree/aorticarcharteries/*.py

# 6. Train RL agent with more realistic vessel generation!
python training_scripts/ArchVariety_train.py
```

---

**Last Updated:** November 10, 2025

