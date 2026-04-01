# Example Workflow: Processing VMR Models for DualDeviceNav Training

## Your Goal
Create DualDeviceNav-compatible training data from your 51 VMR models, with the same format as model 0105 used in the original DualDeviceNav.

## Complete Workflow

### Step 1: Install Dependencies

```bash
# Create conda environment
conda create -n vmtk python=3.9
conda activate vmtk

# Install VMTK
conda install -c vmtk vmtk

# Install PyVista
pip install pyvista
```

### Step 2: Single Model Test

Test on one model first to verify everything works:

```bash
# Test on model 0105_0001
python vmr_processing_tools/create_dualdevicenav_format.py vmr_downloads/0105_0001/

# Check output
ls vmr_downloads/0105_0001/dualdevicenav_format/
# Should see:
#   0105_0001_collision.obj
#   0105_0001_visual.obj
#   Centrelines/
#     Centerline curve - 0105_0001.mrk.json
```

**Verify the JSON has radius:**

```bash
python vmr_processing_tools/verify_radius_data.py vmr_downloads/0105_0001/dualdevicenav_format/

# Should show:
#   ✅ Centerline curve - 0105_0001.mrk.json
#      Points: XXX
#      Radius: X.XX - XX.XX mm
```

**Visualize the processed model:**

```bash
# For model 0011 (using batch file)
vmr_processing_tools\visualize_0011.bat

# Or for any model (using Python directly)
python vmr_processing_tools\visualize_processed_model.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format

# Advanced visualization with radius color-coding
python vmr_processing_tools\visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format --radius-colormap

# Compare collision vs visual mesh
python vmr_processing_tools\visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format --split-view
```

This opens an interactive 3D window showing:
- The processed mesh (collision and/or visual)
- All centerline branches with color coding
- Radius information visualized as tube thickness
- LPS coordinate system axes

**Controls:**
- Mouse drag: Rotate view
- Mouse wheel: Zoom
- Right mouse drag: Pan
- 'q' or close window: Exit

### Step 3: Batch Process All 51 Models

```bash
# Process all filtered models
python vmr_processing_tools/batch_create_dualdevicenav.py vmr_downloads/ \
    --models-list vmr_download_tools/vmr_filtered_models.txt

# This will:
#   - Process all 51 models automatically
#   - Continue even if some fail
#   - Create batch_processing_summary.txt
#   - Take ~2-4 hours (depending on your system)
```

### Step 4: Check Results

```bash
# View summary
cat vmr_downloads/batch_processing_summary.txt

# Expected:
#   Total models: 51
#   Successful: ~35-40 (70-80%)
#   Failed: ~10-15 (20-30%)
```

### Step 5: Manual Processing for Failures

For models that failed automatic centerline extraction:

**Option A: 3D Slicer (Recommended for quality)**

```bash
# 1. Open 3D Slicer
# 2. Load VTP file: vmr_downloads/MODEL_NAME/Meshes/MODEL_NAME.vtp
# 3. Go to: Modules → VMTK → Extract Centerline
# 4. Click endpoints on the mesh visually
# 5. Click "Apply"
# 6. Save As → "Centerline curve - MODEL_NAME.mrk.json"
# 7. Move to: vmr_downloads/MODEL_NAME/dualdevicenav_format/Centrelines/
```

**Option B: Manual Endpoint Specification**

If you know approximate endpoint coordinates:

```python
# Edit create_dualdevicenav_format.py and add:
source_points = [(x1, y1, z1)]  # Start point
target_points = [(x2, y2, z2), (x3, y3, z3)]  # End points

# Then run:
python create_dualdevicenav_format.py vmr_downloads/MODEL_NAME/ \
    --source x1,y1,z1 --targets x2,y2,z2,x3,y3,z3
```

### Step 6: Verify All Models

```bash
# Verify all models have radius data
python vmr_processing_tools/verify_radius_data.py vmr_downloads/*/dualdevicenav_format/

# Should show:
#   51/51 files have radius data
#   ✅ All files have radius data!
```

### Step 7: Use in Training

```python
# Copy DualDeviceNav code and modify paths
import eve_bench

class MyDualDeviceNav(eve_bench.DualDeviceNav):
    def __init__(self):
        # Change these to your processed model
        mesh = "vmr_downloads/0105_0001/dualdevicenav_format/0105_0001_collision.obj"
        visu_mesh = "vmr_downloads/0105_0001/dualdevicenav_format/0105_0001_visual.obj"
        centerline_folder = "vmr_downloads/0105_0001/dualdevicenav_format/Centrelines/"
        
        # Rest stays the same...
```

---

## Expected Output Structure

After processing, each model will have:

```
vmr_downloads/
├── 0078_0001/
│   └── dualdevicenav_format/
│       ├── 0078_0001_collision.obj       # For SOFA physics
│       ├── 0078_0001_visual.obj          # For visualization
│       └── Centrelines/
│           └── Centerline curve - 0078_0001.mrk.json  # With radius!
├── 0079_0001/
│   └── dualdevicenav_format/
│       ├── 0079_0001_collision.obj
│       ├── 0079_0001_visual.obj
│       └── Centrelines/
│           └── Centerline curve - 0079_0001.mrk.json
├── ...
└── batch_processing_summary.txt
```

---

## Troubleshooting

### "ERROR: No centerlines extracted!"

**Cause:** Automatic endpoint detection failed on complex anatomy

**Solutions:**
1. Use 3D Slicer with manual endpoint selection (recommended)
2. Inspect the mesh to find approximate source/target coordinates
3. Try adjusting VMTK parameters (ResamplingStepLength)

### "ERROR: No radius information in centerlines!"

**Cause:** VMTK failed to compute MaximumInscribedSphereRadius

**Solutions:**
1. Check if VTP mesh is valid (open in ParaView/3D Slicer)
2. Mesh may need cleaning (holes, non-manifold edges)
3. Use 3D Slicer VMTK which is more robust

### "Centerlines look wrong"

**Cause:** Wrong source/target points

**Solutions:**
1. Visualize in 3D Slicer to check
2. Re-extract with correct manual endpoints
3. Check if mesh orientation is correct

### Models process slowly

**Expected:** ~2-5 min per model (VMTK centerline extraction is compute-intensive)

**Tips:**
- Run overnight for 51 models (~2-4 hours total)
- Process in parallel if you have multiple cores
- Visual mesh creation can be skipped with `--no-visual`

---

## Quality Checklist

Before using for training, verify:

- [ ] All models have `_collision.obj` files
- [ ] All models have `.mrk.json` files in Centrelines/
- [ ] `verify_radius_data.py` shows ✅ for all files
- [ ] Radius values are reasonable (0.5mm - 15mm for neurovascular)
- [ ] Visual inspection of at least 10% of models in 3D Slicer
- [ ] Test loading in stEVE with one model
- [ ] Coordinate transformation (y, -z, -x) produces correct orientation

---

## FAQ

**Q: Can I skip the visual mesh?**
A: Yes, use `--no-visual` flag. Only collision mesh is required for training.

**Q: How long does batch processing take?**
A: ~2-4 hours for 51 models (2-5 min per model). Run overnight.

**Q: What if only a few models fail?**
A: Expected! ~20-30% failure rate is normal. Process manually in 3D Slicer.

**Q: Can I use PTH files instead?**
A: NO! PTH files have no radius. Training will fail. See RADIUS_COMPLETE_GUIDE.md for why.

**Q: How do I verify the data is correct?**
A: 
1. Run `verify_radius_data.py`
2. Load one model in stEVE and check visually
3. Check JSON has "Radius" measurement with values

**Q: Can I process different anatomies (not neurovascular)?**
A: Yes! The pipeline works for any vascular anatomy in VMR.

---

## Advanced: Custom Transformations

If you need different transformations than DualDeviceNav:

```python
# Edit create_dualdevicenav_format.py

# Change rotation (default: [90, -90, 0])
ROTATION_YZX_DEG = [0, 0, 0]  # No rotation

# Change scaling (default: 10.0 for cm→mm)
SCALING_FACTOR = 1.0  # Keep in cm

# Change coordinate system
# Edit load_points_from_json() coordinate transformation
# Default: (y, -z, -x)
# Change to: (x, y, z) for no transformation
```

---

## Support

If you encounter issues:
1. Check `RADIUS_COMPLETE_GUIDE.md` for detailed explanations
2. Verify VMTK installation: `conda list vmtk`
3. Test on a single model first before batch processing
4. Check `batch_processing_summary.txt` for specific error messages

**Remember:** The goal is DualDeviceNav-compatible data with radius information. Don't skip the radius extraction step!


