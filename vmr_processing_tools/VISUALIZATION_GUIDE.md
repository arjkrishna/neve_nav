# Visualization Guide for Processed Models

After processing your VMR models into DualDeviceNav format, you can visualize them to verify the data quality and understand the vessel anatomy.

## Quick Start

### Basic Visualization (Recommended)

For model 0011:
```bash
# Using batch file (Windows)
vmr_processing_tools\visualize_0011.bat

# Or using Python directly
python vmr_processing_tools\visualize_processed_model.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
```

This shows:
- ✅ Collision mesh (semi-transparent blue)
- ✅ All centerline branches (color-coded)
- ✅ Centerline points
- ✅ Coordinate axes (LPS system)

### Advanced Visualizations

#### 1. Radius Color-Coded Centerlines

Visualize vessel radius as color (blue = small, red = large):

```bash
python vmr_processing_tools\visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format --radius-colormap
```

**Use case:** Verify radius data was extracted correctly and understand vessel size distribution.

#### 2. Centerlines Only (No Mesh)

Show only the centerlines without the mesh:

```bash
python vmr_processing_tools\visualize_processed_model.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format centerlines_only
```

**Use case:** Focus on the vessel tree structure and branching points.

#### 3. Split View Comparison

Compare collision vs visual mesh side-by-side:

```bash
python vmr_processing_tools\visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format --split-view
```

**Use case:** Understand the difference between collision and visual meshes (decimation factors).

#### 4. Visual Mesh Only

Show the higher-resolution visual mesh:

```bash
python vmr_processing_tools\visualize_processed_model.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format visual
```

**Use case:** See more detailed vessel geometry.

## Interactive Controls

All visualizations support the same controls:

| Action | Control |
|--------|---------|
| Rotate view | Left mouse drag |
| Zoom | Mouse wheel |
| Pan | Right mouse drag |
| Reset camera | 'r' key |
| Exit | 'q' key or close window |

## What to Look For

### ✅ Good Signs

1. **Mesh**
   - Smooth surface without holes
   - Proper scaling (vessels should be in millimeters)
   - Correct orientation (ascending aorta pointing upward)

2. **Centerlines**
   - Follow the center of vessels
   - No sudden jumps or discontinuities
   - Branches extend to vessel ends
   - Radius values are reasonable (0.5-15 mm for neurovascular)

3. **Registration**
   - Centerlines are inside the mesh
   - Centerlines follow mesh geometry closely
   - Branching points align with mesh bifurcations

### ❌ Warning Signs

1. **Mesh Issues**
   - ⚠️ Holes or gaps in the surface → Mesh quality problem
   - ⚠️ Inside-out normals → May cause simulation issues
   - ⚠️ Wrong scale (too large/small) → Check SCALING_FACTOR

2. **Centerline Issues**
   - ⚠️ Centerlines outside mesh → Coordinate transformation error
   - ⚠️ Abrupt radius changes → Noisy radius data
   - ⚠️ Missing branches → Incomplete extraction
   - ⚠️ Centerlines too short → Wrong endpoints

3. **Registration Issues**
   - ⚠️ Offset between mesh and centerlines → Coordinate mismatch
   - ⚠️ Rotation mismatch → Check ROTATION_YZX_DEG

## Common Issues and Fixes

### Issue: Centerlines are outside the mesh

**Cause:** Coordinate transformation mismatch

**Fix:**
1. Check that JSON files use LPS coordinate system
2. Verify `ROTATION_YZX_DEG = [90, -90, 0]` in processing script
3. Ensure `(y, -z, -x)` transformation is applied in stEVE

### Issue: Mesh is too large/small

**Cause:** Wrong scaling factor

**Fix:**
1. Check `SCALING_FACTOR = 10.0` (converts cm to mm)
2. Verify VTP file units (VMR uses cm)
3. Adjust scaling if using different source data

### Issue: Radius values look wrong

**Cause:** Failed VMTK extraction or wrong units

**Fix:**
1. Re-run processing with fresh VMTK extraction
2. Verify VTP mesh quality (no holes, manifold)
3. Check radius range: should be 0.5-15 mm for neurovascular

### Issue: Missing branches

**Cause:** Incomplete centerline extraction

**Fix:**
1. Check if PTH files have all branches
2. Re-extract using 3D Slicer with manual endpoint selection
3. Verify all JSON files were created (check Centrelines folder)

## Visualization Options Summary

### `visualize_processed_model.py`

```bash
# Basic usage
python visualize_processed_model.py <model_folder>

# Options
python visualize_processed_model.py <model_folder> collision     # Collision mesh only
python visualize_processed_model.py <model_folder> visual        # Visual mesh only
python visualize_processed_model.py <model_folder> both          # Both meshes
python visualize_processed_model.py <model_folder> centerlines_only  # No mesh
```

### `visualize_with_comparison.py`

```bash
# Basic usage
python visualize_with_comparison.py <model_folder>

# Advanced options
--radius-colormap     # Color-code by radius
--no-mesh            # Centerlines only
--split-view         # Side-by-side mesh comparison
```

## Batch Visualization

To visualize multiple models quickly:

```bash
# Windows batch script
for /d %D in (D:\vmr\vmr\*) do (
    if exist "%D\dualdevicenav_format" (
        python vmr_processing_tools\visualize_processed_model.py "%D\dualdevicenav_format"
    )
)
```

## Export Screenshots

To save a screenshot from any visualization:

```python
# Add to the end of visualization script before plotter.show()
plotter.screenshot('model_0011_visualization.png')
plotter.show()
```

## Understanding the Color Coding

### Standard Visualization
- **Light Blue** = Collision mesh (semi-transparent)
- **Light Coral** = Visual mesh
- **Red, Green, Blue, Yellow, etc.** = Different centerline branches

### Radius Color-Coded Visualization
- **Blue** = Small radius (narrow vessels)
- **Green** = Medium radius
- **Yellow** = Medium-large radius
- **Red** = Large radius (wide vessels)

## Coordinate Systems

### LPS System (3D Slicer Standard)
- **L** (Left): Positive X → Patient's left
- **P** (Posterior): Positive Y → Patient's back
- **S** (Superior): Positive Z → Patient's head

### stEVE Transformation
stEVE applies `(y, -z, -x)` transformation on load:
- Original LPS → stEVE internal coordinates
- This is handled automatically in `DualDeviceNavCustom`

## Tips for Quality Assessment

1. **Visual Inspection Checklist**
   - [ ] Mesh is smooth and complete
   - [ ] Centerlines follow vessel centers
   - [ ] Radius varies smoothly along vessels
   - [ ] Branches extend to vessel ends
   - [ ] Coordinate axes make sense (aorta pointing up)

2. **Quantitative Checks**
   ```bash
   # Check radius statistics
   python vmr_processing_tools\verify_radius_data.py <model_folder>
   
   # Should show reasonable ranges:
   #   Min radius: 0.5-2 mm (small branches)
   #   Max radius: 8-15 mm (aorta)
   #   Mean radius: 3-6 mm
   ```

3. **Compare with Original**
   - Load original VTP in 3D Slicer
   - Load processed OBJ + JSON in PyVista
   - Verify they match

## Integration with stEVE

After verifying the visualization looks correct:

```python
# Test in stEVE
from eve_bench.dualdevicenav import DualDeviceNavCustom

intervention = DualDeviceNavCustom(
    mesh_folder=r"D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format",
    model_name="0011_H_AO_H"
)

# Run a quick test
intervention.reset()
# ... your testing code ...
```

If the visualization looks good but stEVE crashes:
- Check insertion point is inside the vessel
- Verify radius data exists in JSON files
- Test with original DualDeviceNav first

## Advanced: Custom Visualization

Create your own visualization script:

```python
import pyvista as pv
import json
import numpy as np

# Load mesh
mesh = pv.read("path/to/vessel_architecture_collision.obj")

# Load centerlines
with open("path/to/Centerline curve.mrk.json", 'r') as f:
    data = json.load(f)
    
points = []
for markup in data['markups']:
    for point in markup['controlPoints']:
        points.append(point['position'])

points = np.array(points)

# Visualize
plotter = pv.Plotter()
plotter.add_mesh(mesh, opacity=0.3, color='lightblue')
plotter.add_mesh(pv.PolyData(points), color='red', point_size=10)
plotter.show()
```

## Next Steps

After visualization confirms data quality:

1. ✅ Test with play script: `python eve_bench\example\dual_human_play_0011.py`
2. ✅ Train RL agent: `python training_scripts\DualDeviceNav_train_0011.py`
3. ✅ Process more models: `python vmr_processing_tools\batch_create_dualdevicenav.py`

## See Also

- `RADIUS_COMPLETE_GUIDE.md` - Understanding radius data
- `EXAMPLE_WORKFLOW.md` - Complete processing workflow
- `README.md` - Tools overview
- `eve_bench/example/README_CUSTOM_MODELS.md` - Using processed models in stEVE



