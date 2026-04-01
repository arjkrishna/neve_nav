# Visualization Tools for stEVE and VMR Data

This folder contains utility scripts for visualizing and processing VTP/VTU mesh files from VMR (Vascular Model Repository) and other sources.

## Scripts Overview

### 1. `visualize_mesh.py`
**Purpose:** Quick visualization of individual VTP/VTU files using PyVista

**Usage:**
```bash
# View a single mesh
python visualize_mesh.py path/to/file.vtp

# View with edges visible
python visualize_mesh.py path/to/file.vtp --edges

# Compare multiple meshes side-by-side
python visualize_mesh.py file1.vtp file2.vtp file3.vtp

# Save a screenshot
python visualize_mesh.py path/to/file.vtp --screenshot
```

**Features:**
- Displays mesh geometry (surface or volume)
- Shows mesh statistics (points, cells, bounds)
- Visualizes scalar data arrays (radius, curvature, etc.)
- Creates tubes along centerlines if radius data exists
- Interactive 3D rotation, zoom, pan

---

### 2. `visualize_vmr_interactive.py`
**Purpose:** Complete VMR model visualization with surface mesh, volume mesh, and centerlines

**Usage:**
```bash
# View complete VMR model (VTP + centerlines)
python visualize_vmr_interactive.py vmr_downloads/0166_0001/

# Include volume mesh (VTU) - slower but more complete
python visualize_vmr_interactive.py vmr_downloads/0166_0001/ --volume

# Quick view of a single file
python visualize_vmr_interactive.py vmr_downloads/0166_0001/Meshes/0166_0001.vtp
```

**Features:**
- Loads VTP surface meshes (semi-transparent red)
- Optionally loads VTU volume meshes (wireframe)
- Parses and displays all centerline paths from PTH files (colored lines)
- Automatic scaling from cm to mm
- Legend with all components
- Interactive coordinate axes

---

### 3. `convert_for_slicer.py`
**Purpose:** Convert and optimize VTP/VTU files for 3D Slicer visualization

**Usage:**
```bash
# Convert a single file
python convert_for_slicer.py path/to/file.vtp

# Convert all files in a VMR folder
python convert_for_slicer.py vmr_downloads/0166_0001/ --folder

# Create a 3D Slicer scene file (.mrml) for easy loading
python convert_for_slicer.py vmr_downloads/0166_0001/ --scene
```

**Features:**
- Automatic scaling from cm to mm (if needed)
- Mesh cleaning and optimization
- Batch conversion of entire VMR folders
- Creates Slicer scene files (.mrml) for one-click opening
- Ensures compatibility with 3D Slicer

**Output:**
- Converted files saved to `vmr_folder/slicer/`
- Scene files for direct opening in 3D Slicer

---

## Requirements

### Python Dependencies:
```bash
pip install pyvista numpy
```

### For 3D Slicer:
- Download from: https://www.slicer.org/
- Install SlicerVMTK extension for centerline extraction

---

## Typical Workflows

### Workflow 1: Quick Mesh Inspection
```bash
# Quick view of a downloaded VMR model
python visualize_vmr_interactive.py vmr_downloads/0105_0001/
```

### Workflow 2: Prepare for 3D Slicer Analysis
```bash
# Convert VMR model for Slicer
python convert_for_slicer.py vmr_downloads/0105_0001/ --scene

# Then open the generated .mrml file in 3D Slicer
# Use SlicerVMTK to extract centerlines
# Export as .mrk.json for use in stEVE
```

### Workflow 3: Compare Multiple Patients
```bash
# Compare surface meshes from different patients
python visualize_mesh.py \
    vmr_downloads/0105_0001/Meshes/0105_0001.vtp \
    vmr_downloads/0166_0001/Meshes/0166_0001.vtp \
    vmr_downloads/0078_0001/Meshes/0078_0001.vtp
```

---

## Integration with stEVE

These tools help prepare data for use in stEVE training:

1. **Visualize VMR models** to understand anatomy
2. **Convert for 3D Slicer** to extract centerlines
3. **Export centerlines as JSON** (like DualDeviceNav data)
4. **Learn CHS statistics** from multiple patient centerlines
5. **Generate procedural vessels** with data-driven parameters

---

## Tips

### PyVista Controls:
- **Left Mouse**: Rotate view
- **Middle Mouse / Scroll**: Zoom
- **Right Mouse**: Pan
- **Q**: Quit

### 3D Slicer:
- Use **SlicerVMTK** extension for centerline extraction
- Extract centerlines using "Extract Centerline" module
- Export as `.mrk.json` format for compatibility with stEVE

### Performance:
- VTP (surface) files are fast to visualize
- VTU (volume) files are slower - use `--volume` flag only when needed
- For large datasets, use `convert_for_slicer.py` to optimize first

---

## Examples

### Example 1: Visualize VMR 0166
```bash
python visualize_vmr_interactive.py vmr_downloads/0166_0001/
```
Output: Interactive 3D view with surface mesh and all centerline branches

### Example 2: Prepare for Centerline Extraction
```bash
python convert_for_slicer.py vmr_downloads/0105_0001/ --scene
# Opens in Slicer → Use SlicerVMTK → Extract centerlines → Export JSON
```

### Example 3: Batch Visualization
```bash
# In Python
import pyvista as pv
from pathlib import Path

vtp_files = list(Path("vmr_downloads").rglob("*.vtp"))
for vtp_file in vtp_files[:5]:  # View first 5
    mesh = pv.read(str(vtp_file))
    mesh.plot(title=vtp_file.name)
```

---

## Troubleshooting

**Import Error: No module named 'pyvista'**
```bash
pip install pyvista
```

**Mesh appears too small/large:**
- Scripts auto-detect scale (cm vs mm)
- Manually adjust in 3D Slicer if needed

**Can't see centerlines in visualize_vmr_interactive.py:**
- Ensure PTH files exist in `vmr_folder/Paths/`
- Check that PTH files are valid XML format

**3D Slicer won't open file:**
- Use `convert_for_slicer.py` first to ensure compatibility
- Check file isn't corrupted

---

## Related stEVE Files

- **`eve/eve/intervention/vesseltree/vmr.py`**: Loads VMR data in stEVE
- **`eve/eve/intervention/vesseltree/frommesh.py`**: Loads OBJ/VTP meshes
- **`eve_bench/data/dualdevicenav/`**: Example centerline JSON files
- **`eve/eve/intervention/vesseltree/util/cubichermitesplines.py`**: Procedural generation

---

## Contact & Contribution

These scripts were generated as part of the stEVE training project for:
- Visualizing VMR vascular models
- Preparing data for centerline extraction
- Learning statistical parameters for procedural generation

Feel free to modify and extend for your specific needs!


