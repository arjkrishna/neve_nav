# VMR Download Tools

This folder contains scripts and data files for downloading and filtering vascular models from the **Vascular Model Repository (VMR)**.

---

## 📁 File Organization

### Python Scripts (7 files)
1. **`download_vmr_csv.py`** - Download VMR dataset catalog
2. **`extract_correct_vmr_names.py`** - Extract valid VMR model identifiers
3. **`vmr_models_list.py`** - Full list of VMR model names (generated)
4. **`filter_vmr_models.py`** - Filter models by criteria
5. **`vmr_filtered_models_list.py`** - Filtered list of models (generated)
6. **`download_vmr_models.py`** - Download complete VMR models
7. **`download_filtered_vmr.py`** - Download only filtered models

### Data Files (5 files)
1. **`dataset-svprojects.csv`** - Complete VMR catalog (234 KB, 707 entries)
2. **`vmr_model_names.txt`** - All available model names (317 models)
3. **`vmr_filtered_models.txt`** - Filtered model names (51 models)
4. **`vmr_filtered_models_details.csv`** - Detailed info for filtered models
5. **`vmr_download.log`** - Download progress log

---

## 🔄 Complete Workflow

### Step 1: Download VMR Catalog
```bash
python vmr_download_tools/download_vmr_csv.py
```
**Output:** `dataset-svprojects.csv`
- Downloads the complete VMR dataset catalog
- Contains metadata for all available models
- ~707 entries with anatomy type, mesh quality, etc.

---

### Step 2: Extract Model Names
```bash
python vmr_download_tools/extract_correct_vmr_names.py
```
**Output:** `vmr_model_names.txt`, `vmr_models_list.py`
- Parses the CSV to extract valid VMR model identifiers
- Creates both TXT (human-readable) and Python list format
- **317 total models** identified

---

### Step 3: Filter Models by Criteria
```bash
python vmr_download_tools/filter_vmr_models.py
```
**Output:** `vmr_filtered_models.txt`, `vmr_filtered_models_details.csv`, `vmr_filtered_models_list.py`

**Filtering Criteria:**
- ✅ Anatomy type: **Aortic Arch** or **Thoracic/Abdominal Aorta**
- ✅ Has **surface mesh** (VTP)
- ✅ Has **volume mesh** (VTU)
- ✅ Has **centerlines** (PTH)
- ✅ Complete data available

**Result:** **51 filtered models** meeting all criteria

---

### Step 4: Download Models

**Option A: Download All Models (317)**
```bash
python vmr_download_tools/download_vmr_models.py
```
⚠️ Warning: Downloads ~317 models (large download!)

**Option B: Download Filtered Models Only (51) - Recommended**
```bash
python vmr_download_tools/download_filtered_vmr.py
```
✅ Recommended: Downloads only high-quality aortic arch models

**Download Structure:**
```
vmr_downloads/
├── 0078_0001/
│   ├── Meshes/
│   │   ├── 0078_0001.vtp    (Surface mesh)
│   │   └── 0078_0001.vtu    (Volume mesh)
│   └── Paths/
│       ├── aorta.pth        (Centerline)
│       ├── lcca.pth
│       └── ...
├── 0079_0001/
├── 0094_0001/
└── ...
```

---

## 📊 Data File Details

### `dataset-svprojects.csv`
Complete VMR catalog downloaded from the repository.

**Key Columns:**
- `project_name`: Model identifier (e.g., "0166_0001")
- `anatomy`: Anatomy type (e.g., "aortic arch")
- `has_surface_mesh`: Boolean
- `has_volume_mesh`: Boolean
- `has_centerlines`: Boolean
- `mesh_quality`: Quality rating
- `download_url`: Direct download link

### `vmr_model_names.txt`
Plain text list of all 317 available model identifiers:
```
0001_0001
0002_0001
0003_0001
...
0317_0001
```

### `vmr_filtered_models.txt`
Filtered list of 51 high-quality aortic arch models:
```
0078_0001
0079_0001
0094_0001
0095_0001
0105_0001
...
```

### `vmr_filtered_models_details.csv`
Detailed information for the 51 filtered models:
```csv
model_id,anatomy,has_surface,has_volume,has_centerlines,quality
0078_0001,aortic arch,True,True,True,high
0079_0001,thoracic aorta,True,True,True,high
...
```

### `vmr_download.log`
Download progress log showing:
- Models being downloaded
- Success/failure status
- File sizes
- Download timestamps
- Error messages (if any)

---

## 🎯 Use Cases

### Use Case 1: Download for stEVE Training
```bash
# Download filtered aortic arch models
python vmr_download_tools/download_filtered_vmr.py

# Models used in stEVE:
# - 0011_H_AO_H: Used in BasicWireNav
# - 0105_0001: Likely base for DualDeviceNav
# - 0166_0001: Example for mesh generation
```

### Use Case 2: Statistical Analysis of Vessel Geometry
```bash
# Download all aortic arch models
python vmr_download_tools/download_filtered_vmr.py

# Then extract centerlines in 3D Slicer
# Analyze geometry statistics
# Learn parameters for CHS generation
```

### Use Case 3: Custom Filtering
Edit `filter_vmr_models.py` to change criteria:
```python
# Example: Filter for cerebrovascular models
def filter_criteria(row):
    return (
        'cerebro' in row['anatomy'].lower() and
        row['has_surface_mesh'] == True and
        row['has_centerlines'] == True
    )
```

---

## 🔧 Script Details

### 1. `download_vmr_csv.py`
**Purpose:** Download the VMR dataset catalog

**Usage:**
```bash
python vmr_download_tools/download_vmr_csv.py
```

**What it does:**
- Fetches the complete VMR catalog CSV
- Saves to `dataset-svprojects.csv`
- ~707 entries with metadata

**Output:** `dataset-svprojects.csv`

---

### 2. `extract_correct_vmr_names.py`
**Purpose:** Extract valid model identifiers from catalog

**Usage:**
```bash
python vmr_download_tools/extract_correct_vmr_names.py
```

**What it does:**
- Reads `dataset-svprojects.csv`
- Extracts model IDs in correct format
- Creates Python list for scripting
- Creates TXT file for human reference

**Output:**
- `vmr_model_names.txt` (317 models)
- `vmr_models_list.py` (Python format)

---

### 3. `filter_vmr_models.py`
**Purpose:** Filter models by anatomy type and completeness

**Usage:**
```bash
python vmr_download_tools/filter_vmr_models.py
```

**Filtering Logic:**
```python
# Checks for:
- Aortic arch or thoracic/abdominal aorta anatomy
- Has VTP surface mesh
- Has VTU volume mesh  
- Has PTH centerlines
- Complete dataset
```

**Output:**
- `vmr_filtered_models.txt` (51 models)
- `vmr_filtered_models_details.csv` (detailed info)
- `vmr_filtered_models_list.py` (Python format)

---

### 4. `download_vmr_models.py`
**Purpose:** Download all VMR models

**Usage:**
```bash
python vmr_download_tools/download_vmr_models.py
```

**Features:**
- Downloads all 317 models
- Creates organized folder structure
- Logs progress
- Resumes on interruption
- Parallel downloads (configurable)

**⚠️ Warning:** Large download (~several GB)

---

### 5. `download_filtered_vmr.py`
**Purpose:** Download only filtered models (recommended)

**Usage:**
```bash
python vmr_download_tools/download_filtered_vmr.py
```

**Features:**
- Downloads only 51 high-quality models
- Faster than downloading all
- Focused on aortic arch anatomy
- Same features as `download_vmr_models.py`

**✅ Recommended** for most use cases

---

### 6. `vmr_models_list.py`
**Purpose:** Python list of all 317 model names (generated file)

**Usage:**
```python
from vmr_download_tools.vmr_models_list import vmr_models

print(f"Total models: {len(vmr_models)}")
for model in vmr_models[:5]:
    print(model)
```

---

### 7. `vmr_filtered_models_list.py`
**Purpose:** Python list of 51 filtered models (generated file)

**Usage:**
```python
from vmr_download_tools.vmr_filtered_models_list import filtered_models

print(f"Filtered models: {len(filtered_models)}")
for model in filtered_models:
    print(model)
```

---

## 📈 Statistics

- **Total VMR Catalog Entries:** 707
- **Valid Model IDs:** 317
- **Filtered Aortic Arch Models:** 51
- **Models Used in stEVE:** 12 (originally specified in `eve/vmrfiledownload.py`)

### Filtered Models List (51 total):
```
0078_0001  0079_0001  0094_0001  0095_0001  0105_0001
0111_0001  0131_0000  0154_0001  0166_0001  0167_0001
0175_0000  0176_0000  ... (39 more)
```

### Originally Specified in stEVE (12):
```
0078_0001  0079_0001  0094_0001  0095_0001  0105_0001  0111_0001
0131_0000  0154_0001  0166_0001  0167_0001  0175_0000  0176_0000
```

---

## 🔗 Integration with stEVE

### Download for Training:
```bash
# Download filtered models
python vmr_download_tools/download_filtered_vmr.py

# Use in stEVE
# Edit eve/vmrfiledownload.py to add new models
```

### Extract Centerlines:
```bash
# Use visualization tools to prepare for Slicer
python visualization_tools/convert_for_slicer.py vmr_downloads/0166_0001/ --scene

# Extract centerlines in 3D Slicer with VMTK
# Export as .mrk.json format

# Use for CHS parameter learning!
```

---

## 🛠️ Troubleshooting

### Issue: Download fails
**Solution:** Check internet connection, retry with `download_filtered_vmr.py`

### Issue: CSV file not found
**Solution:** Run `download_vmr_csv.py` first

### Issue: No models in filtered list
**Solution:** Check filtering criteria in `filter_vmr_models.py`

### Issue: Incomplete downloads
**Solution:** Scripts support resuming - just run again

---

## 📝 Notes

- All scripts log to `vmr_download.log`
- Downloads go to `../vmr_downloads/` (project root)
- Scripts are idempotent (safe to run multiple times)
- Use filtered download for most efficient workflow

---

## 🎓 Related Resources

- **VMR Website:** http://www.vascularmodel.org/
- **stEVE VMR Module:** `../eve/eve/intervention/vesseltree/vmr.py`
- **Visualization Tools:** `../visualization_tools/`
- **Example Usage:** `../eve_bench/eve_bench/basicwirenav.py` (uses VMR 0011)

---

## 🚀 Quick Start

**Complete workflow from scratch:**
```bash
# 1. Download filtered models (recommended)
python vmr_download_tools/download_filtered_vmr.py

# 2. Visualize a model
python visualization_tools/visualize_vmr_interactive.py vmr_downloads/0166_0001/

# 3. Prepare for 3D Slicer
python visualization_tools/convert_for_slicer.py vmr_downloads/0166_0001/ --scene

# 4. Open in Slicer, extract centerlines with VMTK

# 5. Use centerlines for CHS parameter learning!
```

---

**Last Updated:** November 10, 2025

