# Which Visualization Script Should I Use?

## Quick Answer

**To verify your processed data will work in stEVE:**
```bash
python vmr_processing_tools\visualize_as_steve_sees_it.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
```

**To verify centerline extraction was correct:**
```bash
python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H
```

---

## The Three Visualization Scripts

### 1. `visualize_as_steve_sees_it.py` ⭐ RECOMMENDED

**What it shows:** The processed OBJ mesh with centerlines transformed exactly as stEVE loads them

**When to use:** 
- ✅ After processing is complete
- ✅ To verify mesh and centerlines align correctly
- ✅ Before testing in stEVE
- ✅ To debug coordinate transformation issues

**Input:** `dualdevicenav_format` folder

**What to look for:**
- Centerlines should be INSIDE the mesh
- Alignment check should show `[OK]`
- Mesh and centerlines should overlap perfectly

**Example:**
```bash
python vmr_processing_tools\visualize_as_steve_sees_it.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
```

---

### 2. `visualize_original_vs_processed.py`

**What it shows:** The ORIGINAL VTP mesh with extracted centerlines (before transformations)

**When to use:**
- ✅ To verify VMTK centerline extraction was successful
- ✅ To check if centerlines follow the vessel centers
- ✅ To verify radius values make sense
- ✅ To debug extraction issues

**Input:** VMR model folder (parent of dualdevicenav_format)

**What to look for:**
- Centerlines should follow vessel centers in the VTP mesh
- Radius values should vary smoothly (no sudden jumps)
- All branches should be present

**Example:**
```bash
# Show original VTP + centerlines
python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H

# Compare original VTP vs processed OBJ side-by-side
python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H compare
```

---

### 3. `visualize_processed_model.py` ⚠️ INCORRECT - DON'T USE

**Why it's wrong:** Shows transformed OBJ with un-transformed centerlines - they won't align!

**Problem:** 
- OBJ mesh has rotations, scaling, flipped normals applied
- JSON centerlines are in original LPS coordinates
- They appear misaligned even though the data is correct

**Status:** This was the first script created - use the other two instead!

---

## Typical Workflow

### Step 1: Process the model
```bash
vmr_processing_tools\run_dualdevicenav.bat
```

### Step 2: Verify centerline extraction
```bash
python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H
```

**What to check:**
- ✅ Centerlines are inside the VTP mesh
- ✅ They follow vessel centers
- ✅ Radius values are reasonable (0.5-15 mm)
- ✅ All branches are present

### Step 3: Verify stEVE compatibility
```bash
python vmr_processing_tools\visualize_as_steve_sees_it.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
```

**What to check:**
- ✅ Centerlines are inside the OBJ mesh
- ✅ Alignment check shows `[OK]`
- ✅ Mesh shape looks correct

### Step 4: Test in stEVE
```bash
python eve_bench\example\dual_human_play_0011.py
```

---

## Troubleshooting Visualizations

### Problem: "Centerlines are outside the mesh"

**In `visualize_original_vs_processed.py`:**
- ❌ VMTK extraction failed or used wrong endpoints
- Fix: Re-run processing or use 3D Slicer for manual extraction

**In `visualize_as_steve_sees_it.py`:**
- ❌ Coordinate transformation is wrong
- Fix: Check that transformations match DualDeviceNav settings
  - Rotation: [90, -90, 0]
  - Scaling: 10x
  - JSON transform: (y, -z, -x)

### Problem: "Mesh looks weird/deformed"

**In `visualize_original_vs_processed.py`:**
- If VTP looks wrong: Source data problem
- If both VTP and OBJ look wrong: Not a processing issue

**In `visualize_as_steve_sees_it.py`:**
- This is the TRANSFORMED mesh (rotated, scaled)
- It should look different from the original VTP
- Compare side-by-side using `visualize_original_vs_processed.py compare`

### Problem: "Alignment check shows [WARNING]"

**Causes:**
1. Wrong rotation angles
2. Wrong scaling factor
3. Wrong coordinate transformation in JSON loading
4. Source VTP and PTH files don't match

**Fix:**
1. Verify processing parameters match DualDeviceNav
2. Check that centerlines were extracted from the same VTP file
3. Try processing again with fresh VMTK extraction

---

## Understanding the Transformations

### Original VTP → Processed OBJ

The processing script applies:
1. **Flip normals** (for SOFA compatibility)
2. **Scale 10x** (convert cm → mm)
3. **Rotate [90, -90, 0]** (align coordinate systems)
4. **Decimate 0.9** (reduce mesh complexity)

### Original JSON → stEVE Coordinates

When stEVE loads JSON:
1. **Read LPS coordinates** from JSON file
2. **Apply (y, -z, -x)** transformation
3. **Match processed OBJ coordinates**

### Why Two Different Approaches?

- **OBJ mesh:** Transformed once during processing (saved as file)
- **JSON centerlines:** Transformed on-the-fly during loading (more flexible)

Both end up in the same coordinate system in stEVE!

---

## Quick Reference

| Script | Input | Shows | Use When |
|--------|-------|-------|----------|
| `visualize_as_steve_sees_it.py` | `dualdevicenav_format/` | Transformed data (as stEVE sees it) | Final verification ✅ |
| `visualize_original_vs_processed.py` | `0011_H_AO_H/` | Original VTP + centerlines | Check extraction ✅ |
| `visualize_processed_model.py` | `dualdevicenav_format/` | Misaligned (incorrect!) | Don't use ❌ |

---

## Examples for Model 0011

```bash
# 1. Verify extraction was successful
python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H

# 2. Verify stEVE compatibility
python vmr_processing_tools\visualize_as_steve_sees_it.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format

# 3. Compare original vs processed
python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H compare

# 4. Test in stEVE
python eve_bench\example\dual_human_play_0011.py
```

---

## See Also

- `VISUALIZATION_GUIDE.md` - Detailed visualization guide
- `EXAMPLE_WORKFLOW.md` - Complete processing workflow
- `RADIUS_COMPLETE_GUIDE.md` - Understanding radius data



