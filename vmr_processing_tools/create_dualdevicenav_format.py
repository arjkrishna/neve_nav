"""
Create DualDeviceNav-Compatible Data from VMR Models

This script replicates the EXACT processing pipeline used for DualDeviceNav model 0105:
1. VTP mesh → OBJ files (collision + visual) with correct transformations
2. VTP mesh → VMTK centerline extraction with radius
3. Centerlines → DualDeviceNav JSON format

Usage:
    python create_dualdevicenav_format.py <vmr_model_folder> [--output <output_folder>]

Example:
    python create_dualdevicenav_format.py vmr_downloads/0105_0001/
    python create_dualdevicenav_format.py vmr_downloads/0105_0001/ --output processed_models/model_0105/
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista not installed!")
    print("Install with: pip install pyvista")
    sys.exit(1)

try:
    from vmtk import vmtkscripts
except ImportError:
    print("ERROR: VMTK not installed!")
    print("Install with: conda install -c vmtk vmtk")
    sys.exit(1)


# DualDeviceNav settings (from eve_bench/eve_bench/dualdevicenav.py)
ROTATION_YZX_DEG = [-90, 0, 90]  # Y, Z, X rotation in degrees
SCALING_FACTOR = 10.0  # cm to mm
COORDINATE_SYSTEM = "LPS"  # Left-Posterior-Superior (3D Slicer standard)


def find_vtp_file(folder: str) -> Optional[str]:
    """Find VTP file in VMR model folder"""
    meshes_folder = os.path.join(folder, "Meshes")
    if not os.path.exists(meshes_folder):
        return None
    
    for file in os.listdir(meshes_folder):
        if file.endswith(".vtp"):
            return os.path.join(meshes_folder, file)
    return None


def create_obj_mesh(vtp_file: str, output_obj: str, decimation_factor: float = 0.9):
    """
    Create OBJ mesh from VTP with DualDeviceNav transformations
    
    Args:
        vtp_file: Input VTP mesh file
        output_obj: Output OBJ file path
        decimation_factor: Mesh decimation factor (0.9 = keep 10% of faces)
    """
    print(f"\n{'='*70}")
    print(f"Creating OBJ Mesh: {os.path.basename(output_obj)}")
    print(f"{'='*70}")
    
    # Load VTP mesh
    print("1. Loading VTP mesh...")
    mesh = pv.read(vtp_file)
    print(f"   Original: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    
    # Flip normals (standard for SOFA)
    print("2. Flipping normals...")
    mesh.flip_normals()
    
    # Scale from cm to mm
    print(f"3. Scaling by {SCALING_FACTOR}x (cm -> mm)...")
    mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
    
    # Apply rotations (Y, Z, X order)
    print(f"4. Applying rotations {ROTATION_YZX_DEG}...")
    mesh.rotate_y(ROTATION_YZX_DEG[0], inplace=True)
    mesh.rotate_z(ROTATION_YZX_DEG[1], inplace=True)
    mesh.rotate_x(ROTATION_YZX_DEG[2], inplace=True)
    
    # Decimate mesh
    if decimation_factor > 0:
        print(f"5. Decimating mesh (factor={decimation_factor})...")
        mesh.decimate(decimation_factor, inplace=True)
        print(f"   Decimated: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    
    # Save as OBJ
    print(f"6. Saving OBJ file...")
    pv.save_meshio(output_obj, mesh)
    print(f"   [OK] Saved: {output_obj}")
    
    return mesh


def extract_centerlines_vmtk(vtp_file: str, 
                             source_points: Optional[List[Tuple[float, float, float]]] = None,
                             target_points: Optional[List[Tuple[float, float, float]]] = None,
                             resample_step: float = 1.0) -> Optional[object]:
    """
    Extract centerlines with radius using VMTK
    
    Args:
        vtp_file: Input VTP surface mesh
        source_points: List of source point coordinates [(x,y,z), ...]
        target_points: List of target point coordinates [(x,y,z), ...]
        resample_step: Resampling step length in mm (default: 1.0mm)
    
    Returns:
        VMTK centerlines object with radius information
    """
    print(f"\n{'='*70}")
    print(f"Extracting Centerlines with VMTK")
    print(f"{'='*70}")
    
    # Read surface
    print("\n1. Reading surface mesh...")
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = vtp_file
    reader.Execute()
    surface = reader.Surface
    print(f"   Surface: {surface.GetNumberOfPoints():,} points, {surface.GetNumberOfCells():,} cells")
    
    # Compute centerlines
    print("\n2. Computing centerlines with radius...")
    centerlineFilter = vmtkscripts.vmtkCenterlines()
    centerlineFilter.Surface = surface
    
    if source_points and target_points:
        print(f"   Using {len(source_points)} source points and {len(target_points)} target points")
        centerlineFilter.SeedSelectorName = 'pointlist'
        
        # Flatten lists
        source_flat = [coord for point in source_points for coord in point]
        target_flat = [coord for point in target_points for coord in point]
        
        centerlineFilter.SourcePoints = source_flat
        centerlineFilter.TargetPoints = target_flat
    else:
        print("   [!] No endpoints specified - using automatic detection")
        print("   Note: For best results, specify source and target points manually")
        centerlineFilter.SeedSelectorName = 'openprofiles'
    
    centerlineFilter.AppendEndPoints = 1
    centerlineFilter.Resampling = 1
    centerlineFilter.ResamplingStepLength = resample_step
    
    try:
        centerlineFilter.Execute()
        centerlines = centerlineFilter.Centerlines
        
        if centerlines.GetNumberOfPoints() == 0:
            print("   [ERROR] No centerlines extracted!")
            return None
        
        print(f"   [OK] Extracted {centerlines.GetNumberOfPoints():,} centerline points")
        print(f"   [OK] {centerlines.GetNumberOfCells()} centerline paths")
        
        return centerlines
        
    except Exception as e:
        print(f"   [ERROR] Centerline extraction failed: {str(e)}")
        return None


def convert_vmtk_to_dualdevicenav_json(centerlines, 
                                        output_folder: str,
                                        model_name: str = "centerline",
                                        scaling_factor: float = 10.0):
    """
    Convert VMTK centerlines to DualDeviceNav JSON format
    
    Creates SEPARATE JSON files for each branch (matching DualDeviceNav format).
    
    The JSON format matches exactly what DualDeviceNav expects:
    - Each branch = separate JSON file
    - Coordinates in LPS system (NOT transformed yet)
    - Radius measurements included
    - 3D Slicer markup format
    
    Transformation (y, -z, -x) is applied by stEVE when loading!
    
    Args:
        centerlines: VMTK centerlines object
        output_folder: Output folder for JSON files
        model_name: Base name for the centerline files
        scaling_factor: Scale factor to match mesh (default 10.0 for cm->mm)
    """
    print(f"\n{'='*70}")
    print(f"Converting to DualDeviceNav JSON Format")
    print(f"{'='*70}")
    
    # Extract radius array
    radius_array = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius_array is None:
        print("[ERROR] No radius information in centerlines!")
        return False
    
    # CRITICAL: VMTK extracts centerlines in VTP's original units (cm)
    # We need to scale coordinates AND radii to match the scaled mesh (mm)
    print(f"   Applying scaling factor: {scaling_factor}x (cm -> mm)")
    
    # Group points by cell (each cell is a branch)
    branches = []
    for cell_id in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(cell_id)
        points = []
        radii = []
        
        for i in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(i)
            point = centerlines.GetPoint(point_id)
            radius = radius_array.GetValue(point_id)
            
            # SCALE and ROTATE coordinates to match mesh
            # 1. Scale from cm to mm
            scaled_point = [point[0] * scaling_factor, 
                           point[1] * scaling_factor, 
                           point[2] * scaling_factor]
            
            # 2. Apply rotation [90, -90, 0] to match OBJ mesh rotation
            point_mesh = pv.PolyData([scaled_point])
            point_mesh.rotate_y(ROTATION_YZX_DEG[0], inplace=True)
            point_mesh.rotate_z(ROTATION_YZX_DEG[1], inplace=True)
            point_mesh.rotate_x(ROTATION_YZX_DEG[2], inplace=True)
            rotated_point = point_mesh.points[0]
            
            points.append([rotated_point[0], rotated_point[1], rotated_point[2]])
            radii.append(radius * scaling_factor)
        
        branches.append({"points": points, "radii": radii})
    
    # CRITICAL: Sort branches by length (longest first)
    # This ensures the main aorta/trunk is branch 0 for insertion point detection
    branches.sort(key=lambda b: len(b['points']), reverse=True)
    
    print(f"   Found {len(branches)} branch(es) (sorted by length - longest first)")
    for i, branch in enumerate(branches):
        print(f"   Branch {i}: {len(branch['points'])} points, "
              f"radius {min(branch['radii']):.2f}-{max(branch['radii']):.2f}mm")
    
    # Save each branch as a separate JSON file (like DualDeviceNav)
    saved_files = []
    total_points = 0
    
    for branch_idx, branch in enumerate(branches):
        branch_points = branch["points"]
        branch_radii = branch["radii"]
        
        # Create filename: "Centerline curve (0).mrk.json", "Centerline curve (1).mrk.json", etc.
        if branch_idx == 0:
            filename = f"Centerline curve - {model_name}.mrk.json"
        else:
            filename = f"Centerline curve ({branch_idx}).mrk.json"
        
        output_file = os.path.join(output_folder, filename)
        
        # Create DualDeviceNav-compatible JSON (exact format from existing files)
        slicer_json = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Curve",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": False,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": len(branch_points),
                    "controlPoints": [
                        {
                            "id": str(i+1),
                            "label": f"{model_name}-{branch_idx}-{i+1}",
                            "description": "",
                            "associatedNodeID": "",
                            "position": point,
                            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                            "selected": True,
                            "locked": False,
                            "visibility": False,
                            "positionStatus": "defined"
                        }
                        for i, point in enumerate(branch_points)
                    ],
                    "measurements": [
                        {
                            "name": "length",
                            "enabled": False,
                            "units": "mm",
                            "printFormat": "%-#4.4g%s"
                        },
                        {
                            "name": "curvature mean",
                            "enabled": False,
                            "printFormat": "%5.3f %s"
                        },
                        {
                            "name": "curvature max",
                            "enabled": False,
                            "printFormat": "%5.3f %s"
                        },
                        {
                            "name": "Radius",
                            "enabled": True,
                            "units": "mm",
                            "controlPointValues": branch_radii
                        }
                    ],
                    "display": {
                        "visibility": True,
                        "opacity": 1.0,
                        "color": [0.4, 1.0, 0.0],
                        "selectedColor": [1.0, 0.5, 0.5],
                        "propertiesLabelVisibility": False,
                        "pointLabelsVisibility": False,
                        "textScale": 3.0,
                        "glyphType": "Sphere3D",
                        "glyphScale": 1.0,
                        "glyphSize": 5.0,
                        "useGlyphScale": True,
                        "lineThickness": 0.2,
                        "lineColorFadingStart": 1.0,
                        "lineColorFadingEnd": 10.0,
                    }
                }
            ]
        }
        
        # Save JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(slicer_json, f, indent=4)
        
        saved_files.append(filename)
        total_points += len(branch_points)
        print(f"   [OK] Saved branch {branch_idx+1}: {filename} ({len(branch_points)} points)")
    
    print(f"\n   [SUCCESS] Created {len(saved_files)} JSON files with {total_points} total points")
    print(f"   [SUCCESS] Output folder: {output_folder}")
    print(f"\n   NOTE: Coordinates in LPS (transformation (y,-z,-x) applied by stEVE on load)")
    
    return True


def extract_endpoints_from_pth(vmr_folder: str):
    """
    Extract start and end points from PTH files in VMR data
    
    FIXED: Use a single common source point (aortic root) and only the
    actual branch endpoints as targets to avoid phantom connections.
    
    VMR PTH files have invalid XML (multiple root elements), so we need to
    read the file as text and parse manually.
    
    Returns:
        tuple: (source_points, target_points) - lists of 3D coordinates
    """
    pth_folder = os.path.join(vmr_folder, "Paths")
    if not os.path.exists(pth_folder):
        print(f"   [WARNING] No Paths folder found at {pth_folder}")
        return None, None
    
    all_first_points = []
    all_last_points = []
    
    pth_files = list(Path(pth_folder).glob("*.pth"))
    if not pth_files:
        print(f"   [WARNING] No PTH files found in {pth_folder}")
        return None, None
    
    print(f"   Found {len(pth_files)} PTH files")
    
    for pth_file in pth_files:
        try:
            # Read file and wrap in a single root element to fix XML
            with open(pth_file, 'r') as f:
                content = f.read()
            
            # Remove XML declaration if present (causes issues when wrapping)
            import re
            content = re.sub(r'<\?xml[^?]*\?>\s*', '', content)
            
            # Wrap the content in a root element
            wrapped_xml = f"<root>{content}</root>"
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(wrapped_xml)
            
            # Find path_points in the path element
            path_points = root.findall('.//path_point')
            if not path_points:
                print(f"   [WARNING] No path_points found in {pth_file.name}")
                continue
            
            points = []
            for path_point in path_points:
                pos = path_point.find('pos')
                if pos is not None:
                    x = float(pos.get('x', 0))
                    y = float(pos.get('y', 0))
                    z = float(pos.get('z', 0))
                    points.append((x, y, z))
            
            if len(points) >= 2:
                all_first_points.append(points[0])
                all_last_points.append(points[-1])
                print(f"   [OK] Extracted endpoints from {pth_file.name}: {len(points)} points")
            else:
                print(f"   [WARNING] Not enough points in {pth_file.name}")
                
        except Exception as e:
            print(f"   [WARNING] Could not parse {pth_file.name}: {e}")
            continue
    
    if not all_first_points or not all_last_points:
        return None, None
    
    # Find the common source point (aortic root) - the point that appears most
    # often as a first point, or is closest to multiple paths
    # Strategy: Use the first point that's closest to the geometric center of all first points
    import numpy as np
    first_points_array = np.array(all_first_points)
    center = np.mean(first_points_array, axis=0)
    
    # Find the first point closest to center (likely the aortic root/common source)
    distances_to_center = np.linalg.norm(first_points_array - center, axis=1)
    common_source_idx = np.argmin(distances_to_center)
    common_source = all_first_points[common_source_idx]
    
    # Use ONE source point (the common source)
    source_points = [common_source]
    
    # Use ONLY the branch endpoints as targets (last points)
    target_points = all_last_points
    
    print(f"\n   [INFO] Using SINGLE source point (common root): {common_source}")
    print(f"   [INFO] Using {len(target_points)} target points (branch endpoints)")
    print(f"   [SUCCESS] This will create {len(target_points)} branches, not {len(all_first_points)*len(all_last_points)}!")
    
    return source_points, target_points


def process_vmr_to_dualdevicenav(vmr_folder: str, 
                                  output_folder: Optional[str] = None,
                                  create_visual_mesh: bool = True,
                                  source_points: Optional[List[Tuple]] = None,
                                  target_points: Optional[List[Tuple]] = None):
    """
    Complete pipeline: VMR model → DualDeviceNav format
    
    Args:
        vmr_folder: Path to VMR model folder
        output_folder: Output folder (default: vmr_folder/dualdevicenav_format/)
        create_visual_mesh: Whether to create separate visual mesh
        source_points: Source points for centerline extraction
        target_points: Target points for centerline extraction
    """
    model_name = os.path.basename(vmr_folder.rstrip('/\\'))
    print(f"\n{'='*70}")
    print(f"Processing VMR Model: {model_name}")
    print(f"Creating DualDeviceNav-Compatible Data")
    print(f"{'='*70}")
    
    # Setup output folder
    if output_folder is None:
        output_folder = os.path.join(vmr_folder, 'dualdevicenav_format')
    os.makedirs(output_folder, exist_ok=True)
    
    centerlines_folder = os.path.join(output_folder, 'Centrelines')
    os.makedirs(centerlines_folder, exist_ok=True)
    
    # Find VTP file
    vtp_file = find_vtp_file(vmr_folder)
    if not vtp_file:
        print(f"[ERROR] No VTP file found in {vmr_folder}")
        return False
    
    print(f"\nInput: {vtp_file}")
    print(f"Output: {output_folder}")
    
    # Auto-detect endpoints from PTH files if not provided
    if source_points is None or target_points is None:
        print(f"\n{'='*70}")
        print("Auto-detecting endpoints from PTH files...")
        print(f"{'='*70}")
        auto_source, auto_target = extract_endpoints_from_pth(vmr_folder)
        if auto_source and auto_target:
            source_points = auto_source
            target_points = auto_target
        else:
            print(f"\n   [ERROR] Could not extract endpoints from PTH files!")
            print(f"   [ERROR] Automatic detection requires user interaction.")
            print(f"   [ERROR] Please provide source_points and target_points manually.")
            return False
    
    # Step 1: Create collision mesh OBJ
    collision_obj = os.path.join(output_folder, f"{model_name}_collision.obj")
    try:
        create_obj_mesh(vtp_file, collision_obj, decimation_factor=0.9)
    except Exception as e:
        print(f"[ERROR] Error creating collision mesh: {e}")
        return False
    
    # Step 2: Create visual mesh OBJ (optional, less decimation)
    if create_visual_mesh:
        visual_obj = os.path.join(output_folder, f"{model_name}_visual.obj")
        try:
            create_obj_mesh(vtp_file, visual_obj, decimation_factor=0.8)
        except Exception as e:
            print(f"[WARNING] Visual mesh creation failed: {e}")
    
    # Step 3: Extract centerlines with VMTK
    try:
        centerlines = extract_centerlines_vmtk(vtp_file, source_points, target_points)
        if centerlines is None:
            print(f"[ERROR] Centerline extraction failed")
            return False
    except Exception as e:
        print(f"[ERROR] Error during centerline extraction: {e}")
        return False
    
    # Step 4: Convert to DualDeviceNav JSON format (separate files per branch)
    try:
        success = convert_vmtk_to_dualdevicenav_json(centerlines, centerlines_folder, model_name, SCALING_FACTOR)
        if not success:
            return False
    except Exception as e:
        print(f"[ERROR] Error creating JSON: {e}")
        return False
    
    # Summary
    print(f"\n{'='*70}")
    print(f"[SUCCESS] DualDeviceNav-Compatible Data Created")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Collision mesh: {collision_obj}")
    if create_visual_mesh:
        print(f"  Visual mesh:    {visual_obj}")
    print(f"  Centerlines:    {centerlines_folder}/ (multiple JSON files)")
    print(f"\nTo use in stEVE:")
    print(f"  1. Copy files to your training data folder")
    print(f"  2. Update DualDeviceNav config to point to these files")
    print(f"  3. The coordinate transformation (y,-z,-x) is applied automatically")
    print(f"  4. Each JSON file represents one vessel branch")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create DualDeviceNav-compatible data from VMR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single VMR model
  python create_dualdevicenav_format.py vmr_downloads/0105_0001/

  # Specify output folder
  python create_dualdevicenav_format.py vmr_downloads/0105_0001/ --output processed_models/model_0105/

  # Process without visual mesh
  python create_dualdevicenav_format.py vmr_downloads/0105_0001/ --no-visual

Notes:
  - This script replicates the EXACT DualDeviceNav processing pipeline
  - VTP mesh → OBJ with rotation [90, -90, 0] and scaling 10x
  - VMTK extracts centerlines with radius from VTP mesh
  - JSON saved in LPS coordinates (transformation applied by stEVE on load)
  - For best results, you may need to manually specify source/target points
        """
    )
    
    parser.add_argument("vmr_folder", help="Path to VMR model folder")
    parser.add_argument("-o", "--output", help="Output folder (default: vmr_folder/dualdevicenav_format/)")
    parser.add_argument("--no-visual", action="store_true", help="Don't create separate visual mesh")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vmr_folder):
        print(f"Error: Folder not found: {args.vmr_folder}")
        sys.exit(1)
    
    success = process_vmr_to_dualdevicenav(
        args.vmr_folder,
        args.output,
        create_visual_mesh=not args.no_visual
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

