"""
Create DualDeviceNav-Compatible Data WITHOUT Pre-Rotation

This script creates OBJ and JSON files WITHOUT rotation applied:
- OBJ mesh: Scaled only (cm -> mm), NO rotation
- JSON centerlines: Scaled only, NO rotation
- All rotation is handled by stEVE via rotation_yzx_deg and rotate_branches/rotate_ip

This is a cleaner approach than pre-rotating the data.

Usage:
    python create_dualdevicenav_format_norot.py <vmr_model_folder> [--output <output_folder>]

Example:
    python create_dualdevicenav_format_norot.py D:\vmr\vmr\0011_H_AO_H
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


# Processing settings - NO ROTATION
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


def create_obj_mesh_norot(vtp_file: str, output_obj: str, decimation_factor: float = 0.9):
    """
    Create OBJ mesh from VTP WITHOUT rotation (scaled only)
    
    Args:
        vtp_file: Input VTP mesh file
        output_obj: Output OBJ file path
        decimation_factor: Mesh decimation factor (0.9 = keep 10% of faces)
    """
    print(f"\n{'='*70}")
    print(f"Creating OBJ Mesh (NO ROTATION): {os.path.basename(output_obj)}")
    print(f"{'='*70}")
    
    # Load VTP mesh
    print("1. Loading VTP mesh...")
    mesh = pv.read(vtp_file)
    print(f"   Original: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    
    # Flip normals (standard for SOFA)
    print("2. Flipping normals...")
    mesh.flip_normals()
    
    # Scale from cm to mm (NO ROTATION)
    print(f"3. Scaling by {SCALING_FACTOR}x (cm -> mm)...")
    mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
    print("   [NO ROTATION APPLIED - will be handled by stEVE]")
    
    # Decimate mesh
    if decimation_factor > 0:
        print(f"4. Decimating mesh (factor={decimation_factor})...")
        mesh.decimate(decimation_factor, inplace=True)
        print(f"   Decimated: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    
    # Save as OBJ
    print(f"5. Saving OBJ file...")
    pv.save_meshio(output_obj, mesh)
    print(f"   [OK] Saved: {output_obj}")
    
    return mesh


def extract_centerlines_vmtk_norot(vtp_file: str, 
                                    source_points: Optional[List[Tuple[float, float, float]]] = None,
                                    target_points: Optional[List[Tuple[float, float, float]]] = None,
                                    resample_step: float = 1.0) -> Optional[object]:
    """
    Extract centerlines from VTP using VMTK (NO ROTATION, scaled only)
    
    Args:
        vtp_file: Input VTP mesh file
        source_points: List of source points (inlets)
        target_points: List of target points (outlets)
        resample_step: Resampling step in mm (default: 1.0mm)
    
    Returns:
        VMTK centerlines object with radius data, or None if failed
    """
    print(f"\n{'='*70}")
    print(f"Extracting Centerlines with VMTK (NO ROTATION)")
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
        
        # Flatten lists (points in cm, not scaled yet)
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


def extract_endpoints_from_pth(pth_folder: str) -> Tuple[Optional[List[Tuple]], Optional[List[Tuple]]]:
    """
    Extract source and target points from PTH files
    
    MATCHES THE ORIGINAL: Use a single common source point (aortic root) and only the
    actual branch endpoints as targets to avoid phantom connections.
    
    Returns:
        (source_points, target_points) or (None, None) if failed
    """
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
                print(f"   [OK] Extracted {len(points)} points from {pth_file.name}")
            
        except Exception as e:
            print(f"   [WARNING] Could not parse {pth_file.name}: {e}")
            continue
    
    if not all_first_points or not all_last_points:
        print(f"   [ERROR] Could not extract any valid paths!")
        return None, None
    
    # Use the most common first point as the source (aortic root)
    # In a well-formed aortic arch, all branches start from the same point
    from collections import Counter
    point_counter = Counter(all_first_points)
    source_point = point_counter.most_common(1)[0][0]
    
    # Use only the actual branch endpoints as targets
    target_points = all_last_points
    
    print(f"   [SUCCESS] Extracted 1 source and {len(target_points)} target points")
    print(f"   Source: {source_point}")
    
    return [source_point], target_points


def convert_vmtk_to_dualdevicenav_json_norot(centerlines, 
                                               output_folder: str,
                                               model_name: str = "centerline",
                                               scaling_factor: float = 10.0):
    """
    Convert VMTK centerlines to DualDeviceNav JSON format (NO ROTATION)
    
    Creates SEPARATE JSON files for each branch (matching DualDeviceNav format).
    
    The JSON format matches exactly what DualDeviceNav expects:
    - Each branch = separate JSON file
    - Coordinates in LPS system (NOT transformed yet)
    - Radius measurements included
    - 3D Slicer markup format
    
    Transformation (y, -z, -x) is applied by stEVE when loading!
    NO ROTATION applied during preprocessing (handled by stEVE via rotation_yzx_deg)
    
    Args:
        centerlines: VMTK centerlines object
        output_folder: Output folder for JSON files
        model_name: Base name for the centerline files
        scaling_factor: Scale factor to match mesh (default 10.0 for cm->mm)
    """
    print(f"\n{'='*70}")
    print(f"Converting to DualDeviceNav JSON Format (NO ROTATION)")
    print(f"{'='*70}")
    
    # Extract radius array
    radius_array = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius_array is None:
        print("[ERROR] No radius information in centerlines!")
        return False
    
    # CRITICAL: VMTK extracts centerlines in VTP's original units (cm)
    # We need to scale coordinates AND radii to match the scaled mesh (mm)
    # NO ROTATION applied (handled by stEVE)
    print(f"   Applying scaling factor: {scaling_factor}x (cm -> mm)")
    print(f"   NO rotation applied (will be handled by stEVE)")
    
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
            
            # SCALE coordinates and radius to match mesh (cm -> mm)
            # NO ROTATION (handled by stEVE)
            points.append([point[0] * scaling_factor, 
                          point[1] * scaling_factor, 
                          point[2] * scaling_factor])
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
    print(f"\n   NOTE: Coordinates in LPS, scaled to mm, NO rotation applied")
    print(f"   NOTE: Rotation will be handled by stEVE via rotation_yzx_deg and rotate_branches")
    
    return True


def process_vmr_to_dualdevicenav_norot(vmr_folder: str, 
                                       output_folder: Optional[str] = None,
                                       create_visual_mesh: bool = True,
                                       source_points: Optional[List[Tuple]] = None,
                                       target_points: Optional[List[Tuple]] = None):
    """
    Process VMR model to DualDeviceNav format WITHOUT pre-rotation
    
    Args:
        vmr_folder: Path to VMR model folder
        output_folder: Output folder (default: vmr_folder/dualdevicenav_format_norot/)
        create_visual_mesh: Create separate visual mesh (default: True)
        source_points: Manual source points for centerline extraction
        target_points: Manual target points for centerline extraction
    """
    model_name = os.path.basename(vmr_folder.rstrip('/\\'))
    
    print(f"\n{'='*70}")
    print(f"Processing VMR Model: {model_name}")
    print(f"Creating DualDeviceNav-Compatible Data (NO PRE-ROTATION)")
    print(f"{'='*70}")
    
    # Find VTP file
    vtp_file = find_vtp_file(vmr_folder)
    if not vtp_file:
        print(f"ERROR: No VTP file found in {vmr_folder}/Meshes/")
        return False
    
    # Setup output folder
    if output_folder is None:
        output_folder = os.path.join(vmr_folder, "dualdevicenav_format_norot")
    
    os.makedirs(output_folder, exist_ok=True)
    centerline_folder = os.path.join(output_folder, "Centrelines")
    os.makedirs(centerline_folder, exist_ok=True)
    
    print(f"\nInput: {vtp_file}")
    print(f"Output: {output_folder}")
    
    # Auto-detect endpoints from PTH if not provided
    if source_points is None or target_points is None:
        print(f"\n[DEBUG] Attempting to extract endpoints from PTH files...")
        pth_folder = os.path.join(vmr_folder, "Paths")
        print(f"[DEBUG] PTH folder: {pth_folder}")
        print(f"[DEBUG] PTH folder exists: {os.path.exists(pth_folder)}")
        source_points, target_points = extract_endpoints_from_pth(pth_folder)
        print(f"[DEBUG] Extracted source_points: {source_points is not None}")
        print(f"[DEBUG] Extracted target_points: {target_points is not None}")
        
        if source_points is None or target_points is None:
            print(f"\n[ERROR] PTH extraction failed and automatic GUI detection not supported!")
            print(f"[ERROR] Please check that PTH files exist in: {pth_folder}")
            return False
    
    # Create collision mesh (NO ROTATION)
    collision_obj = os.path.join(output_folder, f"{model_name}_collision.obj")
    create_obj_mesh_norot(vtp_file, collision_obj, decimation_factor=0.9)
    
    # Create visual mesh (NO ROTATION)
    if create_visual_mesh:
        visual_obj = os.path.join(output_folder, f"{model_name}_visual.obj")
        create_obj_mesh_norot(vtp_file, visual_obj, decimation_factor=0.8)
    
    # Extract centerlines with VMTK (NO ROTATION)
    centerlines = extract_centerlines_vmtk_norot(
        vtp_file,
        source_points=source_points,
        target_points=target_points,
        resample_step=1.0
    )
    
    if centerlines is None:
        print(f"\n[ERROR] Centerline extraction failed")
        return False
    
    # Convert to DualDeviceNav JSON format (NO ROTATION)
    try:
        success = convert_vmtk_to_dualdevicenav_json_norot(centerlines, centerline_folder, model_name, SCALING_FACTOR)
        if not success:
            return False
    except Exception as e:
        print(f"\n[ERROR] JSON conversion failed: {str(e)}")
        return False
    
    # Success summary
    print(f"\n{'='*70}")
    print(f"[SUCCESS] DualDeviceNav-Compatible Data Created (NO PRE-ROTATION)")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Collision mesh: {collision_obj}")
    if create_visual_mesh:
        print(f"  Visual mesh:    {visual_obj}")
    print(f"  Centerlines:    {centerline_folder}")
    print(f"\nTo use in stEVE:")
    print(f"  1. Use DualDeviceNavCustom with:")
    print(f"     - rotation_yzx_deg=[90, -90, 0]  (or your desired rotation)")
    print(f"     - rotate_branches=True")
    print(f"     - rotate_ip=True")
    print(f"  2. All rotation is handled by stEVE (clean approach!)")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create DualDeviceNav-compatible data WITHOUT pre-rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single VMR model
  python create_dualdevicenav_format_norot.py D:\\vmr\\vmr\\0011_H_AO_H

  # Specify output folder
  python create_dualdevicenav_format_norot.py D:\\vmr\\vmr\\0011_H_AO_H --output processed_models/model_0011/

Notes:
  - This script creates data WITHOUT pre-rotation (cleaner approach)
  - OBJ mesh: scaled only (cm -> mm), NO rotation
  - JSON centerlines: scaled only, NO rotation
  - All rotation is handled by stEVE via rotation_yzx_deg and rotate_branches/rotate_ip
  - Use with: DualDeviceNavCustom(..., rotation_yzx_deg=[90,-90,0], rotate_branches=True, rotate_ip=True)
        """
    )
    
    parser.add_argument("vmr_folder", help="Path to VMR model folder")
    parser.add_argument("-o", "--output", help="Output folder (default: vmr_folder/dualdevicenav_format_norot/)")
    parser.add_argument("--no-visual", action="store_true", help="Don't create separate visual mesh")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vmr_folder):
        print(f"Error: Folder not found: {args.vmr_folder}")
        sys.exit(1)
    
    success = process_vmr_to_dualdevicenav_norot(
        args.vmr_folder,
        args.output,
        create_visual_mesh=not args.no_visual
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

