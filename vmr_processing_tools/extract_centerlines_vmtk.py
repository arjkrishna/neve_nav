"""
Extract centerlines from VTP/VTU files using VMTK Python library
Alternative to 3D Slicer VMTK for batch processing
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

try:
    import vtk
    from vmtk import vmtkscripts
except ImportError:
    print("ERROR: VMTK not installed!")
    print("Install with: pip install vmtk")
    sys.exit(1)


def extract_centerlines_vmtk_auto(surface_file, output_centerline_file, 
                                   source_points=None, target_points=None):
    """
    Extract centerlines from surface mesh using VMTK (automatic endpoint detection)
    
    Args:
        surface_file: Input VTP surface mesh
        output_centerline_file: Output VTP centerline file
        source_points: List of source point coordinates [(x,y,z), ...]
        target_points: List of target point coordinates [(x,y,z), ...]
    
    Returns:
        centerlines: VMTK centerlines object with radius information
    """
    print(f"\n{'='*70}")
    print(f"Extracting Centerlines: {os.path.basename(surface_file)}")
    print(f"{'='*70}")
    
    # 1. Read surface
    print("\n1. Reading surface mesh...")
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = surface_file
    reader.Execute()
    surface = reader.Surface
    print(f"   Surface: {surface.GetNumberOfPoints():,} points, {surface.GetNumberOfCells():,} cells")
    
    # 2. Compute centerlines
    print("\n2. Computing centerlines...")
    centerlineFilter = vmtkscripts.vmtkCenterlines()
    centerlineFilter.Surface = surface
    
    if source_points and target_points:
        # Manual endpoint specification
        print(f"   Using {len(source_points)} source points and {len(target_points)} target points")
        centerlineFilter.SeedSelectorName = 'pointlist'
        
        # Flatten lists
        source_flat = [coord for point in source_points for coord in point]
        target_flat = [coord for point in target_points for coord in point]
        
        centerlineFilter.SourcePoints = source_flat
        centerlineFilter.TargetPoints = target_flat
    else:
        # Automatic endpoint detection (interactive if no points given)
        print("   ⚠️ No endpoints specified - will attempt automatic detection")
        print("   Note: This may require manual selection in a GUI window")
        centerlineFilter.SeedSelectorName = 'openprofiles'  # Detect open ends automatically
    
    centerlineFilter.AppendEndPoints = 1
    centerlineFilter.Resampling = 1
    centerlineFilter.ResamplingStepLength = 1.0  # 1mm resolution
    
    try:
        centerlineFilter.Execute()
        centerlines = centerlineFilter.Centerlines
        
        # Check if we got valid centerlines
        if centerlines.GetNumberOfPoints() == 0:
            print("   ❌ ERROR: No centerlines extracted!")
            print("   Try specifying source and target points manually")
            return None
        
        print(f"   ✓ Extracted {centerlines.GetNumberOfPoints():,} centerline points")
        print(f"   ✓ {centerlines.GetNumberOfCells()} centerline paths")
        
    except Exception as e:
        print(f"   ❌ ERROR: Centerline extraction failed: {str(e)}")
        return None
    
    # 3. Save centerlines
    print("\n3. Saving centerlines...")
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Surface = centerlines
    writer.OutputFileName = output_centerline_file
    writer.Execute()
    print(f"   ✓ Saved to: {output_centerline_file}")
    
    return centerlines


def extract_radius_from_centerlines(centerlines_vtp):
    """
    Extract radius information from VMTK centerlines
    
    Args:
        centerlines_vtp: VMTK centerlines VTP file or vtkPolyData object
    
    Returns:
        dict: {branch_id: [(x,y,z,radius), ...]}
    """
    # Load if file path
    if isinstance(centerlines_vtp, str):
        reader = vmtkscripts.vmtkSurfaceReader()
        reader.InputFileName = centerlines_vtp
        reader.Execute()
        centerlines = reader.Surface
    else:
        centerlines = centerlines_vtp
    
    # Extract radius array (VMTK stores as "MaximumInscribedSphereRadius")
    radius_array = centerlines.GetPointData().GetArray("MaximumInscribedSphereRadius")
    
    if radius_array is None:
        print("⚠️ Warning: No radius information found in centerlines")
        return None
    
    # Group points by cell (each cell is a branch)
    branches = {}
    for cell_id in range(centerlines.GetNumberOfCells()):
        cell = centerlines.GetCell(cell_id)
        points = []
        
        for i in range(cell.GetNumberOfPoints()):
            point_id = cell.GetPointId(i)
            point = centerlines.GetPoint(point_id)
            radius = radius_array.GetValue(point_id)
            points.append((point[0], point[1], point[2], radius))
        
        branches[cell_id] = points
    
    return branches


def convert_vmtk_centerlines_to_slicer_json(centerlines_vtp, output_json_file, branch_name="centerline"):
    """
    Convert VMTK centerlines (with radius) to 3D Slicer JSON format
    
    Args:
        centerlines_vtp: VMTK centerlines VTP file
        output_json_file: Output JSON file path
        branch_name: Name for the centerline
    """
    print(f"\nConverting to Slicer JSON format...")
    
    # Extract branches with radius
    branches = extract_radius_from_centerlines(centerlines_vtp)
    
    if branches is None:
        print("❌ Cannot convert: No radius information")
        return False
    
    # Load centerlines to get data
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = centerlines_vtp
    reader.Execute()
    centerlines = reader.Surface
    
    # Create JSON for each branch (or combine all)
    all_points = []
    all_radii = []
    
    for branch_id, points in branches.items():
        for x, y, z, radius in points:
            all_points.append([x, y, z])
            all_radii.append(radius)
    
    # Create 3D Slicer markup JSON structure
    slicer_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Curve",
                "coordinateSystem": "LPS",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(all_points),
                "controlPoints": [
                    {
                        "id": str(i+1),
                        "label": f"{branch_name}-{i+1}",
                        "description": "",
                        "associatedNodeID": "",
                        "position": point,
                        "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    }
                    for i, point in enumerate(all_points)
                ],
                "measurements": [
                    {
                        "name": "Radius",
                        "enabled": True,
                        "units": "mm",
                        "printFormat": "%-#4.4g",
                        "controlPointValues": all_radii
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
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(slicer_json, f, indent=4)
    
    print(f"✓ Saved {len(all_points)} points with radius to: {output_json_file}")
    return True


def batch_extract_centerlines(vmr_folder, output_folder=None, source_points=None, target_points=None):
    """
    Batch extract centerlines from VMR model
    
    Args:
        vmr_folder: Path to VMR model folder
        output_folder: Output folder (default: vmr_folder/centerlines_vmtk/)
        source_points: List of source points (one per model)
        target_points: List of target points (one per model)
    """
    model_name = os.path.basename(vmr_folder.rstrip('/\\'))
    print(f"\n{'='*70}")
    print(f"Batch Processing VMR Model: {model_name}")
    print(f"{'='*70}")
    
    if output_folder is None:
        output_folder = os.path.join(vmr_folder, 'centerlines_vmtk')
    os.makedirs(output_folder, exist_ok=True)
    
    # Find VTP file
    meshes_folder = os.path.join(vmr_folder, 'Meshes')
    vtp_file = None
    for file in os.listdir(meshes_folder):
        if file.endswith('.vtp'):
            vtp_file = os.path.join(meshes_folder, file)
            break
    
    if not vtp_file:
        print("❌ Error: No VTP file found")
        return
    
    # Extract centerlines
    centerline_vtp = os.path.join(output_folder, f"{model_name}_centerlines.vtp")
    centerlines = extract_centerlines_vmtk_auto(vtp_file, centerline_vtp, source_points, target_points)
    
    if centerlines is None:
        return
    
    # Convert to JSON
    centerline_json = os.path.join(output_folder, f"{model_name}_centerlines.mrk.json")
    convert_vmtk_centerlines_to_slicer_json(centerline_vtp, centerline_json, branch_name=model_name)
    
    print(f"\n{'='*70}")
    print(f"✓ Complete! Output: {output_folder}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Extract Centerlines using VMTK Python Library")
        print("\nUsage:")
        print("  python extract_centerlines_vmtk.py <vtp_file>")
        print("  python extract_centerlines_vmtk.py <vmr_folder>")
        print("\nExamples:")
        print("  # Extract from VTP file")
        print("  python extract_centerlines_vmtk.py vmr_downloads/0105_0001/Meshes/0105_0001.vtp")
        print("\n  # Extract from VMR folder")
        print("  python extract_centerlines_vmtk.py vmr_downloads/0105_0001/")
        print("\nNote:")
        print("  - Requires VMTK: pip install vmtk")
        print("  - May open GUI for endpoint selection if not automated")
        print("  - For batch processing, consider using 3D Slicer CLI instead")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if os.path.isdir(input_path):
        # VMR folder
        batch_extract_centerlines(input_path)
    elif input_path.endswith('.vtp'):
        # Single VTP file
        output_vtp = input_path.replace('.vtp', '_centerlines.vtp')
        output_json = input_path.replace('.vtp', '_centerlines.mrk.json')
        
        centerlines = extract_centerlines_vmtk_auto(input_path, output_vtp)
        if centerlines:
            convert_vmtk_centerlines_to_slicer_json(output_vtp, output_json)
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)

