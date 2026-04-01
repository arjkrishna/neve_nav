"""
Convert VMR data (VTP/VTU/PTH) to stEVE format (OBJ + JSON)
Creates files compatible with DualDeviceNav benchmark format
"""

import os
import sys
import json
import numpy as np
import pyvista as pv
from pathlib import Path
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET


def convert_vtp_to_obj(vtp_file, output_obj_file, scale_to_mm=True, rotation=None, decimation=0.9):
    """
    Convert VTP surface mesh to OBJ format (like stEVE does)
    
    Args:
        vtp_file: Input VTP file path
        output_obj_file: Output OBJ file path
        scale_to_mm: Scale from cm to mm (default: True)
        rotation: Tuple of (y, z, x) rotation angles in degrees
        decimation: Mesh decimation factor (0-1, lower = more decimation)
    """
    print(f"\nConverting VTP to OBJ: {os.path.basename(vtp_file)}")
    
    # Load mesh
    mesh = pv.read(vtp_file)
    print(f"  Original: {mesh.n_points:,} points, {mesh.n_cells:,} cells")
    
    # Flip normals (like stEVE VMR class does)
    mesh.flip_normals()
    
    # Scale to mm
    if scale_to_mm:
        max_bound = max(abs(mesh.bounds[1] - mesh.bounds[0]),
                       abs(mesh.bounds[3] - mesh.bounds[2]),
                       abs(mesh.bounds[5] - mesh.bounds[4]))
        if max_bound < 500:  # Likely in cm
            print(f"  Scaling from cm to mm (max bound: {max_bound:.1f})")
            mesh.scale([10, 10, 10], inplace=True)
    
    # Apply rotation if specified
    if rotation is not None:
        y_rot, z_rot, x_rot = rotation
        if y_rot != 0:
            mesh.rotate_y(y_rot, inplace=True)
        if z_rot != 0:
            mesh.rotate_z(z_rot, inplace=True)
        if x_rot != 0:
            mesh.rotate_x(x_rot, inplace=True)
        print(f"  Rotation applied: Y={y_rot}°, Z={z_rot}°, X={x_rot}°")
    
    # Decimate mesh
    if decimation < 1.0:
        mesh = mesh.decimate(decimation, inplace=True)
        print(f"  Decimated: {mesh.n_points:,} points, {mesh.n_cells:,} cells ({decimation*100:.0f}% reduction)")
    
    # Save as OBJ
    pv.save_meshio(output_obj_file, mesh)
    print(f"  ✓ Saved to: {output_obj_file}")
    
    return mesh


def convert_pth_to_slicer_json(pth_file, output_json_file, scale_to_mm=True, coordinate_transform=None):
    """
    Convert PTH (XML centerline) to 3D Slicer .mrk.json format
    
    Args:
        pth_file: Input PTH file path
        output_json_file: Output JSON file path
        scale_to_mm: Scale coordinates from cm to mm
        coordinate_transform: Function to transform coordinates (e.g., for DualDeviceNav: (x,y,z) -> (y,-z,-x))
    """
    print(f"\nConverting PTH to JSON: {os.path.basename(pth_file)}")
    
    # Parse PTH file (XML format)
    tree = ET.parse(pth_file)
    root = tree.getroot()
    
    points = []
    for pos in root.iter('pos'):
        x = float(pos.get('x'))
        y = float(pos.get('y'))
        z = float(pos.get('z'))
        
        # Scale to mm
        if scale_to_mm:
            x, y, z = x * 10, y * 10, z * 10
        
        # Apply coordinate transform if specified
        if coordinate_transform is not None:
            x, y, z = coordinate_transform(x, y, z)
        
        points.append([x, y, z])
    
    if len(points) == 0:
        print(f"  ⚠️ Warning: No points found in {pth_file}")
        return
    
    # Create 3D Slicer markup JSON structure
    branch_name = os.path.splitext(os.path.basename(pth_file))[0]
    
    slicer_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Curve",
                "coordinateSystem": "LPS",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": len(points),
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
                    for i, point in enumerate(points)
                ],
                "measurements": [],
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
                    "sliceProjection": False,
                    "sliceProjectionUseFiducialColor": True,
                    "sliceProjectionOutlinedBehindSlicePlane": False,
                    "sliceProjectionColor": [1.0, 1.0, 1.0],
                    "sliceProjectionOpacity": 0.6,
                    "lineThickness": 0.2,
                    "lineColorFadingStart": 1.0,
                    "lineColorFadingEnd": 10.0,
                    "lineColorFadingSaturation": 1.0,
                    "lineColorFadingHueOffset": 0.0,
                    "handlesInteractive": False,
                    "snapMode": "toVisibleSurface"
                }
            }
        ]
    }
    
    # Save JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(slicer_json, f, indent=4)
    
    print(f"  ✓ Converted {len(points)} points to: {output_json_file}")


def process_vmr_model_complete(vmr_folder, output_folder=None, obj_basename=None, rotation=None, coordinate_transform=None):
    """
    Complete conversion of VMR model to stEVE format
    
    Args:
        vmr_folder: Path to VMR model folder (e.g., vmr_downloads/0105_0001/)
        output_folder: Output folder (default: vmr_folder/steve_format/)
        obj_basename: Base name for OBJ files (default: model name)
        rotation: Rotation angles (y, z, x) for mesh
        coordinate_transform: Function to transform centerline coordinates
    
    Output:
        - collision.obj (decimated mesh for collision)
        - visual.obj (higher quality for visualization)
        - Centerlines/ folder with .mrk.json files
    """
    
    model_name = os.path.basename(vmr_folder.rstrip('/\\'))
    print(f"\n{'='*70}")
    print(f"Processing VMR Model: {model_name}")
    print(f"{'='*70}")
    
    # Setup paths
    if output_folder is None:
        output_folder = os.path.join(vmr_folder, 'steve_format')
    os.makedirs(output_folder, exist_ok=True)
    
    centerlines_folder = os.path.join(output_folder, 'Centerlines')
    os.makedirs(centerlines_folder, exist_ok=True)
    
    meshes_folder = os.path.join(vmr_folder, 'Meshes')
    paths_folder = os.path.join(vmr_folder, 'Paths')
    
    if obj_basename is None:
        obj_basename = model_name
    
    # 1. Convert VTP to OBJ files
    vtp_file = None
    for file in os.listdir(meshes_folder):
        if file.endswith('.vtp'):
            vtp_file = os.path.join(meshes_folder, file)
            break
    
    if vtp_file:
        # Collision mesh (more decimated)
        collision_obj = os.path.join(output_folder, f"{obj_basename}_collision.obj")
        convert_vtp_to_obj(vtp_file, collision_obj, rotation=rotation, decimation=0.9)
        
        # Visual mesh (less decimated)
        visual_obj = os.path.join(output_folder, f"{obj_basename}_visual.obj")
        convert_vtp_to_obj(vtp_file, visual_obj, rotation=rotation, decimation=0.95)
    else:
        print("⚠️ Warning: No VTP file found")
    
    # 2. Convert PTH files to JSON
    if os.path.exists(paths_folder):
        pth_files = [f for f in os.listdir(paths_folder) if f.endswith('.pth')]
        print(f"\nFound {len(pth_files)} PTH centerline files")
        
        for pth_file in pth_files:
            pth_path = os.path.join(paths_folder, pth_file)
            branch_name = os.path.splitext(pth_file)[0]
            json_path = os.path.join(centerlines_folder, f"Centerline curve - {branch_name}.mrk.json")
            
            convert_pth_to_slicer_json(pth_path, json_path, coordinate_transform=coordinate_transform)
    else:
        print("\n⚠️ Warning: No Paths folder found - no centerlines to convert")
        print("   You'll need to extract centerlines manually in 3D Slicer")
    
    print(f"\n{'='*70}")
    print(f"✓ Conversion Complete!")
    print(f"{'='*70}")
    print(f"Output folder: {output_folder}")
    print(f"\nFiles created:")
    print(f"  - {obj_basename}_collision.obj")
    print(f"  - {obj_basename}_visual.obj")
    if os.path.exists(centerlines_folder):
        json_count = len([f for f in os.listdir(centerlines_folder) if f.endswith('.json')])
        print(f"  - Centerlines/ ({json_count} .mrk.json files)")
    print()


def batch_process_vmr_models(vmr_downloads_folder, model_ids=None):
    """
    Batch process multiple VMR models
    
    Args:
        vmr_downloads_folder: Path to vmr_downloads folder
        model_ids: List of model IDs to process (default: all)
    """
    
    if model_ids is None:
        # Process all models in folder
        model_ids = [d for d in os.listdir(vmr_downloads_folder) 
                     if os.path.isdir(os.path.join(vmr_downloads_folder, d))]
    
    print(f"\n{'='*70}")
    print(f"Batch Processing {len(model_ids)} VMR Models")
    print(f"{'='*70}\n")
    
    for model_id in model_ids:
        vmr_folder = os.path.join(vmr_downloads_folder, model_id)
        if not os.path.exists(vmr_folder):
            print(f"⚠️ Skipping {model_id}: folder not found")
            continue
        
        try:
            process_vmr_model_complete(vmr_folder)
        except Exception as e:
            print(f"❌ Error processing {model_id}: {str(e)}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Batch Processing Complete!")
    print(f"{'='*70}\n")


# Example coordinate transforms for different cases
def dualdevicenav_transform(x, y, z):
    """Transform used in DualDeviceNav: (x,y,z) -> (y,-z,-x)"""
    return (y, -z, -x)


def identity_transform(x, y, z):
    """No transformation"""
    return (x, y, z)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single model:  python convert_vmr_to_steve_format.py <vmr_folder>")
        print("  Batch:         python convert_vmr_to_steve_format.py <vmr_downloads_folder> --batch")
        print("\nOptions:")
        print("  --rotation Y,Z,X       Apply rotation (e.g., --rotation 90,-90,0)")
        print("  --transform dual       Use DualDeviceNav coordinate transform")
        print("  --batch                Process all models in folder")
        print("\nExamples:")
        print("  # Single model")
        print("  python convert_vmr_to_steve_format.py vmr_downloads/0105_0001/")
        print("\n  # With rotation like DualDeviceNav")
        print("  python convert_vmr_to_steve_format.py vmr_downloads/0105_0001/ --rotation 90,-90,0")
        print("\n  # Batch process all")
        print("  python convert_vmr_to_steve_format.py vmr_downloads/ --batch")
        sys.exit(1)
    
    vmr_path = sys.argv[1]
    
    # Parse options
    rotation = None
    coordinate_transform = None
    batch_mode = '--batch' in sys.argv
    
    if '--rotation' in sys.argv:
        idx = sys.argv.index('--rotation')
        rotation_str = sys.argv[idx + 1]
        rotation = tuple(map(float, rotation_str.split(',')))
    
    if '--transform' in sys.argv:
        idx = sys.argv.index('--transform')
        transform_type = sys.argv[idx + 1]
        if transform_type == 'dual':
            coordinate_transform = dualdevicenav_transform
    
    # Process
    if batch_mode:
        batch_process_vmr_models(vmr_path)
    else:
        process_vmr_model_complete(vmr_path, rotation=rotation, coordinate_transform=coordinate_transform)

