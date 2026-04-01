"""
Convert VTP/VTU files to formats optimized for 3D Slicer
Optionally scales from cm to mm for medical visualization
"""

import pyvista as pv
import numpy as np
import os
import sys


def convert_for_slicer(input_file, output_file=None, scale_to_mm=True):
    """
    Convert and optimize mesh for 3D Slicer
    
    Args:
        input_file: Path to VTP or VTU file
        output_file: Output path (default: same name with _slicer suffix)
        scale_to_mm: Whether to scale from cm to mm (default: True)
    """
    
    print(f"\nConverting: {input_file}")
    
    # Load mesh
    mesh = pv.read(input_file)
    
    # Determine if scaling is needed
    if scale_to_mm:
        # Check bounds to see if likely in cm
        max_bound = max(abs(mesh.bounds[1] - mesh.bounds[0]),
                       abs(mesh.bounds[3] - mesh.bounds[2]),
                       abs(mesh.bounds[5] - mesh.bounds[4]))
        
        if max_bound < 500:  # Likely in cm if max dimension < 500
            print(f"  Scaling from cm to mm (max bound: {max_bound:.1f})")
            mesh.points *= 10
        else:
            print(f"  No scaling needed (max bound: {max_bound:.1f})")
    
    # Clean mesh for Slicer
    mesh = mesh.clean()
    
    # Generate output filename
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_slicer{ext}"
    
    # Save in VTK format (best for Slicer)
    print(f"  Saving to: {output_file}")
    mesh.save(output_file)
    
    print(f"  ✓ Done! Points: {mesh.n_points:,}, Cells: {mesh.n_cells:,}\n")
    
    return output_file


def convert_vmr_folder(vmr_folder, output_folder=None):
    """
    Convert all VTP/VTU files in a VMR folder for Slicer
    
    Args:
        vmr_folder: Path to VMR folder (e.g., vmr_downloads/0166_0001/)
        output_folder: Output folder (default: vmr_folder/slicer/)
    """
    
    if output_folder is None:
        output_folder = os.path.join(vmr_folder, 'slicer')
    
    os.makedirs(output_folder, exist_ok=True)
    
    meshes_folder = os.path.join(vmr_folder, 'Meshes')
    
    if not os.path.exists(meshes_folder):
        print(f"Error: Meshes folder not found: {meshes_folder}")
        return
    
    converted_files = []
    
    for file in os.listdir(meshes_folder):
        if file.endswith(('.vtp', '.vtu')):
            input_path = os.path.join(meshes_folder, file)
            output_path = os.path.join(output_folder, file)
            
            converted_file = convert_for_slicer(input_path, output_path)
            converted_files.append(converted_file)
    
    print(f"\n{'='*60}")
    print(f"Converted {len(converted_files)} files to: {output_folder}")
    print(f"{'='*60}\n")
    print("To view in 3D Slicer:")
    print("  1. Open 3D Slicer")
    print("  2. File → Add Data")
    print(f"  3. Navigate to: {output_folder}")
    print("  4. Select the VTP file and click OK\n")
    
    return converted_files


def create_slicer_scene(vmr_folder, output_mrml=None):
    """
    Create a 3D Slicer scene file (.mrml) for easy loading
    
    Args:
        vmr_folder: Path to VMR folder
        output_mrml: Output MRML file path
    """
    
    if output_mrml is None:
        model_name = os.path.basename(vmr_folder)
        output_mrml = os.path.join(vmr_folder, f"{model_name}_scene.mrml")
    
    meshes_folder = os.path.join(vmr_folder, 'Meshes')
    
    # Find VTP file
    vtp_file = None
    for file in os.listdir(meshes_folder):
        if file.endswith('.vtp'):
            vtp_file = os.path.join(meshes_folder, file)
            break
    
    if not vtp_file:
        print("Error: No VTP file found")
        return
    
    # Create simple MRML scene
    mrml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<MRML  version="Slicer4.11.0" userTags="">
  <Model
    id="vtkMRMLModelNode1"
    name="VesselSurface"
    displayNodeRef="vtkMRMLModelDisplayNode1"
    storageNodeRef="vtkMRMLModelStorageNode1"
    ></Model>
  <ModelDisplay
    id="vtkMRMLModelDisplayNode1"
    name="VesselSurface_Display"
    color="1 0 0"
    edgeColor="0 0 0"
    selectedColor="1 0 0"
    opacity="0.6"
    visibility="true"
    ></ModelDisplay>
  <ModelStorage
    id="vtkMRMLModelStorageNode1"
    name="VesselSurface_Storage"
    fileName="{os.path.abspath(vtp_file)}"
    ></ModelStorage>
</MRML>
"""
    
    with open(output_mrml, 'w') as f:
        f.write(mrml_content)
    
    print(f"\n{'='*60}")
    print(f"Created Slicer scene: {output_mrml}")
    print(f"{'='*60}\n")
    print("To open in 3D Slicer:")
    print(f"  1. Open 3D Slicer")
    print(f"  2. File → Open Scene")
    print(f"  3. Select: {output_mrml}\n")
    
    return output_mrml


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python convert_for_slicer.py <file.vtp>")
        print("  VMR folder:  python convert_for_slicer.py <vmr_folder> --folder")
        print("  Scene file:  python convert_for_slicer.py <vmr_folder> --scene")
        print("\nExamples:")
        print("  python convert_for_slicer.py vmr_downloads/0166_0001/Meshes/0166_0001.vtp")
        print("  python convert_for_slicer.py vmr_downloads/0166_0001/ --folder")
        print("  python convert_for_slicer.py vmr_downloads/0166_0001/ --scene")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if '--scene' in sys.argv:
        # Create MRML scene file
        create_slicer_scene(path)
    elif '--folder' in sys.argv or os.path.isdir(path):
        # Convert entire folder
        convert_vmr_folder(path)
    elif os.path.isfile(path):
        # Convert single file
        convert_for_slicer(path)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)

