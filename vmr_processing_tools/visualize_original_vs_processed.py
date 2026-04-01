"""
Visualize Original vs Processed Data

This script shows the original VTP mesh with extracted centerlines
BEFORE any transformations, so you can verify the centerline extraction
was successful.

Usage:
    python visualize_original_vs_processed.py <vmr_model_folder>
    
Example:
    python visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H
"""

import os
import sys
import json
import numpy as np
import pyvista as pv
from pathlib import Path


def find_vtp_file(model_folder):
    """Find the VTP mesh file in the model folder"""
    meshes_folder = os.path.join(model_folder, "Meshes")
    models_folder = os.path.join(model_folder, "Models")
    
    vtp_files = []
    if os.path.exists(meshes_folder):
        vtp_files.extend(list(Path(meshes_folder).glob("*.vtp")))
    if os.path.exists(models_folder):
        vtp_files.extend(list(Path(models_folder).glob("*.vtp")))
    
    return str(vtp_files[0]) if vtp_files else None


def load_centerline_from_pth(pth_file):
    """Load centerline from PTH file (original orientation, no transformations)"""
    import xml.etree.ElementTree as ET
    import re
    
    # Read file and wrap in a single root element to fix XML
    with open(pth_file, 'r') as f:
        content = f.read()
    
    # Remove XML declaration if present
    content = re.sub(r'<\?xml[^?]*\?>\s*', '', content)
    
    # Wrap the content in a root element
    wrapped_xml = f"<root>{content}</root>"
    root = ET.fromstring(wrapped_xml)
    
    # Find path_points in the path element
    path_points = root.findall('.//path_point')
    if not path_points:
        return None, None
    
    points = []
    for path_point in path_points:
        pos = path_point.find('pos')
        if pos is not None:
            x = float(pos.get('x', 0))
            y = float(pos.get('y', 0))
            z = float(pos.get('z', 0))
            # Scale from cm to mm to match scaled VTP mesh
            SCALING_FACTOR = 10.0
            points.append([x * SCALING_FACTOR, y * SCALING_FACTOR, z * SCALING_FACTOR])
    
    # PTH files don't have radius data
    return np.array(points) if len(points) > 0 else None, None


def visualize_original(model_folder):
    """Visualize the ORIGINAL data before transformations"""
    print("="*80)
    print("Visualizing ORIGINAL Data (Before Transformations)")
    print("="*80)
    print(f"Model folder: {model_folder}\n")
    
    # Find VTP file
    vtp_file = find_vtp_file(model_folder)
    if not vtp_file:
        print("[ERROR] No VTP file found in Meshes/ or Models/ folder")
        return
    
    print(f"Loading original VTP: {vtp_file}")
    original_mesh = pv.read(vtp_file)
    print(f"  Points: {original_mesh.n_points:,}")
    print(f"  Cells: {original_mesh.n_cells:,}")
    print(f"  Bounds: {original_mesh.bounds}")
    
    # Scale VTP mesh by 10x (cm -> mm) to match scaled centerlines from PTH
    # PTH centerlines are in cm, so we scale them to mm as well
    SCALING_FACTOR = 10.0
    original_mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
    print(f"  Scaled bounds (cm -> mm): {original_mesh.bounds}")
    
    # Initialize plotter
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Add original mesh (now scaled to mm)
    plotter.add_mesh(
        original_mesh,
        color='lightblue',
        opacity=0.3,
        label='Original VTP Mesh (scaled to mm)',
        show_edges=True
    )
    
    # Load centerlines from Paths folder (original PTH files)
    paths_folder = os.path.join(model_folder, "Paths")
    
    if os.path.exists(paths_folder):
        print(f"\nLoading centerlines from original PTH files: {paths_folder}")
        
        pth_files = sorted(list(Path(paths_folder).glob("*.pth")))
        
        if not pth_files:
            print(f"[WARNING] No PTH files found in {paths_folder}")
        else:
            colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
                      'orange', 'purple', 'brown', 'pink', 'lime', 'navy']
            
            print(f"Found {len(pth_files)} PTH files\n")
            
            for idx, pth_file in enumerate(pth_files):
                try:
                    points, radii = load_centerline_from_pth(pth_file)
                    
                    if points is not None and len(points) > 1:
                        color = colors[idx % len(colors)]
                        
                        # Create spline
                        spline = pv.Spline(points, n_points=len(points)*5)
                        
                        # Use default tube radius (PTH files don't have radius data)
                        tube = spline.tube(radius=1.0)
                        plotter.add_mesh(
                            tube,
                            color=color,
                            opacity=0.8,
                            label=f'Branch {idx+1} ({pth_file.name})',
                            smooth_shading=True
                        )
                        
                        print(f"  [{idx+1}] {pth_file.name}")
                        print(f"      Points: {len(points)}")
                        print(f"      Bounds: X[{np.min(points[:,0]):.1f}, {np.max(points[:,0]):.1f}] "
                              f"Y[{np.min(points[:,1]):.1f}, {np.max(points[:,1]):.1f}] "
                              f"Z[{np.min(points[:,2]):.1f}, {np.max(points[:,2]):.1f}]")
                        
                        # Add points
                        point_cloud = pv.PolyData(points)
                        plotter.add_mesh(
                            point_cloud,
                            color=color,
                            point_size=8,
                            render_points_as_spheres=True
                        )
                
                except Exception as e:
                    print(f"  [ERROR] Failed to load {pth_file.name}: {e}")
    else:
        print(f"[WARNING] No Paths folder found at: {paths_folder}")
        print("          Cannot show original centerlines")
    
    # Add axes
    plotter.add_axes(
        xlabel='X (mm)',
        ylabel='Y (mm)',
        zlabel='Z (mm)',
        line_width=5
    )
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='upper right')
    
    print("\n" + "="*80)
    print("Visualization Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - Right mouse drag: Pan")
    print("  - 'q' or close window: Exit")
    print("="*80)
    print("\nCoordinate System: LPS (Left-Posterior-Superior)")
    print("NOTE: VTP mesh is scaled by 10x (cm -> mm)")
    print("      Centerlines are loaded from original PTH files and scaled to mm")
    print("      Both are in original orientation (no rotation applied)")
    print("      Centerlines should be INSIDE the mesh")
    print("="*80)
    
    plotter.show()


def visualize_comparison(model_folder):
    """Show original VTP side-by-side with processed OBJ"""
    print("="*80)
    print("Comparison: Original VTP vs Processed OBJ")
    print("="*80)
    
    # Find original VTP
    vtp_file = find_vtp_file(model_folder)
    if not vtp_file:
        print("[ERROR] No VTP file found")
        return
    
    # Find processed OBJ
    processed_folder = os.path.join(model_folder, "dualdevicenav_format")
    collision_obj = os.path.join(processed_folder, "vessel_architecture_collision.obj")
    
    # If default name doesn't exist, try to find any collision mesh
    if not os.path.exists(collision_obj):
        obj_files = [f for f in os.listdir(processed_folder) if f.endswith('_collision.obj')]
        if obj_files:
            collision_obj = os.path.join(processed_folder, obj_files[0])
            print(f"Found collision mesh: {obj_files[0]}")
        else:
            print(f"[ERROR] No processed collision mesh found in: {processed_folder}")
            print(f"Expected: vessel_architecture_collision.obj or <model_name>_collision.obj")
            print("        Run the processing script first!")
            return
    
    # Create split view
    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('white')
    
    # Left: Original VTP (scaled to mm for comparison)
    plotter.subplot(0, 0)
    original = pv.read(vtp_file)
    SCALING_FACTOR = 10.0
    original.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
    plotter.add_text("ORIGINAL VTP\n(Scaled to mm for comparison)", font_size=12, position='upper_edge')
    plotter.add_mesh(original, color='lightblue', opacity=0.5, show_edges=True)
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    print(f"\nOriginal VTP (scaled to mm):")
    print(f"  Points: {original.n_points:,}")
    print(f"  Cells: {original.n_cells:,}")
    print(f"  Bounds: {original.bounds}")
    
    # Right: Processed OBJ
    plotter.subplot(0, 1)
    processed = pv.read(collision_obj)
    plotter.add_text("PROCESSED OBJ\n(After rotation, scaling, decimation)", 
                     font_size=12, position='upper_edge')
    plotter.add_mesh(processed, color='lightcoral', opacity=0.5, show_edges=True)
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    print(f"\nProcessed OBJ:")
    print(f"  Points: {processed.n_points:,}")
    print(f"  Cells: {processed.n_cells:,}")
    print(f"  Bounds: {processed.bounds}")
    
    print(f"\nTransformations applied:")
    print(f"  1. Flip normals")
    print(f"  2. Scale by 10x (cm -> mm)")
    print(f"  3. Rotate [-90, 0, 90] degrees (Y, Z, X)")
    print(f"  4. Decimate by 0.9 factor")
    
    # Link cameras
    plotter.link_views()
    
    print("\n" + "="*80)
    print("Views are linked - rotating one rotates both")
    print("="*80)
    
    plotter.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_original_vs_processed.py <vmr_model_folder> [mode]")
        print("\nModes:")
        print("  original   - Show original VTP with centerlines (default)")
        print("  compare    - Side-by-side comparison of VTP vs OBJ")
        print("\nExamples:")
        print("  python visualize_original_vs_processed.py D:\\vmr\\vmr\\0011_H_AO_H")
        print("  python visualize_original_vs_processed.py D:\\vmr\\vmr\\0011_H_AO_H compare")
        sys.exit(1)
    
    model_folder = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'original'
    
    if not os.path.exists(model_folder):
        print(f"ERROR: Folder not found: {model_folder}")
        sys.exit(1)
    
    if mode == 'compare':
        visualize_comparison(model_folder)
    else:
        visualize_original(model_folder)


if __name__ == "__main__":
    main()


