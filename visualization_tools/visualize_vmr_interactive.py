"""
Interactive VMR mesh viewer with advanced features
Shows both VTP (surface) and VTU (volume) with centerlines
"""

import pyvista as pv
import numpy as np
import os
import sys
from pathlib import Path


def find_vmr_files(vmr_folder):
    """
    Find VTP, VTU, and PTH files in a VMR folder
    
    Args:
        vmr_folder: Path to VMR model folder (e.g., vmr_downloads/0166_0001/)
    
    Returns:
        dict with 'vtp', 'vtu', 'pth_files' paths
    """
    meshes_folder = os.path.join(vmr_folder, 'Meshes')
    paths_folder = os.path.join(vmr_folder, 'Paths')
    
    files = {
        'vtp': None,
        'vtu': None,
        'pth_files': []
    }
    
    # Find VTP and VTU
    if os.path.exists(meshes_folder):
        for file in os.listdir(meshes_folder):
            if file.endswith('.vtp'):
                files['vtp'] = os.path.join(meshes_folder, file)
            elif file.endswith('.vtu'):
                files['vtu'] = os.path.join(meshes_folder, file)
    
    # Find PTH files (centerlines)
    if os.path.exists(paths_folder):
        files['pth_files'] = [
            os.path.join(paths_folder, f) 
            for f in os.listdir(paths_folder) 
            if f.endswith('.pth')
        ]
    
    return files


def visualize_vmr_complete(vmr_folder, show_volume=False):
    """
    Complete visualization of VMR model with all components
    
    Args:
        vmr_folder: Path to VMR model folder
        show_volume: Whether to show VTU volume mesh (slower)
    """
    files = find_vmr_files(vmr_folder)
    
    model_name = os.path.basename(vmr_folder)
    print(f"\n{'='*70}")
    print(f"VMR Model Viewer: {model_name}")
    print(f"{'='*70}\n")
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.add_text(f"VMR Model: {model_name}", position='upper_edge', font_size=12)
    
    # Load and display VTP (surface mesh)
    if files['vtp']:
        print(f"Loading VTP (surface mesh): {os.path.basename(files['vtp'])}")
        vtp_mesh = pv.read(files['vtp'])
        
        print(f"  Points: {vtp_mesh.n_points:,}")
        print(f"  Cells: {vtp_mesh.n_cells:,}")
        print(f"  Bounds: X=[{vtp_mesh.bounds[0]:.1f}, {vtp_mesh.bounds[1]:.1f}], "
              f"Y=[{vtp_mesh.bounds[2]:.1f}, {vtp_mesh.bounds[3]:.1f}], "
              f"Z=[{vtp_mesh.bounds[4]:.1f}, {vtp_mesh.bounds[5]:.1f}]")
        
        # Scale to mm (VMR files are often in cm)
        vtp_mesh_scaled = vtp_mesh.copy()
        vtp_mesh_scaled.points *= 10  # cm to mm
        
        plotter.add_mesh(
            vtp_mesh_scaled,
            color='red',
            opacity=0.4,
            show_edges=False,
            label='Vessel Surface (VTP)',
            smooth_shading=True
        )
        print("  ✓ VTP mesh added\n")
    
    # Load and display VTU (volume mesh) - optional as it's slower
    if show_volume and files['vtu']:
        print(f"Loading VTU (volume mesh): {os.path.basename(files['vtu'])}")
        vtu_mesh = pv.read(files['vtu'])
        
        print(f"  Points: {vtu_mesh.n_points:,}")
        print(f"  Cells: {vtu_mesh.n_cells:,}")
        
        # Scale to mm
        vtu_mesh_scaled = vtu_mesh.copy()
        vtu_mesh_scaled.points *= 10
        
        # Show as wireframe or contour
        plotter.add_mesh(
            vtu_mesh_scaled,
            style='wireframe',
            color='blue',
            opacity=0.1,
            label='Volume Mesh (VTU)'
        )
        print("  ✓ VTU mesh added\n")
    
    # Load and display centerlines from PTH files
    if files['pth_files']:
        print(f"Loading {len(files['pth_files'])} centerline paths (PTH files):")
        
        colors = ['yellow', 'cyan', 'lime', 'magenta', 'orange', 'pink']
        
        for i, pth_file in enumerate(files['pth_files'][:10]):  # Limit to 10 for performance
            branch_name = os.path.splitext(os.path.basename(pth_file))[0]
            
            try:
                # Parse PTH file (simple XML with pos tags)
                import xml.etree.ElementTree as ET
                tree = ET.parse(pth_file)
                root = tree.getroot()
                
                points = []
                for pos in root.iter('pos'):
                    x = float(pos.get('x')) * 10  # Scale to mm
                    y = float(pos.get('y')) * 10
                    z = float(pos.get('z')) * 10
                    points.append([x, y, z])
                
                if len(points) > 1:
                    points_array = np.array(points)
                    
                    # Create polydata for the centerline
                    centerline = pv.PolyData(points_array)
                    
                    # Add as line
                    color = colors[i % len(colors)]
                    plotter.add_mesh(
                        centerline,
                        color=color,
                        line_width=3,
                        label=f'{branch_name}',
                        render_lines_as_tubes=True
                    )
                    
                    print(f"  ✓ {branch_name}: {len(points)} points")
            
            except Exception as e:
                print(f"  ✗ {branch_name}: Failed ({str(e)})")
        
        print()
    
    # Add coordinate axes
    plotter.add_axes(
        xlabel='X (mm)',
        ylabel='Y (mm)',
        zlabel='Z (mm)'
    )
    
    # Add legend
    plotter.add_legend(bcolor='white', face='rectangle', size=(0.3, 0.3))
    
    # Set camera
    plotter.camera_position = 'iso'
    plotter.enable_parallel_projection()
    
    print(f"{'='*70}")
    print("Controls:")
    print("  - Left Mouse: Rotate")
    print("  - Middle Mouse / Scroll: Zoom")
    print("  - Right Mouse: Pan")
    print("  - 'q' or close window: Exit")
    print(f"{'='*70}\n")
    
    # Show
    plotter.show()


def quick_view_vtp(file_path):
    """
    Quick visualization of a single VTP/VTU file
    """
    mesh = pv.read(file_path)
    
    # Scale if needed
    if mesh.bounds[1] < 100:  # Likely in cm, scale to mm
        mesh.points *= 10
    
    mesh.plot(
        color='red',
        opacity=0.6,
        show_edges=False,
        cpos='iso',
        window_size=[1200, 800]
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  VMR folder:  python visualize_vmr_interactive.py <path_to_vmr_folder>")
        print("  Single file: python visualize_vmr_interactive.py <path_to_vtp_or_vtu>")
        print("\nOptions:")
        print("  --volume     Also show VTU volume mesh (slower)")
        print("\nExamples:")
        print("  python visualize_vmr_interactive.py vmr_downloads/0166_0001/")
        print("  python visualize_vmr_interactive.py vmr_downloads/0166_0001/Meshes/0166_0001.vtp")
        print("  python visualize_vmr_interactive.py vmr_downloads/0166_0001/ --volume")
        sys.exit(1)
    
    path = sys.argv[1]
    show_volume = '--volume' in sys.argv
    
    if os.path.isdir(path):
        # VMR folder
        visualize_vmr_complete(path, show_volume=show_volume)
    elif os.path.isfile(path):
        # Single file
        quick_view_vtp(path)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)

