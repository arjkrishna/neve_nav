"""
Visualization script for VTP/VTU mesh files
Usage: python visualize_mesh.py <path_to_vtp_or_vtu_file>
"""

import pyvista as pv
import numpy as np
import sys
import os


def visualize_mesh(file_path, show_edges=False, show_centerlines=True):
    """
    Visualize VTP or VTU mesh files with centerlines
    
    Args:
        file_path: Path to .vtp or .vtu file
        show_edges: Whether to show mesh edges
        show_centerlines: Whether to attempt to show centerlines if available
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Loading mesh: {file_path}")
    print(f"{'='*60}\n")
    
    # Load the mesh
    mesh = pv.read(file_path)
    
    # Print mesh information
    print("Mesh Information:")
    print(f"  Type: {type(mesh).__name__}")
    print(f"  Number of points: {mesh.n_points:,}")
    print(f"  Number of cells: {mesh.n_cells:,}")
    print(f"  Bounds (x, y, z):")
    print(f"    X: [{mesh.bounds[0]:.2f}, {mesh.bounds[1]:.2f}]")
    print(f"    Y: [{mesh.bounds[2]:.2f}, {mesh.bounds[3]:.2f}]")
    print(f"    Z: [{mesh.bounds[4]:.2f}, {mesh.bounds[5]:.2f}]")
    
    # Check for scalar data (like radius, curvature, etc.)
    print(f"\n  Point Data Arrays:")
    if mesh.point_data:
        for name in mesh.point_data.keys():
            array = mesh.point_data[name]
            print(f"    - {name}: shape={array.shape}, "
                  f"range=[{array.min():.2f}, {array.max():.2f}]")
    else:
        print("    (none)")
    
    print(f"\n  Cell Data Arrays:")
    if mesh.cell_data:
        for name in mesh.cell_data.keys():
            array = mesh.cell_data[name]
            print(f"    - {name}: shape={array.shape}, "
                  f"range=[{array.min():.2f}, {array.max():.2f}]")
    else:
        print("    (none)")
    
    print(f"\n{'='*60}")
    print("Rendering... (Close window to exit)")
    print(f"{'='*60}\n")
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add the main mesh
    if mesh.n_cells > 0:
        # Surface mesh (VTP)
        plotter.add_mesh(
            mesh,
            color='red',
            opacity=0.6,
            show_edges=show_edges,
            edge_color='darkred',
            label='Vessel Surface'
        )
    else:
        # Just points (centerline)
        plotter.add_mesh(
            mesh,
            color='blue',
            point_size=5,
            render_points_as_spheres=True,
            label='Centerline Points'
        )
    
    # Try to visualize centerlines if they exist in cell data
    if show_centerlines and 'MaximumInscribedSphereRadius' in mesh.point_data:
        # VMTK centerlines have radius information
        radii = mesh.point_data['MaximumInscribedSphereRadius']
        
        # Create tubes along centerline with varying radii
        if hasattr(mesh, 'lines') and mesh.lines is not None and len(mesh.lines) > 0:
            try:
                tubes = mesh.tube(radius=radii, n_sides=12)
                plotter.add_mesh(
                    tubes,
                    color='lightblue',
                    opacity=0.4,
                    label='Centerline (with radius)'
                )
            except:
                pass  # If tubing fails, skip
        
        # Show radius as color
        plotter.add_mesh(
            mesh.copy(),
            scalars='MaximumInscribedSphereRadius',
            cmap='jet',
            point_size=8,
            render_points_as_spheres=True,
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Radius (mm)'},
            opacity=0.8,
            label='Centerline colored by radius'
        )
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Add legend
    plotter.add_legend()
    
    # Set camera and show
    plotter.camera_position = 'iso'
    plotter.show()


def compare_meshes(file_paths):
    """
    Compare multiple meshes side by side
    
    Args:
        file_paths: List of paths to mesh files
    """
    n_meshes = len(file_paths)
    plotter = pv.Plotter(shape=(1, n_meshes))
    
    for i, file_path in enumerate(file_paths):
        mesh = pv.read(file_path)
        plotter.subplot(0, i)
        plotter.add_text(os.path.basename(file_path), font_size=10)
        plotter.add_mesh(mesh, color='red', opacity=0.6)
        plotter.add_axes()
    
    plotter.link_views()
    plotter.show()


def export_screenshot(file_path, output_path=None):
    """
    Export a screenshot of the mesh
    
    Args:
        file_path: Path to mesh file
        output_path: Path to save screenshot (default: same name with .png)
    """
    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '_screenshot.png'
    
    mesh = pv.read(file_path)
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color='red', opacity=0.8)
    plotter.add_axes()
    plotter.camera_position = 'iso'
    plotter.screenshot(output_path)
    print(f"Screenshot saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single mesh:  python visualize_mesh.py <path_to_mesh.vtp>")
        print("  Compare:      python visualize_mesh.py <mesh1.vtp> <mesh2.vtp> ...")
        print("  Screenshot:   python visualize_mesh.py <mesh.vtp> --screenshot")
        print("\nOptions:")
        print("  --edges         Show mesh edges")
        print("  --no-centerline Don't show centerline visualization")
        print("\nExample:")
        print("  python visualize_mesh.py vmr_downloads/0166_0001/Meshes/0166_0001.vtp")
        sys.exit(1)
    
    # Parse arguments
    show_edges = '--edges' in sys.argv
    show_centerlines = '--no-centerline' not in sys.argv
    screenshot = '--screenshot' in sys.argv
    
    # Get file paths (excluding flags)
    file_paths = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if screenshot:
        export_screenshot(file_paths[0])
    elif len(file_paths) > 1:
        compare_meshes(file_paths)
    else:
        visualize_mesh(file_paths[0], show_edges=show_edges, show_centerlines=show_centerlines)

