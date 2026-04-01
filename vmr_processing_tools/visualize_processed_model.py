"""
Visualize Processed DualDeviceNav Model

This script visualizes the processed model data (OBJ mesh + JSON centerlines)
to verify that the processing was successful.

Features:
- Shows collision and/or visual mesh
- Overlays all centerlines with radius information
- Color-codes different branches
- Interactive 3D view with PyVista

Usage:
    python visualize_processed_model.py <path_to_dualdevicenav_format_folder>
    
Example:
    python visualize_processed_model.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
"""

import os
import sys
import json
import numpy as np
import pyvista as pv
from pathlib import Path


def load_centerline_from_json(json_file):
    """
    Load centerline coordinates and radius from JSON file
    
    Returns:
        points: Nx3 array of coordinates
        radii: N array of radius values
        name: Branch name
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    radii = []
    name = os.path.basename(json_file)
    
    for markup in data['markups']:
        if markup['type'] == 'Curve':
            # Extract points
            for point in markup['controlPoints']:
                pos = point['position']
                points.append([pos[0], pos[1], pos[2]])
            
            # Extract radius
            if 'measurements' in markup:
                for measurement in markup['measurements']:
                    if measurement['name'] == 'Radius':
                        radii.extend(measurement['controlPointValues'])
    
    points = np.array(points)
    radii = np.array(radii) if radii else None
    
    return points, radii, name


def create_centerline_tube(points, radii=None, color='red', tube_radius=None):
    """
    Create a tube representation of a centerline
    
    Args:
        points: Nx3 array of centerline points
        radii: N array of vessel radius at each point
        color: Color for the tube
        tube_radius: Fixed tube radius (if None, uses radii or default)
    """
    if len(points) < 2:
        return None
    
    # Create spline through points
    spline = pv.Spline(points, n_points=len(points)*5)
    
    # Determine tube radius
    if tube_radius is not None:
        radius = tube_radius
    elif radii is not None:
        # Use average radius for visualization
        radius = np.mean(radii) * 0.5  # Scale down for better visibility
    else:
        radius = 1.0  # Default radius
    
    # Create tube
    tube = spline.tube(radius=radius)
    
    return tube


def visualize_processed_model(model_folder, show_collision=True, show_visual=False, 
                              show_centerlines=True, tube_radius=None):
    """
    Visualize the processed DualDeviceNav model
    
    Args:
        model_folder: Path to dualdevicenav_format folder
        show_collision: Show collision mesh
        show_visual: Show visual mesh
        show_centerlines: Show centerlines with radius
        tube_radius: Fixed tube radius (None = auto from radius data)
    """
    print("="*80)
    print("Visualizing Processed DualDeviceNav Model")
    print("="*80)
    print(f"Model folder: {model_folder}\n")
    
    # Initialize plotter
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Load and add meshes
    collision_mesh_path = os.path.join(model_folder, "vessel_architecture_collision.obj")
    visual_mesh_path = os.path.join(model_folder, "vessel_architecture_visual.obj")
    
    mesh_added = False
    
    if show_collision and os.path.exists(collision_mesh_path):
        print(f"Loading collision mesh: {collision_mesh_path}")
        collision_mesh = pv.read(collision_mesh_path)
        plotter.add_mesh(
            collision_mesh,
            color='lightblue',
            opacity=0.3,
            label='Collision Mesh',
            show_edges=True
        )
        mesh_added = True
        print(f"  Points: {collision_mesh.n_points:,}")
        print(f"  Cells: {collision_mesh.n_cells:,}")
    
    if show_visual and os.path.exists(visual_mesh_path):
        print(f"\nLoading visual mesh: {visual_mesh_path}")
        visual_mesh = pv.read(visual_mesh_path)
        plotter.add_mesh(
            visual_mesh,
            color='lightcoral',
            opacity=0.3,
            label='Visual Mesh',
            show_edges=False
        )
        mesh_added = True
        print(f"  Points: {visual_mesh.n_points:,}")
        print(f"  Cells: {visual_mesh.n_cells:,}")
    
    if not mesh_added:
        print("[WARNING] No mesh files found or selected for display")
    
    # Load and add centerlines
    if show_centerlines:
        centerlines_folder = os.path.join(model_folder, "Centrelines")
        
        if os.path.exists(centerlines_folder):
            print(f"\nLoading centerlines from: {centerlines_folder}")
            
            # Get all JSON files
            json_files = sorted([
                f for f in os.listdir(centerlines_folder)
                if f.endswith('.json')
            ])
            
            if not json_files:
                print("[WARNING] No JSON centerline files found")
            else:
                print(f"Found {len(json_files)} centerline files\n")
                
                # Color palette for different branches
                colors = [
                    'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
                    'orange', 'purple', 'brown', 'pink', 'lime', 'navy'
                ]
                
                for idx, json_file in enumerate(json_files):
                    json_path = os.path.join(centerlines_folder, json_file)
                    
                    try:
                        points, radii, name = load_centerline_from_json(json_path)
                        
                        if len(points) > 1:
                            color = colors[idx % len(colors)]
                            
                            # Create tube visualization
                            tube = create_centerline_tube(points, radii, color, tube_radius)
                            
                            if tube is not None:
                                plotter.add_mesh(
                                    tube,
                                    color=color,
                                    opacity=0.8,
                                    label=f'Branch {idx+1}',
                                    smooth_shading=True
                                )
                            
                            # Add points
                            point_cloud = pv.PolyData(points)
                            plotter.add_mesh(
                                point_cloud,
                                color=color,
                                point_size=5,
                                render_points_as_spheres=True
                            )
                            
                            # Print info
                            if radii is not None and len(radii) > 0:
                                print(f"  [{idx+1}] {name}")
                                print(f"      Points: {len(points)}")
                                print(f"      Radius: {np.min(radii):.2f} - {np.max(radii):.2f} mm "
                                      f"(mean: {np.mean(radii):.2f} mm)")
                            else:
                                print(f"  [{idx+1}] {name}")
                                print(f"      Points: {len(points)}")
                                print(f"      Radius: No radius data")
                        
                    except Exception as e:
                        print(f"  [ERROR] Failed to load {json_file}: {e}")
        else:
            print(f"[WARNING] Centerlines folder not found: {centerlines_folder}")
    
    # Add axes and labels
    plotter.add_axes(
        xlabel='X (mm)',
        ylabel='Y (mm)',
        zlabel='Z (mm)',
        line_width=5,
        labels_off=False
    )
    
    # Add legend
    plotter.add_legend(
        size=(0.2, 0.2),
        loc='upper right'
    )
    
    # Show coordinate system info
    print("\n" + "="*80)
    print("Visualization Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - Right mouse drag: Pan")
    print("  - 'q' or close window: Exit")
    print("="*80)
    print("\nCoordinate System: LPS (Left-Posterior-Superior)")
    print("Note: This is the format before stEVE applies (y, -z, -x) transformation")
    print("="*80)
    
    # Show the plot
    plotter.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_processed_model.py <path_to_dualdevicenav_format_folder>")
        print("\nExample:")
        print("  python visualize_processed_model.py D:\\vmr\\vmr\\0011_H_AO_H\\dualdevicenav_format")
        print("\nOptions:")
        print("  Add 'collision' to show only collision mesh")
        print("  Add 'visual' to show only visual mesh")
        print("  Add 'both' to show both meshes")
        print("  Add 'centerlines_only' to show only centerlines")
        sys.exit(1)
    
    model_folder = sys.argv[1]
    
    if not os.path.exists(model_folder):
        print(f"ERROR: Folder not found: {model_folder}")
        sys.exit(1)
    
    # Parse options
    args = [arg.lower() for arg in sys.argv[2:]]
    
    if 'centerlines_only' in args:
        show_collision = False
        show_visual = False
        show_centerlines = True
    elif 'collision' in args and 'visual' not in args and 'both' not in args:
        show_collision = True
        show_visual = False
        show_centerlines = True
    elif 'visual' in args and 'collision' not in args and 'both' not in args:
        show_collision = False
        show_visual = True
        show_centerlines = True
    elif 'both' in args:
        show_collision = True
        show_visual = True
        show_centerlines = True
    else:
        # Default: show collision mesh + centerlines
        show_collision = True
        show_visual = False
        show_centerlines = True
    
    visualize_processed_model(
        model_folder,
        show_collision=show_collision,
        show_visual=show_visual,
        show_centerlines=show_centerlines
    )


if __name__ == "__main__":
    main()



