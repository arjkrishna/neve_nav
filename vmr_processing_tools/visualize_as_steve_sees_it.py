"""
Visualize Processed Data As stEVE Sees It (NO ROTATION VERSION)

This script shows the processed OBJ mesh with centerlines WITHOUT rotation,
for data created with create_dualdevicenav_format.py

The data is in the raw orientation (scaled only, no rotation).
All rotation will be handled by stEVE via rotation_yzx_deg parameter.

Usage:
    python visualize_as_steve_sees_it.py <dualdevicenav_format_folder>
    
Example:
    python visualize_as_steve_sees_it.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
"""

import os
import sys
import json
import numpy as np
import pyvista as pv


def load_centerline_with_mesh_transform(json_file):
    """
    Load centerline from JSON (NO ROTATION).
    
    IMPORTANT: The JSON coordinates and radii are ALREADY SCALED to mm
    (scaling happens in create_dualdevicenav_format.py when saving JSON).
    
    Transformations applied:
    1. NONE - coordinates loaded as-is from JSON
    2. NO SCALING (already done in JSON)
    3. NO ROTATION (will be handled by stEVE)
    
    This shows how the centerlines and mesh relate BEFORE stEVE loads them.
    The OBJ mesh has ONLY scaling (no rotation).
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points_transformed = []
    radii = []
    
    for markup in data['markups']:
        if markup['type'] == 'Curve':
            for point in markup['controlPoints']:
                pos = point['position']
                # Coordinates from JSON are ALREADY in mm (scaled in create_dualdevicenav_format.py)
                # NO rotation applied (will be handled by stEVE)
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2])
                
                # Use coordinates as-is (NO rotation)
                points_transformed.append([x, y, z])
            
            if 'measurements' in markup:
                for measurement in markup['measurements']:
                    if measurement['name'] == 'Radius':
                        # Radii are ALREADY in mm (scaled in create_dualdevicenav_format.py)
                        radii.extend(measurement['controlPointValues'])
    
    return np.array(points_transformed), np.array(radii) if radii else None


def visualize_as_steve_loads_it(model_folder):
    """Visualize exactly how stEVE will see the data"""
    print("="*80)
    print("Visualization: As stEVE Sees It")
    print("="*80)
    print(f"Model folder: {model_folder}\n")
    
    # Load processed OBJ mesh (already transformed)
    # Try both naming conventions
    collision_obj = os.path.join(model_folder, "vessel_architecture_collision.obj")
    visual_obj = os.path.join(model_folder, "vessel_architecture_visual.obj")
    
    # If default names don't exist, try to find any .obj files
    if not os.path.exists(collision_obj):
        # Try to find collision mesh with any name
        obj_files = [f for f in os.listdir(model_folder) if f.endswith('_collision.obj')]
        if obj_files:
            collision_obj = os.path.join(model_folder, obj_files[0])
            print(f"Found collision mesh: {obj_files[0]}")
        else:
            print(f"[ERROR] No processed collision mesh found in: {model_folder}")
            print(f"\nExpected file: vessel_architecture_collision.obj")
            print(f"Or: <model_name>_collision.obj")
            print(f"\n>>> You need to run the processing script first! <<<")
            print(f"\nRun this command:")
            print(f"  vmr_processing_tools\\run_dualdevicenav.bat")
            print(f"\nOr manually:")
            print(f"  python vmr_processing_tools\\create_dualdevicenav_format.py D:\\vmr\\vmr\\0011_H_AO_H")
            return
    
    print(f"Loading processed collision mesh...")
    collision_mesh = pv.read(collision_obj)
    print(f"  Points: {collision_mesh.n_points:,}")
    print(f"  Cells: {collision_mesh.n_cells:,}")
    print(f"  Bounds: {collision_mesh.bounds}")
    
    # Initialize plotter
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Add collision mesh
    plotter.add_mesh(
        collision_mesh,
        color='lightblue',
        opacity=0.3,
        label='Collision Mesh (OBJ)',
        show_edges=True
    )
    
    # Load and transform centerlines
    centerlines_folder = os.path.join(model_folder, "Centrelines")
    
    if not os.path.exists(centerlines_folder):
        print(f"[ERROR] Centerlines folder not found: {centerlines_folder}")
        return
    
    print(f"\nLoading centerlines...")
    print(f"NO rotation applied (clean data for stEVE processing)")
    
    # Sort files by branch number (branch 0 = main file without number, then (1), (2), etc.)
    def get_branch_number(filename):
        # "Centerline curve - model.mrk.json" = branch 0
        if '(' not in filename:
            return 0
        # "Centerline curve (N).mrk.json" = branch N
        import re
        match = re.search(r'\((\d+)\)', filename)
        return int(match.group(1)) if match else 999
    
    json_files = sorted(
        [f for f in os.listdir(centerlines_folder) if f.endswith('.json')],
        key=get_branch_number
    )
    
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
              'orange', 'purple', 'brown', 'pink', 'lime', 'navy']
    
    print(f"Found {len(json_files)} centerline files\n")
    
    all_centerline_points = []
    
    for idx, json_file in enumerate(json_files):
        json_path = os.path.join(centerlines_folder, json_file)
        
        try:
            # Load with mesh transformations (scale + rotate)
            points, radii = load_centerline_with_mesh_transform(json_path)
            all_centerline_points.extend(points)
            
            if len(points) > 1:
                color = colors[idx % len(colors)]
                
                # Create spline
                spline = pv.Spline(points, n_points=len(points)*5)
                
                # Use radius for tube thickness
                if radii is not None and len(radii) > 0:
                    avg_radius = np.mean(radii) * 0.5  # Scale for visibility
                    tube = spline.tube(radius=avg_radius)
                    plotter.add_mesh(
                        tube,
                        color=color,
                        opacity=0.8,
                        label=f'Branch {idx+1}',
                        smooth_shading=True
                    )
                    
                    print(f"  [{idx+1}] {json_file}")
                    print(f"      Points: {len(points)}")
                    print(f"      Radius: {np.min(radii):.2f} - {np.max(radii):.2f} mm "
                          f"(mean: {np.mean(radii):.2f} mm)")
                    print(f"      Bounds: X[{np.min(points[:,0]):.1f}, {np.max(points[:,0]):.1f}] "
                          f"Y[{np.min(points[:,1]):.1f}, {np.max(points[:,1]):.1f}] "
                          f"Z[{np.min(points[:,2]):.1f}, {np.max(points[:,2]):.1f}]")
                else:
                    tube = spline.tube(radius=1.0)
                    plotter.add_mesh(tube, color=color, opacity=0.8, label=f'Branch {idx+1}')
                    print(f"  [{idx+1}] {json_file}: {len(points)} points (no radius)")
                
                # Add points
                point_cloud = pv.PolyData(points)
                plotter.add_mesh(
                    point_cloud,
                    color=color,
                    point_size=8,
                    render_points_as_spheres=True
                )
        
        except Exception as e:
            print(f"  [ERROR] Failed to load {json_file}: {e}")
    
    # Check alignment
    if all_centerline_points:
        all_points = np.array(all_centerline_points)
        mesh_bounds = collision_mesh.bounds
        points_bounds = [
            np.min(all_points[:,0]), np.max(all_points[:,0]),
            np.min(all_points[:,1]), np.max(all_points[:,1]),
            np.min(all_points[:,2]), np.max(all_points[:,2])
        ]
        
        print("\n" + "="*80)
        print("ALIGNMENT CHECK:")
        print(f"  Mesh bounds:       X[{mesh_bounds[0]:.1f}, {mesh_bounds[1]:.1f}] "
              f"Y[{mesh_bounds[2]:.1f}, {mesh_bounds[3]:.1f}] "
              f"Z[{mesh_bounds[4]:.1f}, {mesh_bounds[5]:.1f}]")
        print(f"  Centerline bounds: X[{points_bounds[0]:.1f}, {points_bounds[1]:.1f}] "
              f"Y[{points_bounds[2]:.1f}, {points_bounds[3]:.1f}] "
              f"Z[{points_bounds[4]:.1f}, {points_bounds[5]:.1f}]")
        
        # Check if centerlines are inside mesh
        margin = 20  # mm tolerance
        x_ok = (points_bounds[0] >= mesh_bounds[0] - margin and 
                points_bounds[1] <= mesh_bounds[1] + margin)
        y_ok = (points_bounds[2] >= mesh_bounds[2] - margin and 
                points_bounds[3] <= mesh_bounds[3] + margin)
        z_ok = (points_bounds[4] >= mesh_bounds[4] - margin and 
                points_bounds[5] <= mesh_bounds[5] + margin)
        
        if x_ok and y_ok and z_ok:
            print("  Status: [OK] Centerlines are inside mesh!")
        else:
            print("  Status: [WARNING] Centerlines may be outside mesh!")
            if not x_ok:
                print("    - X axis misalignment")
            if not y_ok:
                print("    - Y axis misalignment")
            if not z_ok:
                print("    - Z axis misalignment")
    
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
    print("\nNOTE: Visualization shows mesh + centerlines WITHOUT rotation")
    print("      - OBJ mesh: Scale x10 only, NO rotation")
    print("      - JSON centerlines: Scale x10 only, NO rotation")
    print("      - They should overlap perfectly!")
    print("      - stEVE will apply rotation_yzx_deg + (y,-z,-x) when loading")
    print("="*80)
    
    plotter.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_as_steve_sees_it.py <dualdevicenav_format_folder>")
        print("\nExample:")
        print("  python visualize_as_steve_sees_it.py D:\\vmr\\vmr\\0011_H_AO_H\\dualdevicenav_format")
        sys.exit(1)
    
    model_folder = sys.argv[1]
    
    if not os.path.exists(model_folder):
        print(f"ERROR: Folder not found: {model_folder}")
        sys.exit(1)
    
    visualize_as_steve_loads_it(model_folder)


if __name__ == "__main__":
    main()

