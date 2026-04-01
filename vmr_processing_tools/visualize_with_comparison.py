"""
Advanced Visualization with Side-by-Side Comparison

This script provides multiple visualization modes:
1. Single model view (mesh + centerlines)
2. Centerlines only with radius color-coding
3. Multiple models comparison (if available)

Usage:
    python visualize_with_comparison.py <model_folder> [options]
    
Options:
    --radius-colormap    Color-code centerlines by radius value
    --no-mesh           Show only centerlines
    --split-view        Show collision and visual mesh side-by-side
    
Examples:
    # Basic visualization
    python visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format
    
    # Centerlines with radius color-coding
    python visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format --radius-colormap
    
    # Centerlines only
    python visualize_with_comparison.py D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format --no-mesh
"""

import os
import sys
import json
import numpy as np
import pyvista as pv
from pathlib import Path


def load_centerline_from_json(json_file):
    """Load centerline from JSON with radius data"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    points = []
    radii = []
    
    for markup in data['markups']:
        if markup['type'] == 'Curve':
            for point in markup['controlPoints']:
                pos = point['position']
                points.append([pos[0], pos[1], pos[2]])
            
            if 'measurements' in markup:
                for measurement in markup['measurements']:
                    if measurement['name'] == 'Radius':
                        radii.extend(measurement['controlPointValues'])
    
    return np.array(points), np.array(radii) if radii else None


def visualize_with_radius_colormap(model_folder):
    """Visualize centerlines with radius values shown as colors"""
    print("="*80)
    print("Radius Color-Coded Visualization")
    print("="*80)
    
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    centerlines_folder = os.path.join(model_folder, "Centrelines")
    json_files = sorted([
        f for f in os.listdir(centerlines_folder)
        if f.endswith('.json')
    ])
    
    all_radii = []
    centerline_data = []
    
    # First pass: collect all radius data for consistent color scale
    for json_file in json_files:
        json_path = os.path.join(centerlines_folder, json_file)
        points, radii = load_centerline_from_json(json_path)
        if radii is not None and len(radii) > 0:
            all_radii.extend(radii)
            centerline_data.append((points, radii, json_file))
    
    if not all_radii:
        print("[ERROR] No radius data found in centerlines")
        return
    
    min_radius = np.min(all_radii)
    max_radius = np.max(all_radii)
    
    print(f"\nGlobal radius range: {min_radius:.2f} - {max_radius:.2f} mm")
    print(f"Found {len(centerline_data)} centerlines with radius data\n")
    
    # Second pass: create visualizations with consistent color scale
    for idx, (points, radii, name) in enumerate(centerline_data):
        if len(points) > 1:
            # Create spline
            spline = pv.Spline(points, n_points=len(points)*5)
            
            # Interpolate radius values along spline
            spline_points = spline.points
            interp_radii = np.interp(
                np.linspace(0, 1, len(spline_points)),
                np.linspace(0, 1, len(radii)),
                radii
            )
            
            # Create tubes with varying radius
            tubes = []
            n_segments = len(spline_points) - 1
            
            for i in range(n_segments):
                p1 = spline_points[i]
                p2 = spline_points[i+1]
                r = interp_radii[i] * 0.3  # Scale for visibility
                
                # Create small tube segment
                line = pv.Line(p1, p2)
                tube = line.tube(radius=r, n_sides=20)
                tube['radius'] = np.full(tube.n_points, interp_radii[i])
                tubes.append(tube)
            
            # Merge tubes
            merged = tubes[0]
            for tube in tubes[1:]:
                merged = merged.merge(tube)
            
            # Add to plotter with color mapping
            plotter.add_mesh(
                merged,
                scalars='radius',
                cmap='jet',
                clim=[min_radius, max_radius],
                scalar_bar_args={
                    'title': 'Radius (mm)',
                    'vertical': True,
                    'position_x': 0.85,
                    'position_y': 0.05,
                    'width': 0.1,
                    'height': 0.9
                },
                smooth_shading=True
            )
            
            print(f"  [{idx+1}] {name}: {len(points)} points, "
                  f"radius {np.min(radii):.2f}-{np.max(radii):.2f} mm")
    
    # Add axes
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z', line_width=5)
    
    print("\n" + "="*80)
    print("Color scale: Blue (small radius) -> Red (large radius)")
    print("="*80)
    
    plotter.show()


def visualize_split_view(model_folder):
    """Show collision and visual mesh side-by-side"""
    print("="*80)
    print("Split View: Collision vs Visual Mesh")
    print("="*80)
    
    collision_path = os.path.join(model_folder, "vessel_architecture_collision.obj")
    visual_path = os.path.join(model_folder, "vessel_architecture_visual.obj")
    
    if not os.path.exists(collision_path) or not os.path.exists(visual_path):
        print("[ERROR] Both collision and visual mesh files required for split view")
        return
    
    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background('white')
    
    # Left: Collision mesh
    plotter.subplot(0, 0)
    collision = pv.read(collision_path)
    plotter.add_text("Collision Mesh", font_size=12, position='upper_edge')
    plotter.add_mesh(collision, color='lightblue', show_edges=True)
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    print(f"\nCollision Mesh:")
    print(f"  Points: {collision.n_points:,}")
    print(f"  Cells: {collision.n_cells:,}")
    
    # Right: Visual mesh
    plotter.subplot(0, 1)
    visual = pv.read(visual_path)
    plotter.add_text("Visual Mesh", font_size=12, position='upper_edge')
    plotter.add_mesh(visual, color='lightcoral', show_edges=False, smooth_shading=True)
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    
    print(f"\nVisual Mesh:")
    print(f"  Points: {visual.n_points:,}")
    print(f"  Cells: {visual.n_cells:,}")
    
    # Link cameras
    plotter.link_views()
    
    print("\n" + "="*80)
    print("Views are linked - rotating one will rotate both")
    print("="*80)
    
    plotter.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced visualization of processed DualDeviceNav models'
    )
    parser.add_argument(
        'model_folder',
        help='Path to dualdevicenav_format folder'
    )
    parser.add_argument(
        '--radius-colormap',
        action='store_true',
        help='Color-code centerlines by radius value'
    )
    parser.add_argument(
        '--no-mesh',
        action='store_true',
        help='Show only centerlines (no mesh)'
    )
    parser.add_argument(
        '--split-view',
        action='store_true',
        help='Show collision and visual mesh side-by-side'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_folder):
        print(f"ERROR: Folder not found: {args.model_folder}")
        sys.exit(1)
    
    # Choose visualization mode
    if args.split_view:
        visualize_split_view(args.model_folder)
    elif args.radius_colormap:
        visualize_with_radius_colormap(args.model_folder)
    else:
        # Import and use the basic visualization
        from visualize_processed_model import visualize_processed_model
        visualize_processed_model(
            args.model_folder,
            show_collision=not args.no_mesh,
            show_visual=False,
            show_centerlines=True
        )


if __name__ == "__main__":
    main()



