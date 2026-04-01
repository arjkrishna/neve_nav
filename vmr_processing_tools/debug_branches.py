"""
Quick debug script to check branch loading and insertion point detection
"""
import sys
import os
import json
import numpy as np
import re

# Define the load_branches function locally
def load_points_from_json(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    points = []
    radii = []
    for markup in data["markups"]:
        if markup["type"] == "Curve":
            control_points = markup["controlPoints"]
            for point in control_points:
                position = point["position"]
                x = float(position[0])
                y = float(position[1])
                z = float(position[2])
                points.append((y, -z, -x))  # Transform to stEVE coordinates

            if "measurements" in markup:
                measurements = markup["measurements"]
                for measurement in measurements:
                    if measurement["name"] == "Radius":
                        radii.extend(measurement["controlPointValues"])

    points = np.array(points, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32) if radii else None
    filename = os.path.splitext(os.path.basename(json_file_path))[0]
    
    # Simple branch class
    class Branch:
        def __init__(self, name, coords, radii):
            self.name = name
            self.coordinates = coords
            self.radii = radii
    
    return Branch(filename, points, radii)

def load_branches(folder_path):
    # Get all matching files
    files = [f for f in os.listdir(folder_path) 
             if f.startswith("Centerline curve ") and f.endswith(".json")]
    
    # Sort by branch number
    def get_branch_number(filename):
        if '(' not in filename:
            return 0
        match = re.search(r'\((\d+)\)', filename)
        return int(match.group(1)) if match else 999
    
    files.sort(key=get_branch_number)
    
    # Load branches
    centerlines = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        centerline = load_points_from_json(file_path)
        centerlines.append(centerline)
    
    return centerlines

MODEL_0011_PATH = r"D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format"
centerline_folder = os.path.join(MODEL_0011_PATH, "Centrelines")

print("="*80)
print("Loading branches from:", centerline_folder)
print("="*80)

# List files to see order
import os
files = [f for f in os.listdir(centerline_folder) if f.endswith('.json')]
print("\nFiles found (unsorted):")
for f in files:
    print(f"  {f}")

# Load branches using the fixed function
branches = load_branches(centerline_folder)

print(f"\nLoaded {len(branches)} branches (in sorted order):")
print("="*80)

for i, branch in enumerate(branches):
    print(f"\nBranch {i}: {branch.name}")
    print(f"  Points: {len(branch.coordinates)}")
    print(f"  First point (insertion if i==0): {branch.coordinates[0]}")
    print(f"  Last point: {branch.coordinates[-1]}")
    if hasattr(branch, 'radii') and branch.radii is not None:
        print(f"  Radius range: {min(branch.radii):.2f} - {max(branch.radii):.2f} mm")

print("\n" + "="*80)
print("INSERTION POINT (from branch 0):")
print(f"  {branches[0].coordinates[0]}")
print("="*80)

# Check mesh bounds
import pyvista as pv
mesh_path = os.path.join(MODEL_0011_PATH, "0011_H_AO_H_collision.obj")
mesh = pv.read(mesh_path)
bounds = mesh.bounds
center = mesh.center

print("\nMESH BOUNDS:")
print(f"  X: {bounds[0]:.2f} to {bounds[1]:.2f}")
print(f"  Y: {bounds[2]:.2f} to {bounds[3]:.2f}")
print(f"  Z: {bounds[4]:.2f} to {bounds[5]:.2f}")
print(f"  Center: {center}")

print("\n" + "="*80)
print("CHECKING if insertion point is within mesh bounds...")
ins = branches[0].coordinates[0]
if (bounds[0] <= ins[0] <= bounds[1] and
    bounds[2] <= ins[1] <= bounds[3] and
    bounds[4] <= ins[2] <= bounds[5]):
    print("  [OK] Insertion point is INSIDE mesh bounds")
else:
    print("  [ERROR] Insertion point is OUTSIDE mesh bounds!")
    print(f"  Distance from center: {((ins[0]-center[0])**2 + (ins[1]-center[1])**2 + (ins[2]-center[2])**2)**0.5:.2f}")
print("="*80)

