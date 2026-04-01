"""
Verify that extracted centerline JSON files contain radius information
Usage: python verify_radius_data.py <path_to_json_files_or_folders>
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def verify_radius(json_file: Path) -> Tuple[bool, dict]:
    """
    Verify if a JSON file contains radius data
    
    Returns:
        (has_radius, info_dict)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, {"error": str(e)}
    
    for markup in data.get("markups", []):
        for measure in markup.get("measurements", []):
            if measure.get("name") == "Radius":
                radii = measure.get("controlPointValues", [])
                if radii:
                    return True, {
                        "points": len(radii),
                        "min_radius": min(radii),
                        "max_radius": max(radii),
                        "avg_radius": sum(radii) / len(radii)
                    }
    
    return False, {"error": "No radius measurement found"}


def verify_all(paths: List[str]) -> None:
    """
    Verify all JSON files in given paths
    """
    # Collect all JSON files
    json_files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.json':
            json_files.append(path)
        elif path.is_dir():
            json_files.extend(path.rglob("*.mrk.json"))
            json_files.extend(path.rglob("*centerline*.json"))
    
    if not json_files:
        print("❌ No JSON files found!")
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files to verify\n")
    print("="*70)
    
    # Verify each file
    results = []
    for json_file in sorted(set(json_files)):
        has_radius, info = verify_radius(json_file)
        results.append(has_radius)
        
        if has_radius:
            print(f"✅ {json_file.name}")
            print(f"   Points: {info['points']}")
            print(f"   Radius: {info['min_radius']:.2f} - {info['max_radius']:.2f} mm")
            print(f"   Average: {info['avg_radius']:.2f} mm")
        else:
            print(f"❌ {json_file.name}")
            print(f"   Error: {info.get('error', 'Unknown error')}")
        print()
    
    # Summary
    print("="*70)
    print(f"\nSummary: {sum(results)}/{len(results)} files have radius data")
    
    if all(results):
        print("✅ All files have radius data!")
        sys.exit(0)
    else:
        print("❌ Some files are missing radius data!")
        print("\nFiles without radius:")
        for json_file, has_radius in zip(json_files, results):
            if not has_radius:
                print(f"  - {json_file}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Verify Radius Data in Centerline JSON Files")
        print("\nUsage:")
        print("  python verify_radius_data.py <file_or_folder> [file_or_folder ...]")
        print("\nExamples:")
        print("  # Verify single file")
        print("  python verify_radius_data.py centerlines.mrk.json")
        print()
        print("  # Verify all files in folder")
        print("  python verify_radius_data.py vmr_downloads/0105_0001/")
        print()
        print("  # Verify multiple folders")
        print("  python verify_radius_data.py vmr_downloads/*/centerlines_vmtk/")
        print()
        print("  # Verify all downloaded models")
        print("  python verify_radius_data.py vmr_downloads/")
        sys.exit(1)
    
    verify_all(sys.argv[1:])


