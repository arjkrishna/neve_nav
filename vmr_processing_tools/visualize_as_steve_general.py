"""
Visualize Processed Data As stEVE Sees It - Generic Model

This script shows the processed OBJ mesh with centerlines WITHOUT rotation,
for data created with create_dualdevicenav_format.py

The data is in the raw orientation (scaled only, no rotation).
All rotation will be handled by stEVE via rotation_yzx_deg parameter.

Usage:
    python visualize_as_steve_general.py --model_name 0011
    python visualize_as_steve_general.py --model_name 0011_H_AO_H
    
    Or in Docker:
    docker run ... python3 .../visualize_as_steve_general.py --model_name 0011
"""

import os
import sys
import argparse
from pathlib import Path

# Add the script directory to path to allow importing from same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the visualization function from the original script
from visualize_as_steve_sees_it import visualize_as_steve_loads_it


def find_model_folder(vmr_root: str, model_identifier: str) -> str:
    """
    Find the dualdevicenav_format folder given a model identifier.
    
    Args:
        vmr_root: Root directory containing VMR models (e.g., "D:\\vmr\\vmr" or "/vmr_host/vmr")
        model_identifier: Model name or number (e.g., "0011" or "0011_H_AO_H")
    
    Returns:
        str: Path to dualdevicenav_format folder
    
    Raises:
        FileNotFoundError: If model not found
    """
    vmr_root_path = Path(vmr_root)
    
    if not vmr_root_path.exists():
        raise FileNotFoundError(f"VMR root directory not found: {vmr_root}")
    
    # If model_identifier looks like a full name (contains underscores), use it directly
    if '_' in model_identifier:
        full_model_name = model_identifier
        model_folder = vmr_root_path / full_model_name
    else:
        # Search for folders starting with the number
        matching_folders = []
        for folder in vmr_root_path.iterdir():
            if folder.is_dir() and folder.name.startswith(model_identifier + '_'):
                matching_folders.append(folder)
        
        if len(matching_folders) == 0:
            available = sorted([f.name for f in vmr_root_path.iterdir() if f.is_dir()])[:10]
            raise FileNotFoundError(
                f"No model found starting with '{model_identifier}' in {vmr_root}\n"
                f"Available models: {available}..."
            )
        elif len(matching_folders) > 1:
            print(f"WARNING: Multiple models found starting with '{model_identifier}':")
            for folder in matching_folders:
                print(f"  - {folder.name}")
            print(f"Using first match: {matching_folders[0].name}")
            full_model_name = matching_folders[0].name
            model_folder = matching_folders[0]
        else:
            full_model_name = matching_folders[0].name
            model_folder = matching_folders[0]
    
    dualdevicenav_format = model_folder / "dualdevicenav_format"
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    if not dualdevicenav_format.exists():
        raise FileNotFoundError(
            f"Processed data not found: {dualdevicenav_format}\n"
            f"Please run the processing script first:\n"
            f"  vmr_processing_tools\\run_dualdevicenav.bat {model_folder}"
        )
    
    return str(dualdevicenav_format)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Processed Data As stEVE Sees It - Generic Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using model number
  python visualize_as_steve_general.py --model_name 0011
  
  # Using full model name
  python visualize_as_steve_general.py --model_name 0011_H_AO_H
  
  # In Docker
  docker run ... python3 .../visualize_as_steve_general.py --model_name 0011
        """
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model identifier: number (e.g., '0011') or full name (e.g., '0011_H_AO_H')"
    )
    
    parser.add_argument(
        "--vmr_root",
        type=str,
        default=None,
        help="Root directory containing VMR models (auto-detects Docker vs Windows)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect VMR root (Docker vs Windows)
    if args.vmr_root:
        vmr_root = args.vmr_root
    elif os.path.exists("/vmr_host/vmr"):
        vmr_root = "/vmr_host/vmr"
    else:
        vmr_root = r"D:\vmr\vmr"
    
    print("="*80)
    print("Visualize Processed Data As stEVE Sees It")
    print("="*80)
    print(f"Model identifier: {args.model_name}")
    print(f"VMR root: {vmr_root}")
    print("="*80)
    print()
    
    # Find model folder
    try:
        dualdevicenav_format = find_model_folder(vmr_root, args.model_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Found processed data: {dualdevicenav_format}")
    print()
    
    # Run visualization
    visualize_as_steve_loads_it(dualdevicenav_format)


if __name__ == "__main__":
    main()

