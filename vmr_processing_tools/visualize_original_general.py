"""
Visualize Original Data - Generic Model

This script shows the original VTP mesh with extracted centerlines
BEFORE any transformations, so you can verify the centerline extraction
was successful.

Usage:
    python visualize_original_general.py --model_name 0011
    python visualize_original_general.py --model_name 0011_H_AO_H [--mode compare]
    
    Or in Docker:
    docker run ... python3 .../visualize_original_general.py --model_name 0011
"""

import os
import sys
import argparse
from pathlib import Path

# Add the script directory to path to allow importing from same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the visualization functions from the original script
from visualize_original_vs_processed import visualize_original, visualize_comparison


def find_model_folder(vmr_root: str, model_identifier: str) -> str:
    """
    Find the model folder given a model identifier.
    
    Args:
        vmr_root: Root directory containing VMR models (e.g., "D:\\vmr\\vmr" or "/vmr_host/vmr")
        model_identifier: Model name or number (e.g., "0011" or "0011_H_AO_H")
    
    Returns:
        str: Path to model folder
    
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
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    return str(model_folder)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Original Data - Generic Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using model number (original mode)
  python visualize_original_general.py --model_name 0011
  
  # Using full model name (compare mode)
  python visualize_original_general.py --model_name 0011_H_AO_H --mode compare
  
  # In Docker
  docker run ... python3 .../visualize_original_general.py --model_name 0011
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
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['original', 'compare'],
        default='original',
        help="Visualization mode: 'original' (default) or 'compare' (side-by-side VTP vs OBJ)"
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
    print("Visualize Original Data")
    print("="*80)
    print(f"Model identifier: {args.model_name}")
    print(f"VMR root: {vmr_root}")
    print(f"Mode: {args.mode}")
    print("="*80)
    print()
    
    # Find model folder
    try:
        model_folder = find_model_folder(vmr_root, args.model_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"Found model folder: {model_folder}")
    print()
    
    # Run visualization
    if args.mode == 'compare':
        visualize_comparison(model_folder)
    else:
        visualize_original(model_folder)


if __name__ == "__main__":
    main()

