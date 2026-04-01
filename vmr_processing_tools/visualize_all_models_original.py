"""
Visualize All Verified Models - Original vs Processed

This script visualizes all 44 verified models sequentially.
Press any key in the terminal to advance to the next model.

Usage:
    python visualize_all_models_original.py [--vmr-root D:\vmr\vmr] [--mode original|compare]
"""

import os
import sys
from pathlib import Path

# Add script directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import the visualization functions
from visualize_original_vs_processed import visualize_original, visualize_comparison

# Default VMR root folder
DEFAULT_VMR_ROOT = r"D:\vmr\vmr"

# Path to the results file
RESULTS_FILE = "vmr_file_check_results.txt"


def extract_verified_models(results_file: str) -> list:
    """
    Extract list of verified model names from the results file.
    
    Returns:
        List of model names (e.g., ['0001_H_AO_SVD', '0002_H_AO_SVD', ...])
    """
    models = []
    
    if not os.path.exists(results_file):
        # Try relative to script directory
        script_dir = Path(__file__).parent
        results_file = script_dir.parent / RESULTS_FILE
        if not results_file.exists():
            print(f"ERROR: Results file not found: {RESULTS_FILE}")
            return []
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    in_ready_section = False
    for line in lines:
        line = line.strip()
        
        # Check if we're in the READY MODELS section
        if "READY MODELS:" in line:
            in_ready_section = True
            continue
        
        # Stop at the next section
        if in_ready_section and line.startswith("---"):
            continue
        
        if in_ready_section and "VTP ONLY" in line:
            break
        
        # Extract model name (format: "0001_H_AO_SVD                  | VTP: 2 files | PTH: 5 files")
        if in_ready_section and "|" in line:
            model_name = line.split("|")[0].strip()
            if model_name and not model_name.startswith("VTP files") and not model_name.startswith("PTH files"):
                models.append(model_name)
    
    # Sort models (0001 first)
    models.sort()
    
    return models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize all verified models sequentially - Original vs Processed",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--vmr-root",
        default=DEFAULT_VMR_ROOT,
        help=f"Root folder containing VMR models (default: {DEFAULT_VMR_ROOT})"
    )
    
    parser.add_argument(
        "--mode",
        default="original",
        choices=["original", "compare"],
        help="Visualization mode: 'original' shows VTP with centerlines, 'compare' shows side-by-side (default: original)"
    )
    
    parser.add_argument(
        "--results-file",
        default=RESULTS_FILE,
        help=f"File containing verified models list (default: {RESULTS_FILE})"
    )
    
    args = parser.parse_args()
    
    # Find results file
    results_file = args.results_file
    if not os.path.exists(results_file):
        # Try in parent directory
        script_dir = Path(__file__).parent
        results_file = script_dir.parent / RESULTS_FILE
        if not results_file.exists():
            print(f"ERROR: Results file not found: {args.results_file}")
            print(f"       Also tried: {results_file}")
            sys.exit(1)
    
    print("="*80)
    print("Visualize All Verified Models - Original vs Processed")
    print("="*80)
    print(f"VMR Root: {args.vmr_root}")
    print(f"Mode: {args.mode}")
    print(f"Results File: {results_file}")
    print("="*80)
    print()
    
    # Extract verified models
    print("Extracting verified models from results file...")
    models = extract_verified_models(str(results_file))
    
    if not models:
        print("ERROR: No models found in results file!")
        sys.exit(1)
    
    print(f"Found {len(models)} verified models")
    print(f"Starting from: {models[0]}")
    print()
    print("="*80)
    print("INSTRUCTIONS:")
    print("  - Each model will be visualized in a PyVista window")
    print("  - Close the window or press 'q' to close")
    print("  - Then press ENTER in this terminal to view the next model")
    print("  - Press Ctrl+C to exit early")
    print("="*80)
    print()
    
    # Process each model
    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(models)}] Visualizing: {model_name}")
        print(f"{'='*80}")
        
        # Construct path to model folder
        model_folder = os.path.join(args.vmr_root, model_name)
        
        # Check if model exists
        if not os.path.exists(model_folder):
            print(f"[SKIP] Model folder not found: {model_name}")
            print(f"       Missing: {model_folder}")
            print()
            input("Press ENTER to continue to next model...")
            continue
        
        # Visualize
        try:
            if args.mode == "compare":
                visualize_comparison(model_folder)
            else:
                visualize_original(model_folder)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTED] Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Failed to visualize {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print()
            input("Press ENTER to continue to next model...")
            continue
        
        # Wait for user to press Enter before next model
        if i < len(models):
            print()
            print(f"Finished visualizing {model_name}")
            print(f"Next model: {models[i] if i < len(models) else 'N/A'}")
            try:
                input("Press ENTER to view next model (or Ctrl+C to exit)...")
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Exiting...")
                sys.exit(0)
    
    print()
    print("="*80)
    print(f"[COMPLETE] Visualized all {len(models)} models!")
    print("="*80)


if __name__ == "__main__":
    main()

