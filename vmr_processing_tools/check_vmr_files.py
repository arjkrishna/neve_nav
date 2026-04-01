"""
Check VMR Models for Required Files

This script checks all VMR models to verify they have:
1. VTP files (for mesh)
2. PTH files (for centerlines)

Usage:
    python check_vmr_files.py <vmr_root_folder>
    
Example:
    python check_vmr_files.py D:\vmr\vmr
"""

import os
import sys
from pathlib import Path


def check_vmr_model(model_folder):
    """
    Check a single VMR model for required files
    
    Returns:
        dict: Status of files in this model
    """
    model_name = os.path.basename(model_folder)
    
    # Check for VTP files
    vtp_files = []
    meshes_folder = os.path.join(model_folder, "Meshes")
    models_folder = os.path.join(model_folder, "Models")
    
    if os.path.exists(meshes_folder):
        vtp_files.extend(list(Path(meshes_folder).glob("*.vtp")))
    if os.path.exists(models_folder):
        vtp_files.extend(list(Path(models_folder).glob("*.vtp")))
    
    # Check for PTH files
    pth_files = []
    paths_folder = os.path.join(model_folder, "Paths")
    
    if os.path.exists(paths_folder):
        pth_files = list(Path(paths_folder).glob("*.pth"))
    
    return {
        "name": model_name,
        "vtp_count": len(vtp_files),
        "vtp_files": [f.name for f in vtp_files],
        "pth_count": len(pth_files),
        "pth_files": [f.name for f in pth_files],
        "has_vtp": len(vtp_files) > 0,
        "has_pth": len(pth_files) > 0,
        "ready": len(vtp_files) > 0 and len(pth_files) > 0
    }


def check_all_models(vmr_root):
    """
    Check all VMR models in the root folder
    
    Args:
        vmr_root: Path to VMR root folder containing model subfolders
    """
    print("="*80)
    print("Checking VMR Models for Required Files")
    print("="*80)
    print(f"Root folder: {vmr_root}\n")
    
    if not os.path.exists(vmr_root):
        print(f"[ERROR] Folder not found: {vmr_root}")
        return
    
    # Get all subdirectories
    model_folders = [f for f in Path(vmr_root).iterdir() if f.is_dir()]
    
    if not model_folders:
        print(f"[ERROR] No model folders found in {vmr_root}")
        return
    
    print(f"Found {len(model_folders)} model folders\n")
    
    # Check each model
    results = []
    for model_folder in sorted(model_folders):
        result = check_vmr_model(str(model_folder))
        results.append(result)
    
    # Summary statistics
    ready_models = [r for r in results if r["ready"]]
    vtp_only = [r for r in results if r["has_vtp"] and not r["has_pth"]]
    pth_only = [r for r in results if r["has_pth"] and not r["has_vtp"]]
    missing_both = [r for r in results if not r["has_vtp"] and not r["has_pth"]]
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total models:              {len(results)}")
    print(f"Ready for processing:      {len(ready_models)} (has VTP + PTH)")
    print(f"Has VTP only:              {len(vtp_only)} (missing PTH)")
    print(f"Has PTH only:              {len(pth_only)} (missing VTP)")
    print(f"Missing both:              {len(missing_both)}")
    
    # Detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    if ready_models:
        print(f"\n[READY] {len(ready_models)} models ready for processing:")
        print("-"*80)
        for r in ready_models:
            print(f"  {r['name']:30s} | VTP: {r['vtp_count']} | PTH: {r['pth_count']}")
    
    if vtp_only:
        print(f"\n[WARNING] {len(vtp_only)} models have VTP but no PTH:")
        print("-"*80)
        for r in vtp_only:
            print(f"  {r['name']:30s} | VTP: {r['vtp_count']} | PTH: 0")
        print("\n  Note: These can still be processed, but require manual endpoint selection")
    
    if pth_only:
        print(f"\n[WARNING] {len(pth_only)} models have PTH but no VTP:")
        print("-"*80)
        for r in pth_only:
            print(f"  {r['name']:30s} | VTP: 0 | PTH: {r['pth_count']}")
        print("\n  Note: Cannot process without VTP mesh files")
    
    if missing_both:
        print(f"\n[ERROR] {len(missing_both)} models missing both VTP and PTH:")
        print("-"*80)
        for r in missing_both:
            print(f"  {r['name']}")
    
    # Save results to file
    output_file = "vmr_file_check_results.txt"
    with open(output_file, 'w') as f:
        f.write("VMR Models File Check Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total models: {len(results)}\n")
        f.write(f"Ready for processing: {len(ready_models)}\n\n")
        
        f.write("READY MODELS:\n")
        f.write("-"*80 + "\n")
        for r in ready_models:
            f.write(f"{r['name']:30s} | VTP: {r['vtp_count']} files | PTH: {r['pth_count']} files\n")
            f.write(f"  VTP files: {', '.join(r['vtp_files'])}\n")
            f.write(f"  PTH files: {', '.join(r['pth_files'])}\n\n")
        
        if vtp_only:
            f.write("\n\nVTP ONLY (no PTH):\n")
            f.write("-"*80 + "\n")
            for r in vtp_only:
                f.write(f"{r['name']:30s} | VTP: {r['vtp_count']} files\n")
        
        if pth_only:
            f.write("\n\nPTH ONLY (no VTP):\n")
            f.write("-"*80 + "\n")
            for r in pth_only:
                f.write(f"{r['name']:30s} | PTH: {r['pth_count']} files\n")
        
        if missing_both:
            f.write("\n\nMISSING BOTH:\n")
            f.write("-"*80 + "\n")
            for r in missing_both:
                f.write(f"{r['name']}\n")
    
    print(f"\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_vmr_files.py <vmr_root_folder>")
        print("Example: python check_vmr_files.py D:\\vmr\\vmr")
        sys.exit(1)
    
    vmr_root = sys.argv[1]
    check_all_models(vmr_root)


if __name__ == "__main__":
    main()


