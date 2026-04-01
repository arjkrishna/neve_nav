"""
Batch Process All 44 Verified VMR Models

This script processes all 44 verified models from vmr_file_check_results.txt
by calling create_dualdevicenav_format.py for each model.

Usage:
    python process_all_verified_models.py [--vmr-root D:\vmr\vmr] [--skip-existing]
    
    Or use the batch file:
    process_all_verified_models.bat [--vmr-root D:\vmr\vmr] [--skip-existing]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

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
    
    return models


def process_model(vmr_root: str, model_name: str, python_script: str, skip_existing: bool = False) -> dict:
    """
    Process a single model by calling the Python script directly.
    
    Returns:
        dict with 'success', 'model', 'error' keys
    """
    model_folder = os.path.join(vmr_root, model_name)
    output_folder = os.path.join(model_folder, "dualdevicenav_format")
    
    # Check if already processed
    if skip_existing and os.path.exists(output_folder):
        centerlines_folder = os.path.join(output_folder, "Centrelines")
        collision_obj = os.path.join(output_folder, f"{model_name}_collision.obj")
        if os.path.exists(centerlines_folder) and os.path.exists(collision_obj):
            # Check if there are JSON files
            json_files = list(Path(centerlines_folder).glob("*.json"))
            if json_files:
                return {
                    'success': True,
                    'model': model_name,
                    'skipped': True,
                    'message': 'Already processed'
                }
    
    # Check if model folder exists
    if not os.path.exists(model_folder):
        return {
            'success': False,
            'model': model_name,
            'error': f'Model folder not found: {model_folder}'
        }
    
    # Call the Python script directly (avoiding batch file pause)
    script_path = os.path.join(os.path.dirname(__file__), python_script)
    if not os.path.exists(script_path):
        return {
            'success': False,
            'model': model_name,
            'error': f'Python script not found: {python_script}'
        }
    
    try:
        # Call Python script directly
        result = subprocess.run(
            [sys.executable, script_path, model_folder],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per model
            cwd=os.path.dirname(script_path)
        )
        
        if result.returncode == 0:
            return {
                'success': True,
                'model': model_name,
                'skipped': False,
                'message': 'Processed successfully'
            }
        else:
            # Extract error message from output
            error_msg = "Unknown error"
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                # Get last meaningful error line
                for line in reversed(error_lines):
                    if line.strip() and not line.strip().startswith('File'):
                        error_msg = line.strip()[:200]
                        break
            elif result.stdout:
                # Sometimes errors go to stdout
                error_lines = result.stdout.strip().split('\n')
                for line in reversed(error_lines):
                    if 'ERROR' in line or 'error' in line or 'Error' in line:
                        error_msg = line.strip()[:200]
                        break
            
            return {
                'success': False,
                'model': model_name,
                'error': f'Processing failed: {error_msg}',
                'stderr': result.stderr[-500:] if result.stderr else None,
                'stdout': result.stdout[-500:] if result.stdout else None
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'model': model_name,
            'error': 'Processing timed out (>30 minutes)'
        }
    except Exception as e:
        return {
            'success': False,
            'model': model_name,
            'error': f'Exception: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all 44 verified VMR models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--vmr-root",
        default=DEFAULT_VMR_ROOT,
        help=f"Root folder containing VMR models (default: {DEFAULT_VMR_ROOT})"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that are already processed"
    )
    
    parser.add_argument(
        "--python-script",
        default="create_dualdevicenav_format.py",
        help="Python script to call for each model (default: create_dualdevicenav_format.py)"
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
    print("Batch Processing All Verified VMR Models")
    print("="*80)
    print(f"VMR Root: {args.vmr_root}")
    print(f"Results File: {results_file}")
    print(f"Python Script: {args.python_script}")
    print(f"Skip Existing: {args.skip_existing}")
    print("="*80)
    print()
    
    # Extract verified models
    print("Extracting verified models from results file...")
    models = extract_verified_models(str(results_file))
    
    if not models:
        print("ERROR: No models found in results file!")
        sys.exit(1)
    
    print(f"Found {len(models)} verified models")
    print()
    
    # Process each model
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    start_time = datetime.now()
    
    for i, model_name in enumerate(models, 1):
        print(f"[{i}/{len(models)}] Processing: {model_name}")
        print("-" * 80)
        
        result = process_model(
            args.vmr_root,
            model_name,
            args.python_script,
            args.skip_existing
        )
        
        if result.get('skipped'):
            results['skipped'].append(result)
            print(f"  [SKIPPED] {model_name} - {result.get('message', 'Already processed')}")
        elif result['success']:
            results['success'].append(result)
            print(f"  [SUCCESS] {model_name}")
        else:
            results['failed'].append(result)
            print(f"  [FAILED]  {model_name}")
            print(f"            Error: {result.get('error', 'Unknown error')}")
            if result.get('stderr'):
                print(f"            Details: {result['stderr'][:200]}")
        
        print()
    
    # Summary
    elapsed = datetime.now() - start_time
    print("="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total models:     {len(models)}")
    print(f"Successful:       {len(results['success'])}")
    print(f"Failed:           {len(results['failed'])}")
    print(f"Skipped:          {len(results['skipped'])}")
    print(f"Time elapsed:     {elapsed}")
    print("="*80)
    
    # List failed models
    if results['failed']:
        print("\nFAILED MODELS:")
        print("-" * 80)
        for result in results['failed']:
            print(f"  {result['model']}: {result.get('error', 'Unknown error')}")
    
    # Save log
    log_file = f"batch_process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write("Batch Processing Log\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"VMR Root: {args.vmr_root}\n")
        f.write(f"Python Script: {args.python_script}\n")
        f.write(f"Total: {len(models)}, Success: {len(results['success'])}, "
                f"Failed: {len(results['failed'])}, Skipped: {len(results['skipped'])}\n")
        f.write("="*80 + "\n\n")
        
        if results['failed']:
            f.write("FAILED MODELS:\n")
            f.write("-" * 80 + "\n")
            for result in results['failed']:
                f.write(f"{result['model']}: {result.get('error', 'Unknown')}\n")
                if result.get('stderr'):
                    f.write(f"  Details: {result['stderr']}\n")
            f.write("\n")
        
        if results['success']:
            f.write("SUCCESSFUL MODELS:\n")
            f.write("-" * 80 + "\n")
            for result in results['success']:
                f.write(f"{result['model']}\n")
    
    print(f"\nLog saved to: {log_file}")
    
    # Exit with error code if any failed
    sys.exit(0 if len(results['failed']) == 0 else 1)


if __name__ == "__main__":
    main()

