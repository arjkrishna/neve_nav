"""
Batch process multiple VMR models to DualDeviceNav format

Usage:
    python batch_create_dualdevicenav.py <vmr_downloads_folder>
    python batch_create_dualdevicenav.py vmr_downloads/ --models-list vmr_filtered_models.txt
"""

import os
import sys
import argparse
from pathlib import Path
from create_dualdevicenav_format import process_vmr_to_dualdevicenav


def batch_process(vmr_base_folder: str, 
                  models_list: str = None,
                  output_base: str = None,
                  continue_on_error: bool = True):
    """
    Batch process multiple VMR models
    
    Args:
        vmr_base_folder: Base folder containing VMR models
        models_list: Optional text file with model names (one per line)
        output_base: Base output folder
        continue_on_error: Continue processing if one model fails
    """
    # Get list of models
    if models_list and os.path.exists(models_list):
        print(f"Loading model list from: {models_list}")
        with open(models_list, 'r') as f:
            models = [line.strip() for line in f if line.strip()]
    else:
        print(f"Scanning for VMR models in: {vmr_base_folder}")
        models = [d for d in os.listdir(vmr_base_folder) 
                 if os.path.isdir(os.path.join(vmr_base_folder, d))]
    
    print(f"\nFound {len(models)} models to process\n")
    print("="*70)
    
    # Process each model
    results = {"success": [], "failed": []}
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing: {model_name}")
        print("-"*70)
        
        model_folder = os.path.join(vmr_base_folder, model_name)
        
        if not os.path.exists(model_folder):
            print(f"⚠️ Warning: Folder not found: {model_folder}")
            results["failed"].append((model_name, "Folder not found"))
            continue
        
        # Set output folder
        if output_base:
            output_folder = os.path.join(output_base, model_name, 'dualdevicenav_format')
        else:
            output_folder = os.path.join(model_folder, 'dualdevicenav_format')
        
        try:
            success = process_vmr_to_dualdevicenav(
                model_folder,
                output_folder,
                create_visual_mesh=True
            )
            
            if success:
                results["success"].append(model_name)
                print(f"✓ {model_name} - SUCCESS")
            else:
                results["failed"].append((model_name, "Processing failed"))
                print(f"✗ {model_name} - FAILED")
                
                if not continue_on_error:
                    print("\nStopping due to error (use --continue to keep going)")
                    break
                    
        except Exception as e:
            results["failed"].append((model_name, str(e)))
            print(f"✗ {model_name} - ERROR: {e}")
            
            if not continue_on_error:
                print("\nStopping due to error (use --continue to keep going)")
                break
        
        print("-"*70)
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"\nTotal models: {len(models)}")
    print(f"✓ Successful: {len(results['success'])}")
    print(f"✗ Failed:     {len(results['failed'])}")
    
    if results["success"]:
        print("\n✓ Successful models:")
        for model in results["success"]:
            print(f"  - {model}")
    
    if results["failed"]:
        print("\n✗ Failed models:")
        for model, reason in results["failed"]:
            print(f"  - {model}: {reason}")
    
    print("\n" + "="*70)
    
    # Save summary to file
    summary_file = os.path.join(vmr_base_folder, "batch_processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Batch Processing Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total models: {len(models)}\n")
        f.write(f"Successful: {len(results['success'])}\n")
        f.write(f"Failed: {len(results['failed'])}\n\n")
        
        f.write("Successful models:\n")
        for model in results["success"]:
            f.write(f"  {model}\n")
        
        f.write("\nFailed models:\n")
        for model, reason in results["failed"]:
            f.write(f"  {model}: {reason}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return len(results["failed"]) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch process VMR models to DualDeviceNav format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all models in vmr_downloads/
  python batch_create_dualdevicenav.py vmr_downloads/

  # Process only models from a list
  python batch_create_dualdevicenav.py vmr_downloads/ --models-list vmr_filtered_models.txt

  # Specify output base folder
  python batch_create_dualdevicenav.py vmr_downloads/ --output processed_models/

  # Stop on first error
  python batch_create_dualdevicenav.py vmr_downloads/ --no-continue

Notes:
  - By default, continues processing even if some models fail
  - Creates dualdevicenav_format/ folder in each model folder
  - Saves processing summary to batch_processing_summary.txt
  - For models that fail automatic centerline extraction, 
    you may need to manually process them in 3D Slicer
        """
    )
    
    parser.add_argument("vmr_folder", help="Base folder containing VMR models")
    parser.add_argument("--models-list", help="Text file with model names to process")
    parser.add_argument("-o", "--output", help="Base output folder")
    parser.add_argument("--no-continue", action="store_true", 
                       help="Stop processing on first error")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.vmr_folder):
        print(f"Error: Folder not found: {args.vmr_folder}")
        sys.exit(1)
    
    success = batch_process(
        args.vmr_folder,
        args.models_list,
        args.output,
        continue_on_error=not args.no_continue
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


