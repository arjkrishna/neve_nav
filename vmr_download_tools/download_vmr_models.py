import os
import logging
import shutil
import wget
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vmr_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VMR_Download")


def download_vmr_model(model_name: str, base_path: str) -> bool:
    """Download a single VMR model to the specified path."""
    
    path_vmr = os.path.join(base_path, "vmr")
    path_model = os.path.join(path_vmr, model_name)

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(path_vmr):
        os.makedirs(path_vmr)

    model_zip_url = f"https://www.vascularmodel.com/svprojects/{model_name}.zip"
    model_zip_path = os.path.join(path_vmr, f"{model_name}.zip")
    
    # Check if already downloaded and extracted
    if os.path.exists(path_model):
        logger.info(f"[SKIP] {model_name} - already exists")
        return True
    
    # Check if zip exists
    if os.path.exists(model_zip_path):
        logger.info(f"[SKIP DOWNLOAD] {model_name} - zip file exists, will extract")
    else:
        try:
            logger.info(f"[DOWNLOADING] {model_name}")
            logger.info(f"  URL: {model_zip_url}")
            wget.download(model_zip_url, model_zip_path)
            print()  # New line after wget progress
            logger.info(f"  [OK] Downloaded successfully")
        except Exception as e:
            logger.error(f"  [ERROR] Failed to download {model_name}: {str(e)}")
            # Clean up partial download
            if os.path.exists(model_zip_path):
                os.remove(model_zip_path)
            return False
    
    # Extract the zip file
    try:
        logger.info(f"[EXTRACTING] {model_name}")
        shutil.unpack_archive(model_zip_path, path_vmr)
        logger.info(f"  [OK] Extracted to {path_model}")
        
        # Optionally remove zip after successful extraction
        # os.remove(model_zip_path)
        
        return True
    except Exception as e:
        logger.error(f"  [ERROR] Failed to extract {model_name}: {str(e)}")
        return False


def download_all_vmr_models(base_path: str, model_list: list = None):
    """Download all or selected VMR models."""
    
    # If no model list provided, load from file
    if model_list is None:
        with open('vmr_model_names.txt', 'r') as f:
            model_list = [line.strip() for line in f if line.strip()]
    
    logger.info("="*60)
    logger.info(f"VMR Model Downloader")
    logger.info(f"Target directory: {base_path}")
    logger.info(f"Total models to download: {len(model_list)}")
    logger.info("="*60)
    logger.info("")
    logger.info("Downloading vascular models from https://vascularmodel.com/")
    logger.info("Please cite appropriately when using for publications.")
    logger.info("")
    
    successful = []
    failed = []
    skipped = []
    
    for i, model in enumerate(model_list, 1):
        logger.info(f"\n[{i}/{len(model_list)}] Processing: {model}")
        
        result = download_vmr_model(model, base_path)
        
        if result:
            if os.path.exists(os.path.join(base_path, "vmr", model)):
                successful.append(model)
            else:
                skipped.append(model)
        else:
            failed.append(model)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Skipped (already existed): {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("")
    
    if failed:
        logger.info("Failed models:")
        for model in failed:
            logger.info(f"  - {model}")
        
        # Save failed models to file for retry
        with open('vmr_failed_downloads.txt', 'w') as f:
            for model in failed:
                f.write(f"{model}\n")
        logger.info("\nFailed models saved to vmr_failed_downloads.txt")
    
    logger.info(f"\nAll downloaded models are in: {os.path.join(base_path, 'vmr')}")
    logger.info("="*60)


if __name__ == "__main__":
    # Target path
    target_path = r"D:\vmr"
    
    # You can specify specific models here, or leave None to download all
    # Example for stEVE project models:
    steve_models = [
        "0078_H_PULM_H",
        "0079_H_PULM_H", 
        "0094_A_AO_H",
        "0095_A_AO_H",
        "0105_H_PULM_H",
        "0111_H_PULM_H",
        "0131_H_PULM_PAH",
        "0154_H_AO_H",
        "0166_H_PULMFON_SVD",
        "0167_H_PULMFON_SVD",
        "0175_H_CORO_KD",
        "0176_H_CORO_KD",
    ]
    
    print("\nVMR Model Downloader")
    print("="*60)
    print("Options:")
    print("1. Download only stEVE project models (12 models)")
    print("2. Download ALL models (316 models - will take a long time!)")
    print("3. Download from custom list")
    print("="*60)
    
    choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        logger.info("Downloading stEVE project models...")
        download_all_vmr_models(target_path, steve_models)
    elif choice == "2":
        confirm = input("\nAre you sure you want to download ALL 316 models? This will take considerable time and disk space. (yes/no): ").strip().lower()
        if confirm == "yes":
            logger.info("Downloading all VMR models...")
            download_all_vmr_models(target_path, None)
        else:
            logger.info("Download cancelled.")
    elif choice == "3":
        custom_file = input("Enter path to file with model names (one per line): ").strip()
        if os.path.exists(custom_file):
            with open(custom_file, 'r') as f:
                custom_models = [line.strip() for line in f if line.strip()]
            logger.info(f"Downloading {len(custom_models)} models from {custom_file}...")
            download_all_vmr_models(target_path, custom_models)
        else:
            logger.error(f"File not found: {custom_file}")
    else:
        logger.error("Invalid choice. Exiting.")

