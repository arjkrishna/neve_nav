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
        logger.info(f"[SKIP DOWNLOAD] {model_name} - zip file exists, extracting...")
    else:
        try:
            logger.info(f"[DOWNLOADING] {model_name}")
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


def download_filtered_models(base_path: str):
    """Download filtered VMR models (Human, Aorta, with Meshes)."""
    
    # Load filtered model names
    with open('vmr_filtered_models.txt', 'r') as f:
        model_list = [line.strip() for line in f if line.strip()]
    
    logger.info("="*60)
    logger.info(f"VMR Model Downloader - Filtered Models")
    logger.info(f"Criteria: Species=Human, Anatomy=Aorta, has Meshes")
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
    logger.info(f"Successfully downloaded: {len(successful)}")
    logger.info(f"Already existed (skipped): {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("")
    
    if successful:
        logger.info(f"Successfully downloaded models:")
        for model in successful:
            logger.info(f"  - {model}")
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
        logger.info("")
    
    logger.info(f"All downloaded models are in: {os.path.join(base_path, 'vmr')}")
    logger.info("="*60)


if __name__ == "__main__":
    # Target path
    target_path = r"D:\vmr"
    
    logger.info("\n" + "="*60)
    logger.info("VMR Filtered Model Downloader")
    logger.info("="*60)
    logger.info(f"This will download 50 Human Aorta models with Meshes to: {target_path}")
    logger.info("")
    
    download_filtered_models(target_path)

