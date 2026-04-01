import requests
import csv
import re

def download_vmr_dataset():
    """Download and parse the VMR dataset CSV file."""
    
    csv_url = "https://www.vascularmodel.com/dataset/dataset-svprojects.csv"
    
    print(f"Downloading: {csv_url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(csv_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save the CSV file
        with open('dataset-svprojects.csv', 'w', encoding='utf-8', newline='') as f:
            f.write(response.text)
        print("[OK] CSV file downloaded successfully")
        
        # Parse the CSV
        lines = response.text.split('\n')
        csv_reader = csv.DictReader(lines)
        
        model_names = set()
        model_pattern = re.compile(r'\d{4}_\d{4}')
        
        all_rows = []
        for row in csv_reader:
            all_rows.append(row)
            # Check all fields for model IDs
            for key, value in row.items():
                if value:
                    matches = model_pattern.findall(value)
                    model_names.update(matches)
        
        print(f"\nTotal rows in CSV: {len(all_rows)}")
        if all_rows:
            print(f"CSV columns: {list(all_rows[0].keys())}")
        
        # Convert to sorted list
        model_list = sorted(list(model_names))
        
        print(f"\n{'='*60}")
        print(f"Found {len(model_list)} unique model IDs:")
        print(f"{'='*60}")
        
        for model in model_list:
            print(f"  {model}")
        
        # Save model names
        with open('vmr_all_models.txt', 'w') as f:
            for model in model_list:
                f.write(f"{model}\n")
        print(f"\nModel names saved to vmr_all_models.txt")
        
        # Also create a formatted Python list
        with open('vmr_models_list.py', 'w') as f:
            f.write("# All VMR model names from vascularmodel.com\n")
            f.write("VMR_MODELS = [\n")
            for model in model_list:
                f.write(f'    "{model}",\n')
            f.write("]\n")
        print("Python list saved to vmr_models_list.py")
        
        return model_list
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print("VMR Dataset CSV Downloader")
    print("="*60)
    models = download_vmr_dataset()

