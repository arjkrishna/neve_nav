import csv

def extract_vmr_names():
    """Extract the correct VMR model names from the Name column."""
    
    print("Extracting VMR model names from CSV...")
    print("="*60)
    
    model_names = []
    
    with open('dataset-svprojects.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        
        for row in csv_reader:
            name = row.get('Name', '').strip()
            if name:
                model_names.append(name)
    
    # Sort the names
    model_names.sort()
    
    print(f"Found {len(model_names)} model names:\n")
    
    # Display all names
    for i, name in enumerate(model_names, 1):
        print(f"{i:3d}. {name}")
    
    # Save to file
    with open('vmr_model_names.txt', 'w') as f:
        for name in model_names:
            f.write(f"{name}\n")
    
    print(f"\n{'='*60}")
    print(f"Model names saved to vmr_model_names.txt")
    
    # Also create Python list
    with open('vmr_models_list.py', 'w') as f:
        f.write("# All VMR model names from vascularmodel.com\n")
        f.write("# Format: XXXX_S_ANATOMY_CONDITION\n")
        f.write("VMR_MODELS = [\n")
        for name in model_names:
            f.write(f'    "{name}",\n')
        f.write("]\n")
    
    print("Python list saved to vmr_models_list.py")
    
    return model_names

if __name__ == "__main__":
    models = extract_vmr_names()

