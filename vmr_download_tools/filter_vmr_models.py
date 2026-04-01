import csv

def filter_vmr_models():
    """Filter VMR models by Species=Human, Anatomy=Aorta, and has Meshes."""
    
    print("Filtering VMR models...")
    print("Criteria:")
    print("  - Species: Human")
    print("  - Anatomy: Aorta")
    print("  - Must have: Meshes")
    print("="*60)
    
    filtered_models = []
    
    with open('dataset-svprojects.csv', 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        
        for row in csv_reader:
            name = row.get('Name', '').strip()
            species = row.get('Species', '').strip()
            anatomy = row.get('Anatomy', '').strip()
            meshes = row.get('Meshes', '').strip()
            
            # Check if meets all criteria
            if (species.lower() == 'human' and 
                anatomy.lower() == 'aorta' and 
                meshes == '1' and 
                name):
                
                # Also capture other useful info
                model_info = {
                    'Name': name,
                    'Age': row.get('Age', ''),
                    'Sex': row.get('Sex', ''),
                    'Disease': row.get('Disease', ''),
                    'Procedure': row.get('Procedure', ''),
                    'Images': row.get('Images', ''),
                    'Paths': row.get('Paths', ''),
                    'Segmentations': row.get('Segmentations', ''),
                    'Models': row.get('Models', ''),
                    'Meshes': row.get('Meshes', ''),
                    'Notes': row.get('Notes', '')
                }
                filtered_models.append(model_info)
    
    # Sort by name
    filtered_models.sort(key=lambda x: x['Name'])
    
    print(f"\nFound {len(filtered_models)} models matching criteria:\n")
    
    # Display results
    for i, model in enumerate(filtered_models, 1):
        print(f"{i:3d}. {model['Name']:<25} | Sex: {model['Sex']:<6} | Age: {model['Age']:<6} | Disease: {model['Disease']}")
    
    # Save model names to file
    model_names = [m['Name'] for m in filtered_models]
    
    with open('vmr_filtered_models.txt', 'w') as f:
        for name in model_names:
            f.write(f"{name}\n")
    
    print(f"\n{'='*60}")
    print(f"Model names saved to: vmr_filtered_models.txt")
    
    # Save detailed info to CSV
    with open('vmr_filtered_models_details.csv', 'w', newline='', encoding='utf-8') as f:
        if filtered_models:
            fieldnames = filtered_models[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_models)
    
    print(f"Detailed info saved to: vmr_filtered_models_details.csv")
    
    # Create Python list
    with open('vmr_filtered_models_list.py', 'w') as f:
        f.write("# VMR models with Species=Human, Anatomy=Aorta, has Meshes\n")
        f.write("# Extracted from vascularmodel.com\n")
        f.write(f"# Total: {len(model_names)} models\n\n")
        f.write("VMR_HUMAN_AORTA_MODELS = [\n")
        for name in model_names:
            f.write(f'    "{name}",\n')
        f.write("]\n")
    
    print(f"Python list saved to: vmr_filtered_models_list.py")
    print("="*60)
    
    return model_names

if __name__ == "__main__":
    models = filter_vmr_models()

