import pandas as pd
import json
import glob

def get_area_from_metadata(file_path, i):
    metadata_path = f'{file_path}/{i}/metadata.json'
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    area = metadata["tissue_area_og_mask"]
    return area





# Load metadata.json to get areas
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# Get all CSV files
csv_files = glob.glob("*.csv")

# Dictionary to store results
results = {}

for file in csv_files:
    # Read CSV file
    df = pd.read_csv(file)
    
    # Get the corresponding area from metadata
    area = metadata.get(file, None)
    if area is None:
        print(f"Warning: No area found for {file}, skipping.")
        continue
    
    # Compute the average number of nuclei per unit area for each class
    avg_nuclei_per_area = df.mean() / area
    
    # Store the results
    results[file] = avg_nuclei_per_area.to_dict()

# Print results
for file, values in results.items():
    print(f"Results for {file}:")
    for cell_type, value in values.items():
        print(f"  {cell_type}: {value:.4f}")