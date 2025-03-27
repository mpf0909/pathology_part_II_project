import os
import pandas as pd

input_dir = '/rds/user/mf774/hpc-work/part_II_project/in-house/create-masks/segmentation-annotation-data/universal-stardist/universal-stardist-txts/'
output_dir = '/rds/user/mf774/hpc-work/part_II_project/in-house/create-masks/segmentation-annotation-data/universal-stardist/universal-stardist-csvs/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all .txt files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".txt", ".csv"))

        # Read the tab-separated file and convert it to CSV
        df = pd.read_csv(input_path, delimiter="\t")
        df.to_csv(output_path, index=False)

        print(f"Converted {filename} to {output_path}")

print("All .txt files have been processed.")
