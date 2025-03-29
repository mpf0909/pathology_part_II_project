import numpy as np
import pandas as pd

# Load labels.npy with memory mapping
labels = np.load('/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/labels.npy', mmap_mode='r')

# Initialize a list to store counts
data = []

# Iterate through all patches and count unique nuclei classified as lymphocytes and non-lymphocytes
for patch in labels:
    #instance_layer = patch[16:240, 16:240, 0]  # Instance IDs
    #class_layer = patch[16:240, 16:240, 1]  # Classification labels
    instance_layer = patch[:,:,0]  # Instance IDs
    class_layer = patch[:,:,1]
    
    # Get unique nuclei IDs for lymphocytes and non-lymphocytes
    lymphocyte_ids = set(instance_layer[class_layer == 1])
    non_lymphocyte_ids = set(instance_layer[class_layer == 2])
    
    lymphocyte_count = len(lymphocyte_ids)
    non_lymphocyte_count = len(non_lymphocyte_ids)
    
    data.append([lymphocyte_count, non_lymphocyte_count])

# Create a DataFrame and save to CSV
df = pd.DataFrame(data, columns=['epithelial cells', 'non-epithelial cells'])
df.to_csv('/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/counts.csv', index=False)
    