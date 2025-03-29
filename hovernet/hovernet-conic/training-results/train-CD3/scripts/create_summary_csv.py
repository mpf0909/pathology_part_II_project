import joblib
import csv
import pandas as pd
import numpy as np

SPLIT = 'fold_1'
DATA_DIR = '/rds/user/mf774/hpc-work/part_II_project/in-house/create-training-data/final-training-data/CD3/'
PATCH_CSV = '/rds/user/mf774/hpc-work/part_II_project/in-house/create-training-data/final-training-data/CD3/patch_info.csv'
PRED_COUNTS_CSV = f'/rds/user/mf774/hhpc-work/part_II_project/hovernet/hovernet-conic/training-results/train-CD3/validation/{SPLIT}/accurate_valid_pred_cell.csv'
OUTPUT_CSV = '/rds/user/mf774/hpc-work/part_II_project/hovernet/hovernet-conic/training-results/train-CD3/analysis/summary.csv'
splits = joblib.load(f'{DATA_DIR}/{SPLIT}_splits.dat')
pathology_cell_counts = {
    "POST_IHC_PS23-25749_A1_PS23-28165_A1_cd_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-20420_A1_PS23-20442_A1_normal_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-17345_A1_PS23-17706_A1_normal_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-25204_A1_PS23-17242_A1_normal_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-15535_A1_non-spec_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-22706_A1_PS23-22706_B1_PS23-24449_A1_cd_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-18316_A1_PS23-18379_A1_PS23-18656_A1_normal_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-18001_A1_normal_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    "POST_IHC_PS23-18669_A1_normal_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0},
    # "POST_IHC_PS23-17071_A1_cd_HE-CD3.svs": {"lymphocyte": 0, "non-lymphocyte": 0}
}

# get validation indices
valid_indices = splits[0]['valid']

# use validation indices to select the appropriate filenames for each fold
def extract_filenames(csv_path, valid_indices):
    df = pd.read_csv(csv_path, header=0)
    filenames = df.iloc[:, 0].tolist()
    # print(filenames)
    extracted_filenames = [filenames[i] for i in valid_indices if i < len(filenames)]
    return extracted_filenames

extracted_filenames = extract_filenames(PATCH_CSV, valid_indices)

# calculate number of lymphocytes and non-lymphocytes for each wsi
all_wsis = []

with open(PRED_COUNTS_CSV) as fd:
    reader=list(csv.reader(fd))

for i in range(len(extracted_filenames)):
    wsi = extracted_filenames[i].split(":")[0]
    all_wsis.append(wsi)
    for idx, row in enumerate(reader):
        if idx == 0:
            continue
        if idx == i:
            counts = row
            pathology_cell_counts[wsi]["lymphocyte"] += (int(counts[0]))
            pathology_cell_counts[wsi]["non-lymphocyte"] += (int(counts[1]))
            continue

# write cell counts to csv
with open(OUTPUT_CSV, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["wsi", "total_nuclei", "positive_nuclei", "negative_nuclei"])

    for wsi, counts in pathology_cell_counts.items():
        total_nuclei = counts["lymphocyte"] + counts["non-lymphocyte"]
        positive_nuclei = counts["lymphocyte"]
        negative_nuclei = counts["non-lymphocyte"]

        if total_nuclei > 0:
            writer.writerow([wsi, total_nuclei, positive_nuclei, negative_nuclei])

print(np.unique(all_wsis))


