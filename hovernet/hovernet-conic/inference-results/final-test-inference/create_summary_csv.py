import joblib
import csv
import os
import pandas as pd
import numpy as np

MODEL = 'CK'
ROOT_PATH = '/rds/user/mf774/rds-part-ii-project-3PvMefeRXQQ/part_II_project/in-house/inference-data/'
OUTPUT_CSV = f'/rds/user/mf774/hpc-work/part_II_project/hovernet/hovernet-conic/inference-results/final-test-inference/summary_{MODEL}.csv'

pathology_cell_counts = {}

all_diagnoses = os.listdir(ROOT_PATH)
for diagnosis in all_diagnoses:
    all_cases_per_diagnosis = os.listdir(os.path.join(ROOT_PATH, diagnosis))
    for case in all_cases_per_diagnosis:
        if case == "PS23-19293":
            # skip this case since inference did not finish running in 12 hours permitted by university HPC
            continue
        if MODEL == 'lizard':
            pathology_cell_counts[case] = {"neutrophil": 0, "epithelial":0, "lymphocyte":0, "plasma":0, "eosinophil":0, "connective":0}
        else:
            pathology_cell_counts[case] = {"positive cell": 0,"negative cell": 0}
        case_path = os.path.join(ROOT_PATH, diagnosis, case)
        try:
            subdirectory = os.listdir(case_path)[0]
            path_to_model_results_for_case = os.path.join(case_path, subdirectory)
        except:
            continue
        path_to_cell_counts = os.path.join(path_to_model_results_for_case, MODEL, 'valid_pred_cell.csv')
        print(path_to_cell_counts)
        with open(path_to_cell_counts) as fd:
            reader = list(csv.reader(fd))
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                counts = row
                if MODEL == 'lizard':
                    pathology_cell_counts[case]["neutrophil"] += (int(counts[0]))
                    pathology_cell_counts[case]["epithelial"] += (int(counts[1]))
                    pathology_cell_counts[case]["lymphocyte"] += (int(counts[2]))
                    pathology_cell_counts[case]["plasma"] += (int(counts[3]))
                    pathology_cell_counts[case]["eosinophil"] += (int(counts[4]))
                    pathology_cell_counts[case]["connective"] += (int(counts[5]))
                else:
                    pathology_cell_counts[case]["positive cell"] += (int(counts[0]))
                    pathology_cell_counts[case]["negative cell"] += (int(counts[1]))

# write cell counts to csv
with open(OUTPUT_CSV, mode="a", newline="") as file:
    writer = csv.writer(file)
    if MODEL == 'lizard':
        writer.writerow(["wsi", "total_cells", "neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"])
    else:
        writer.writerow(["wsi", "total_cells", "positive_cells", "negative_cells"])
    for wsi, counts in pathology_cell_counts.items():
        if MODEL == 'lizard':
            total_cells = counts["neutrophil"] + counts["epithelial"] + counts["lymphocyte"] + counts["plasma"] + counts["eosinophil"] + counts["connective"]
            neutrophil = counts["neutrophil"]
            epithelial = counts["epithelial"]
            lymphocyte = counts["lymphocyte"]
            plasma = counts["plasma"]
            eosinophil = counts["eosinophil"]
            connective = counts["connective"]
            if total_cells > 0:
                writer.writerow([wsi, total_cells, neutrophil, epithelial, lymphocyte, plasma, eosinophil, connective])
        else:
            total_cells = counts["positive cell"] + counts["negative cell"]
            positive_nuclei = counts["positive cell"]
            negative_nuclei = counts["negative cell"]
            if total_cells > 0:
                writer.writerow([wsi, total_cells, positive_nuclei, negative_nuclei])