import os
all_directories_prefix_path = '/rds/user/mf774/hpc-work/part_II_project/in-house/inference-data/'
with open('/rds/user/mf774/hpc-work/part_II_project/hovernet/hovernet-conic/inference-results/final-test-inference/directories.txt', "w") as out_file:
    all_diagnosis_categories = os.listdir(all_directories_prefix_path)
    for diagnosis in all_diagnosis_categories:
        all_cases_paths = os.path.join(all_directories_prefix_path, diagnosis)
        all_cases_names = os.listdir(all_cases_paths)
        for case_name in all_cases_names:
            case_path = os.path.join(all_cases_paths, case_name)
            try:
                subdirectory = os.listdir(case_path)[0]
                final_path = os.path.join(case_path, f"'{subdirectory}'")
            except:
                continue
            out_file.write(final_path + "\n")
