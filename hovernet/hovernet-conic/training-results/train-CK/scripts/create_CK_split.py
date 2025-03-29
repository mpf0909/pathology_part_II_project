import numpy as np
import joblib
import pandas as pd

SEED = 5
info = pd.read_csv('/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/patch_info.csv')
file_names = np.squeeze(info.to_numpy()).tolist()
file_names = [v.split(':')[0] for v in file_names]
print(len(file_names))

def find_indices(lst, target_set):
    return [i for i, val in enumerate(lst) if val in target_set]

# Manually assigned files for each fold
fold1_files = set([
    "POST_IHC_PS23-22706_A1_PS23-22706_B1_PS23-24449_A1_cd_HE-CK.svs",
    "POST_IHC_PS23-17071_A1_cd_HE-CK.svs",
    "POST_IHC_PS23-25204_A1_PS23-17242_A1_normal_HE-CK.svs",
    "POST_IHC_PS23-20420_A1_PS23-20442_A1_normal_HE-CK.svs",
    "POST_IHC_PS23-18669_A1_normal_HE-CK.svs",
    "POST_IHC_PS23-14642_A1_ulcer_HE-CK.svs",
    "POST_IHC_PS23-18359_D1_adenoma_HE-CK.svs",
    "POST_IHC_PS23-18359_B1_adenoma_HE-CK.svs",
    "POST_IHC_PS23-24970_A1_PS23-09489_A1_carcinoma_HE-CK.svs",
    "POST_IHC_PS23-15709_A1_PS23-20460_net_HE-CK.svs",
    "POST_IHC_PS23-16539_A_PS23-16539_B1_PS23-10072_A1_eosc_HE-CK.svs",
])

fold2_files = set([
    "POST_IHC_PS23-21268_A1_PS23-21268_B1_cd_HE-CK.svs",
    "POST_IHC_PS23-25749_A1_PS23-28165_A1_cd_HE-CK.svs",
    "POST_IHC_PS23-17345_A1_PS23-17706_A1_normal_HE-CK.svs",
    "POST_IHC_PS23-17771_A1_PS23-17948_normal_HE-CK.svs",
    "POST_IHC_PS23-18001_A1_normal_HE-CK.svs",
    "POST_IHC_PS23-18359_D2_adenoma_HE-CK.svs",
    "POST_IHC_PS23-18359_A1_adenoma_HE-CK.svs",
    "POST_IHC_PS23-19820_A_PS23-20019_A1_PS23-20493_A1_adenoma_HE-CK.svs",
    "POST_IHC_PS23-15535_A1_non-spec_HE-CK.svs"
])

assert fold1_files.isdisjoint(fold2_files)

train_indices = find_indices(file_names, fold1_files)
valid_indices = find_indices(file_names, fold2_files)

fold_1_splits = [{
    'train': train_indices,
    'valid': valid_indices
}]

fold_2_splits = [{
    'train': valid_indices,
    'valid': train_indices
}]

joblib.dump(fold_1_splits, '/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/fold_1_splits.dat')
joblib.dump(fold_2_splits, '/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/fold_2_splits.dat')