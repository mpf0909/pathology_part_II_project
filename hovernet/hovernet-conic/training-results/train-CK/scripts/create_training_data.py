import numpy as np
import os
import re

DIRECTORIES = '/rds/user/mf774/hpc-work/part_II_project/in-house/create-training-data/training-data/'

fold_1_files = ["POST_IHC_PS23-22706_A1_PS23-22706_B1_PS23-24449_A1_cd_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-17071_A1_cd_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-25204_A1_PS23-17242_A1_normal_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-20420_A1_PS23-20442_A1_normal_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-18669_A1_normal_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-14642_A1_ulcer_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-18359_D1_adenoma_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-18359_B1_adenoma_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-24970_A1_PS23-09489_A1_carcinoma_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-15709_A1_PS23-20460_net_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-16539_A_PS23-16539_B1_PS23-10072_A1_eosc_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-21268_A1_PS23-21268_B1_cd_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-25749_A1_PS23-28165_A1_cd_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-17345_A1_PS23-17706_A1_normal_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-17771_A1_PS23-17948_normal_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-18001_A1_normal_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-18359_D2_adenoma_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-18359_A1_adenoma_HE-CK.svs_aligned.ome.tif - Image0",
                "POST_IHC_PS23-19820_A_PS23-20019_A1_PS23-20493_A1_adenoma_HE-CK_aligned.ome.tif - Image0",
                "POST_IHC_PS23-15535_A1_non-spec_HE-CK.svs_aligned.ome.tif - Image0"]


def insert_svs_if_missing(filename):
    if not re.search(r'\.svs_aligned\.ome\.tif', filename):
        # Insert .svs before _aligned.ome.tif
        filename = re.sub(r'(_aligned\.ome\.tif)', r'.svs\1', filename)
    return filename

def make_fold(files, fold_num):
    patch_info_path = f"/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/patch_info.csv"
    os.makedirs(f"fold_{fold_num}", exist_ok=True)
    
    with open(patch_info_path, "w") as f:
        f.write("patch_info\n")  # Header for CSV
    
    patch_names = []
    
    for i, wsi in enumerate(files):
        updated_wsi_name = insert_svs_if_missing(wsi)
        clean_wsi = re.sub(r"_aligned\.ome\.tif - Image0$", "", updated_wsi_name)  # Remove the unwanted part

        he_npy_file = np.load(f'{DIRECTORIES}/{wsi}/he-images.npy')
        mask_npy_file = np.load(f'{DIRECTORIES}/{wsi}/masks.npy')
        
        num_patches = he_npy_file.shape[0]
        patch_names.extend([f"{clean_wsi}:{j:04d}" for j in range(num_patches)])
        
        if i == 0:
            fold_images = he_npy_file
            fold_masks = mask_npy_file
        else:
            fold_images = np.concatenate((fold_images, he_npy_file), axis=0)
            fold_masks = np.concatenate((fold_masks, mask_npy_file), axis=0)
    
    perm = np.random.permutation(fold_masks.shape[0])
    fold_images_shuffled = fold_images[perm]
    fold_masks_shuffled = fold_masks[perm]
    shuffled_patch_names = [patch_names[i] for i in perm]
    
    np.save(f"/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/images.npy", fold_images_shuffled.astype(np.uint8))
    np.save(f"/rds/user/mf774/hpc-work/part_II_project/in-house/train-hovernet/train-CK/data/labels.npy", fold_masks_shuffled.astype(np.uint32))
    
    # Write patch names in shuffled order to CSV
    with open(patch_info_path, "a") as f:
        for patch_name in shuffled_patch_names:
            f.write(f"{patch_name}\n")
    
    print(f"Created shuffled images.npy and labels.npy for fold {fold_num}, with patch info in patch_info.csv")

make_fold(fold_1_files, '1')