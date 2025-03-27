import glob
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def list_all_imgs(*dirs):
    """Return sorted lists of image files from multiple directories."""
    return [sorted(glob.glob(os.path.join(d, "*.png"))) for d in dirs[:-1]] + [sorted(glob.glob(os.path.join(dirs[-1], "*.npy")))]

def extract_coordinates(filename):
    """Extract x and y coordinates from a filename."""
    match = re.search(r'x=(\d+),y=(\d+)', filename)
    return (int(match.group(1)), int(match.group(2))) if match else None

def load_image(image_path):
    """Load an image and convert to a numpy array."""
    return np.array(Image.open(image_path))

def load_npy(npy_path):
    """Load a .npy file."""
    return np.load(npy_path)

def count_nuclei(mask):
    """Count the number of nuclei in a mask and classify them."""
    instance_layer = mask[:, :, 0]
    class_layer = mask[:, :, 1]
    
    unique_instances = np.unique(instance_layer)
    unique_instances = unique_instances[unique_instances > 0]  # Ignore background (0)
    
    unique_positive = np.unique(instance_layer[class_layer == 1])
    unique_negative = np.unique(instance_layer[class_layer == 2])
    
    return len(unique_instances), len(unique_positive), len(unique_negative)

def match_and_save(he_dir, mask_dir, ihc_dir, output_dir):
    print(f"Processing: {he_dir}, {mask_dir}, {ihc_dir} -> {output_dir}")

    he_imgs, ihc_imgs, mask_npys = list_all_imgs(he_dir, ihc_dir, mask_dir)

    # Build dictionaries of files with coordinates as keys
    def build_dict(files):
        return {coords: f for f in files if (coords := extract_coordinates(f))}

    he_dict = build_dict(he_imgs)
    ihc_dict = build_dict(ihc_imgs)
    mask_dict = build_dict(mask_npys)

    matched_coords = set(he_dict.keys()) & set(ihc_dict.keys()) & set(mask_dict.keys())

    if not matched_coords:
        print("No matched samples found.")
        return

    image_list, ihc_list, mask_list = [], [], []
    total_nuclei_wsi = 0
    positive_nuclei_wsi = 0
    negative_nuclei_wsi = 0

    # Load images and masks in parallel
    with ThreadPoolExecutor() as executor:
        he_futures = {coords: executor.submit(load_image, he_dict[coords]) for coords in matched_coords}
        ihc_futures = {coords: executor.submit(load_image, ihc_dict[coords]) for coords in matched_coords}
        mask_futures = {coords: executor.submit(load_npy, mask_dict[coords]) for coords in matched_coords}

        for coords in tqdm(matched_coords, desc="Processing images", unit="file"):
            he_img = he_futures[coords].result()
            ihc_img = ihc_futures[coords].result()
            mask = mask_futures[coords].result()

            image_list.append(he_img)
            ihc_list.append(ihc_img)
            mask_list.append(mask)

            # Count nuclei
            # Count nuclei **for the patch**
            total_nuclei, positive_nuclei, negative_nuclei = count_nuclei(mask)

            # **Accumulate counts for the WSI**
            total_nuclei_wsi += total_nuclei
            positive_nuclei_wsi += positive_nuclei
            negative_nuclei_wsi += negative_nuclei

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'he-images.npy'), np.stack(image_list, axis=0))
    np.save(os.path.join(output_dir, 'ihc-images.npy'), np.stack(ihc_list, axis=0))
    np.save(os.path.join(output_dir, 'masks.npy'), np.stack(mask_list, axis=0))

    # Save nuclei counts
    output_csv = "/rds/user/mf774/hpc-work/part_II_project/in-house/create-training-data/analysis/summary.csv"
    wsi_filename = os.path.basename(he_dir)  # Use the WSI folder name as identifier
    df = pd.DataFrame([[wsi_filename, total_nuclei_wsi, positive_nuclei_wsi, negative_nuclei_wsi]], 
                      columns=["wsi", "total_nuclei", "positive_nuclei", "negative_nuclei"])
    df.to_csv("summary.csv", mode="a", header=not os.path.exists(output_csv), index=False)

    print(f"Saved {len(image_list)} matched samples.")

if __name__ == "__main__":
    base_dirs = {
        "he": "/rds/user/mf774/hpc-work/part_II_project/in-house/create-he-ihc-patches/he-ihc-patches/pre-ihc/patches_256_256/",
        "ihc": "/rds/user/mf774/hpc-work/part_II_project/in-house/create-he-ihc-patches/he-ihc-patches/post-ihc/patches_256_256/",
        "mask": "/rds/user/mf774/rds/hpc-work/part_II_project/in-house/create-masks/mask-patches/universal-stardist-patches/"
    }

    subdirectories = {key: sorted(os.listdir(base_dirs[key])) for key in base_dirs}

    for subdir in zip(subdirectories["he"], subdirectories["ihc"], subdirectories["mask"]):
        he_subdir, ihc_subdir, mask_subdir = (os.path.join(base_dirs[key], subdir[i]) for i, key in enumerate(base_dirs))
        output_dir = f"universal-stardist/{subdir[2]}"
        match_and_save(he_subdir, mask_subdir, ihc_subdir, output_dir)