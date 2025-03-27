import os
import numpy as np
import pandas as pd
import json
import cv2
from skimage.draw import polygon
from tqdm import tqdm

AREA_THRESHOLD = 2000
DOWNSAMPLE = 0.5
MIN_NUCLEI = 3
STRIDE = 256
GEOJSON_DIR = "qupath-geojsons"
PATCHES_DIR = "qupath-patches"

# for geojsons in which bounding box was accidentally deleted during manual quality control. Bounding box is therefore
# the last item in the geojson, not the first
EXCEPTIONS = ['geojsons/POST_IHC_PS23-18001_A1_normal_HE-CK.svs_aligned.ome.tif - Image0.geojson',
              'geojsons/POST_IHC_PS23-18001_A1_normal_HE-CD3.svs_aligned.ome.tif - Image0.geojson',
              'geojsons/POST_IHC_PS23-18359_A1_adenoma_HE-CK.svs_aligned.ome.tif - Image0.geojson',
              'geojsons/POST_IHC_PS23-21433_A1_PS23-21433_B1_PS23-22604_A1_cd_HE-CD3.svs_aligned.ome.tif - Image0.geojson',
              'geojsons/POST_IHC_PS23-22706_A1_PS23-22706_B1_PS23-24449_A1_cd_HE-CK.svs_aligned.ome.tif - Image0.geojson',
]

# adapted from https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def extract_mask_from_geojson(file_path):

    """
    Converts geojson file annotations into a mask of dimensions (N, M, 2) where N = image height, M = image width,
    channel 1 = nuclei instance id and channel 2 = nuclei classification
    """

    with open(file_path) as f:
        data = json.load(f)

    # TODO extract dimensions of bounding box (once find a way to draw bounding box)
    # for now these correspond to the maximum possible dimensions of the slide .tif file
    if file_path.strip() in [e.strip() for e in EXCEPTIONS]:
        wsi_bounding_box = data[-1]["geometry"]["coordinates"][0]
        img_width = int(max(coordinate[0] for coordinate in wsi_bounding_box))
        img_height = int(max(coordinate[1] for coordinate in wsi_bounding_box))
        data.pop(-1)
    else:
        wsi_bounding_box = data[0]["geometry"]["coordinates"][0]
        # remove first entry since in json file since refers to the bounding box not cell annotations
        img_width = int(max(coordinate[0] for coordinate in wsi_bounding_box))
        img_height = int(max(coordinate[1] for coordinate in wsi_bounding_box))
        data.pop(0)


    # Initialize an empty mask: first channel for instance IDs, second for class labels
    mask = np.zeros((img_width+1, img_height+1, 2), dtype=np.int32)

    # Define a mapping from annotation class names to numeric labels
    class_mapping = {
        "Positive": 1,
        "Negative": 2,
        # add other classes if needed
    }

    # TODO: check whether HoverNet expects 0 indexing
    instance_id = 1  # Start instance numbering at 1

    # Loop through each annotation feature
    for feature in tqdm(data, desc="Processing annotations", unit="annotations"):
        # note the first feature is the bounding box
        geometry = feature['geometry']
        properties = feature['properties']
        
        # Get the class type from properties (adjust the key as needed)
        classification_dict = properties["classification"]
        class_type = classification_dict["name"]
        class_val = class_mapping.get(class_type, 0)  # Default to 0 if not found

        # Process only if the geometry is a polygon
        if geometry['type'] == 'Polygon':
            # GeoJSON polygons: coordinates[0] is the outer ring
            coords = geometry['coordinates'][0]

            # Separate x and y coordinates.
            # Note: GeoJSON typically gives [x, y]. In numpy arrays, y is the row and x is the column.
            xs = [int(pt[0]) for pt in coords]
            ys = [int(pt[1]) for pt in coords]

            # check if polygon is below threshold size for nucleus
            area = PolyArea(xs, ys)
            if area <= AREA_THRESHOLD:
                # Use skimage.draw.polygon to get pixel indices inside the polygon.
                rr, cc = polygon(c=ys, r=xs)#,shape=(img_height, img_width))
                # Fill in the mask:
                # - Channel 0 gets the unique instance id.
                # - Channel 1 gets the class label.
                mask[rr, cc, 0] = instance_id
                mask[rr, cc, 1] = class_val
            # Increment instance id for the next annotation.
                instance_id += 1
            else:
                continue

    # Save the resulting mask as a .npy file
    return mask


def crop_and_downsample_mask(mask, stride, downsample_factor):

    """
    Remove bottom and right padding and downsample to make image dimensions directly divisible by stride and mask at right
    magnification
    """

    # downsample first
    downsampled_mask = cv2.resize(mask, (int(mask.shape[1]*downsample_factor), int(mask.shape[0]*downsample_factor)),interpolation=cv2.INTER_NEAREST)
    downsampled_mask = np.array(downsampled_mask).astype('uint32')

    # ensure divisible by stride/downsample_factor

    x, y, _ = downsampled_mask.shape
    new_x = int((x//(stride/downsample_factor)) * stride/downsample_factor)
    new_y = int((y//(stride/downsample_factor)) * stride/downsample_factor)
    return downsampled_mask[0:new_x, 0:new_y, :]


def extract_patches_from_mask(mask, stride):
    
    """
    Convert mask of size (N, M, 2) into mask of size (Q, 256, 256, 2)
    """

    print("Loading large .npy file...")
    width, height, depth = mask.shape
    num_patches_x = width // stride
    num_patches_y = height // stride

    # Extract patches without padding
    patches = []
    for i in tqdm(range(num_patches_x), desc="Converting patches into (N, 256, 256, 2) format", leave=False):
        for j in range(num_patches_y):
            patch = mask[i*stride:(i+1)*stride, j*stride:(j+1)*stride, :]
            patches.append(patch)

    # Convert list to numpy array
    patches = np.array(patches)
    return patches, stride, num_patches_x, num_patches_y


def filter_patches(output_dir, filename, output_csv, patches, num_patches_x, num_patches_y, stride, min_nuclei, downsample_factor):
    
    """
    Remove patches with fewer than min_nuclei.
    Takes numpy array of dimension (N, 256, 256, 2) as input.
    Saves patches with coordinates in the filename.
    """

    os.makedirs(output_dir, exist_ok=True)

    total_patches = num_patches_x * num_patches_y
    total_positive_nuclei = 0
    total_negative_nuclei = 0

    print(f"Processing {total_patches} patches...")

    patch_index = 0
    saved_count = 0   
    seen_instances = set()
    with tqdm(total=total_patches, desc="Filtering patches", unit="patch") as pbar:
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                patch = patches[patch_index]
                instance_layer = patch[:, :, 0]
                class_layer = patch[:, :, 1]
                
                unique_instance_ids = np.unique(instance_layer)
                unique_instance_ids = unique_instance_ids[unique_instance_ids > 0]
                nuclei_counts = len(unique_instance_ids)
                
                if nuclei_counts >= min_nuclei:
                    x = i * stride
                    y = j * stride
                    np.save(os.path.join(output_dir, f"{filename}[x={int(x//downsample_factor)},y={int(y//downsample_factor)}].npy"), np.transpose(patch, (1, 0, 2)))
                    saved_count += 1
                    
                    # Identify unique instances for positive and negative nuclei
                    unique_positive_nuclei = np.unique(instance_layer[class_layer == 1])
                    unique_negative_nuclei = np.unique(instance_layer[class_layer == 2])
                    
                    for nucleus in unique_positive_nuclei:
                        if nucleus > 0 and nucleus not in seen_instances:
                            total_positive_nuclei += 1
                            seen_instances.add(nucleus)
                    
                    for nucleus in unique_negative_nuclei:
                        if nucleus > 0 and nucleus not in seen_instances:
                            total_negative_nuclei += 1
                            seen_instances.add(nucleus)
                
                patch_index += 1
                pbar.update(1)

    results_df = pd.DataFrame(
        {
            "filename": [filename],
            "number of patches": [saved_count],
            "positive nuclei": [total_positive_nuclei],
            "negative nuclei": [total_negative_nuclei],
            "total nuclei": [total_positive_nuclei+total_negative_nuclei],
        }
    )

    try:
        results_df.to_csv(output_csv, mode="a", header=not pd.io.common.file_exists(output_csv), index=False)
    except AttributeError:
        results_df.to_csv(output_csv, mode="w", header=True, index=False)

    print("Results saved to", output_csv)
    return


if __name__ == "__main__":
    os.makedirs(PATCHES_DIR, exist_ok=True)
    output_csv = os.path.join(PATCHES_DIR, "summary.csv")

    for filename in os.listdir(GEOJSON_DIR):
        if filename.lower().endswith(".geojson"):
            geojson_path = os.path.join(GEOJSON_DIR, filename)
            output_subdir = os.path.join(PATCHES_DIR, os.path.splitext(filename)[0])

            print(f"Processing {filename}...")
            mask = extract_mask_from_geojson(geojson_path)
            mask = crop_and_downsample_mask(mask, STRIDE, DOWNSAMPLE)
            patches, stride, num_patches_x, num_patches_y = extract_patches_from_mask(mask, STRIDE)
            filter_patches(output_subdir, filename, output_csv, patches, num_patches_x, num_patches_y, stride, MIN_NUCLEI, DOWNSAMPLE)

    print("Processing complete!")