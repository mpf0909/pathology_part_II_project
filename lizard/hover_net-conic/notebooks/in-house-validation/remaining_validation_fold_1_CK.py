import sys
import logging
import os
import cv2
import joblib
import torch
import tiatoolbox
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.utils import io as IPyIO
from tqdm import tqdm
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours
from misc.utils import cropping_center, recur_find_ext, rmdir
from run_utils.utils import convert_pytorch_checkpoint
from net_desc import HoVerNetConic

mpl.rcParams['figure.dpi'] = 300
sys.path.append('../')


SPLIT = 'fold_1'
# Random seed for deterministic
SEED = 5
# The number of nuclei within the dataset/predictions.
# For in-house CD3 dataset, we have 2 (+1 for background) types in total.
NUM_TYPES = 3
# The path to the directory containg images.npy etc.
DATA_DIR = '/rds/user/mf774/hpc-work/part_II_project/in-house/training-CK/data/'
# The path to the pretrained weights
CHECKPOINT_PATH = f'/rds/user/mf774/hpc-work/part_II_project/in-house/training-CK/models/baseline/{SPLIT}/model/01/net_epoch=50.tar'
# The path to contain output and intermediate processing results
OUT_DIR = '/rds/user/mf774/hpc-work/part_II_project/in-house/training-CK/validation/'

checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))['desc']
checkpoint = convert_pytorch_checkpoint(checkpoint)  # to turn from Multiple GPU Mode to Single GPU Mode
model = HoVerNetConic(num_types=NUM_TYPES)
model.load_state_dict(checkpoint)

# Tile prediction
predictor = SemanticSegmentor(
    model=model,
    num_loader_workers=2,
    batch_size=64,
)

# Define the input/output configurations
ioconfig = IOSegmentorConfig(
    input_resolutions=[
        {'units': 'baseline', 'resolution': 1.0},
    ],
    output_resolutions=[
        {'units': 'baseline', 'resolution': 1.0},
        {'units': 'baseline', 'resolution': 1.0},
        {'units': 'baseline', 'resolution': 1.0},
    ],
    save_resolution={'units': 'baseline', 'resolution': 1.0},
    patch_input_shape=[256, 256],
    patch_output_shape=[256, 256],
    stride_shape=[256, 256],
)

logger = logging.getLogger()
logger.disabled = True

file_map = joblib.load(f'{OUT_DIR}/{SPLIT}/raw/file_map.dat')
# all_images = set(os.listdir(f'{OUT_DIR}/{SPLIT}/imgs/'))

all_images = {os.path.join(f'{OUT_DIR}/{SPLIT}/imgs/', img) for img in os.listdir(f'{OUT_DIR}/{SPLIT}/imgs/')}
previously_inferred_images = {os.path.basename(entry[0]) for entry in file_map}
missing_images = list(all_images - previously_inferred_images)
missing_images.sort()

# capture all the printing to avoid cluttering the console
output_file = predictor.predict(
    missing_images,
    masks=None,
    mode='tile',
    #on_gpu=True,
    ioconfig=ioconfig,
    crash_on_exception=True,
    save_dir=f'{OUT_DIR}/{SPLIT}/remaining_raw/'
)