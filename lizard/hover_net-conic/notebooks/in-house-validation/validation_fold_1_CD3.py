import sys
import logging
import os

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tiatoolbox
from IPython.utils import io as IPyIO
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 300

# adding the project root folder
sys.path.append('../')
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours

from misc.utils import cropping_center, recur_find_ext, rmdir
from run_utils.utils import convert_pytorch_checkpoint

SPLIT = 'fold_1'
# Random seed for deterministic
SEED = 5
# The number of nuclei within the dataset/predictions.
# For in-house CD3 dataset, we have 2 (+1 for background) types in total.
NUM_TYPES = 3
# The path to the directory containg images.npy etc.
DATA_DIR = '/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/data/'
# The path to the pretrained weights
CHECKPOINT_PATH = f'/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/models/baseline/{SPLIT}/model/01/net_epoch=50.tar'
# The path to contain output and intermediate processing results
OUT_DIR = '/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/validation/'
# the fold to use

from net_desc import HoVerNetConic

checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))['desc']
checkpoint = convert_pytorch_checkpoint(checkpoint)  # to turn from Multiple GPU Mode to Single GPU Mode
model = HoVerNetConic(num_types=NUM_TYPES)
model.load_state_dict(checkpoint)

imgs = np.load(f'{DATA_DIR}/images.npy')
labels = np.load(f'{DATA_DIR}/labels.npy')

splits = joblib.load(f'{DATA_DIR}/{SPLIT}_splits.dat')
valid_indices = splits[0]['valid']
# print(valid_indices)

img_dir = f'{OUT_DIR}/{SPLIT}/imgs'
os.makedirs(img_dir, exist_ok=True)
for idx in valid_indices:
    img = imgs[idx]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{img_dir}/{idx:04d}.png', img)

valid_labels = labels[valid_indices]
np.save(f'{OUT_DIR}/{SPLIT}/valid_true.npy', valid_labels)

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

infer_img_paths = recur_find_ext(f'{OUT_DIR}/{SPLIT}/imgs/', ext=['.png'])

# capture all the printing to avoid cluttering the console
output_file = predictor.predict(
    infer_img_paths,
    masks=None,
    mode='tile',
    #on_gpu=True,
    ioconfig=ioconfig,
    crash_on_exception=True,
    save_dir=f'{OUT_DIR}/{SPLIT}/raw/'
)