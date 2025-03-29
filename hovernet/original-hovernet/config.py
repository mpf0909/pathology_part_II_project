import importlib
import random
import cv2
import numpy as np
from dataset import get_dataset

class Config(object):
    """Configuration file."""

    # Set the dataset here
    DATASET = "pannuke"  # Change to "CPM17" if needed

    def __init__(self):
        self.seed = 10
        self.logging = True
        self.debug = False

        ACCEPTED_DATASETS = ["CPM17", "pannuke"]
        if Config.DATASET not in ACCEPTED_DATASETS:
            raise ValueError(f"{Config.DATASET} is not a valid dataset")

        self.dataset_name = Config.DATASET

        if self.dataset_name == "CPM17":
            self.model_mode = "original"
            self.nr_type = None
            self.type_classification = False
            self.act_shape = [270, 270]
            self.out_shape = [80, 80]
            self.log_dir = f"/rds/user/mf774/hpc-work/part_II_project/hovernet/original-hovernet/cpm17/hover_net/training-results/{self.dataset_name}/"
            self.train_dir_list = ["/rds/user/mf774/hpc-work/part_II_project/opensource/cpm17/patches/train/540x540_164x164/"]
            self.valid_dir_list = ["/rds/user/mf774/hpc-work/part_II_project/opensource/cpm17/patches/valid/540x540_164x164/"]
        elif self.dataset_name == "pannuke":
            self.model_mode = "fast"
            self.nr_type = 6
            self.type_classification = True
            self.act_shape = [256, 256]
            self.out_shape = [164, 164]
            self.log_dir = f"/rds/user/mf774/hpc-work/part_II_project/hovernet/original-hovernet/pannuke/hover_net/training-results/{self.dataset_name}"
            self.train_dir_list = ["/rds/user/mf774/hpc-work/part_II_project/opensource/pannuke/hover_net_format/train/"]
            self.valid_dir_list = ["/rds/user/mf774/hpc-work/part_II_project/opensource/pannuke/hover_net_format/val"]

        # Validation of shapes
        if self.model_mode == "original":
            if self.act_shape != [270, 270] or self.out_shape != [80, 80]:
                raise ValueError("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        elif self.model_mode == "fast":
            if self.act_shape != [256, 256] or self.out_shape != [164, 164]:
                raise ValueError("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.shape_info = {
            "train": {"input_shape": self.act_shape, "mask_shape": self.out_shape},
            "valid": {"input_shape": self.act_shape, "mask_shape": self.out_shape},
        }

        # Dataset initialization
        self.dataset = get_dataset(self.dataset_name)

        # Import model config dynamically
        module = importlib.import_module(f"models.hovernet.opt")
        self.model_config = module.get_config(self.nr_type, self.model_mode)