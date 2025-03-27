"""Script aligns whole-slide images and generates overlay plots."""

import subprocess

WSI_DIRECTORY = "/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/all-unaligned-wsis/"
WSI_ALIGNED_DIRECTORY = "/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/aligned-wsis/"

PRE_IHC_FILE_NAME = "PS23-16539_A_PS23-16539_B1_PS23-10072_A1_Eosc_HE-CD3.svs"
POST_IHC_FILE_NAME = "POST_IHC_PS23-16539_A_PS23-16539_B1_PS23-10072_A1eosc_HE-CD3.svs"

WSI_1_PATH = f"{WSI_DIRECTORY}{PRE_IHC_FILE_NAME}"
WSI_2_PATH = f"{WSI_DIRECTORY}{POST_IHC_FILE_NAME}"
SAVE_PATH_1 = f"{WSI_ALIGNED_DIRECTORY}{PRE_IHC_FILE_NAME}_aligned.ome.tif"
SAVE_PATH_2 = f"{WSI_ALIGNED_DIRECTORY}{POST_IHC_FILE_NAME}_aligned.ome.tif"

cmd_1 = [
    "python",
    "align_wsis.py",
    WSI_1_PATH,
    WSI_2_PATH,
    "--save_path_one",
    SAVE_PATH_1,
    "--save_path_two",
    SAVE_PATH_2,
]

cmd_2 = ["python", "plotting.py", SAVE_PATH_1, SAVE_PATH_2]

print("Running command:", " ".join(cmd_1))
subprocess.run(cmd_1, check=False)
print("Running command:", " ".join(cmd_2))
subprocess.run(cmd_2, check=False)
