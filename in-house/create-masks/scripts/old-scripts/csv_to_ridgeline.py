"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Set theme
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

print("Loading CSV files...")
# Load all 21 CSVs
file_list = glob.glob("CK_distributions/*.csv")  # Change this to your actual CSV path
df_list = []

for file in file_list:
    df = pd.read_csv(file)
    df["Image Name"] = file.split("/")[-1]  # Assign the file name as the 'Image' identifier
    df_list.append(df)

df_combined = pd.concat(df_list, ignore_index=True)

print(f"Loaded {len(file_list)} CSV files. Total data points: {df_combined.shape[0]}")

# Check if required column exists
if "Cytoplasm: DAB_vec OD mean" not in df_combined.columns:
    raise ValueError("Column 'Cytoplasm: DAB_vec OD mean' not found in CSV files.")

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(len(df_combined["Image Name"].unique()), rot=-.25, light=.7)

print("Setting up the grid...")
g = sns.FacetGrid(df_combined, row="Image Name", hue="Image Name", aspect=15, height=1.2, palette=pal)

# Draw the KDE plots
print("Plotting KDEs...")
g.map(sns.kdeplot, "Cytoplasm: DAB_vec OD mean",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

g.map(sns.kdeplot, "Cytoplasm: DAB_vec OD mean", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.set(xlim=(-0.2, 2))  # Set x-axis from -0.2 to 2

# Reference line at y=0
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define a function to label each row with the image name
def label(x, color, label):
    ax = plt.gca()
    ax.text(-0.2, .65, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes, fontsize=8)

g.map(label, "Cytoplasm: DAB_vec OD mean")

# Adjust subplot overlap
g.figure.subplots_adjust(hspace=-0.25)

# Remove axes details
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

print("Displaying plot...")
plt.savefig("test.png")
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Set theme
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Dictionary mapping file names to labels
file_names = {
    "POST_IHC_PS23-14642_A1_ulcer_HE-CK_aligned.ome.csv": "ulcer",
    "POST_IHC_PS23-15535_A1_non-spec_HE-CK.svs_aligned.ome.csv": "non-spec",
    "POST_IHC_PS23-15709_A1_PS23-20460_net_HE-CK.svs_aligned.ome.csv": "net",
    "POST_IHC_PS23-16539_A_PS23-16539_B1_PS23-10072_A1_eosc_HE-CK.svs_aligned.ome.csv": "eosc",
    "POST_IHC_PS23-17071_A1_cd_HE-CK.svs_aligned.ome.csv": "coeliac",
    "POST_IHC_PS23-17771_A1_PS23-17948_normal_HE-CK.svs_aligned.ome.csv": "normal",
    "POST_IHC_PS23-18001_A1_normal_HE-CK.svs_aligned.ome.csv": "normal",
    "POST_IHC_PS23-18316_A1_PS23-18379_A1_PS23-18656_A1_normal_HE-CK.svs_aligned.ome.csv": "normal",
    "POST_IHC_PS23-18359_A1_adenoma_HE-CK.svs_aligned.ome.csv": "adenoma",
    "POST_IHC_PS23-18359_B1_adenoma_HE-CK.svs_aligned.ome.csv": "adenoma",
    "POST_IHC_PS23-18359_D1_adenoma_HE-CK.svs_aligned.ome.csv": "adenoma",
    "POST_IHC_PS23-18359_D2_adenoma_HE-CK.svs_aligned.ome.csv": "adenoma",
    "POST_IHC_PS23-18669_A1_normal_HE-CK.svs_aligned.ome.csv": "normal",
    "POST_IHC_PS23-19820_A_PS23-20019_A1_PS23-20493_A1_adenoma_HE-CK.svs_aligned.ome.csv": "adenoma",
    "POST_IHC_PS23-20420_A1_PS23-20442_A1_normal_HE-CK.svs_aligned.ome.csv": "normal",
    "POST_IHC_PS23-21268_A1_PS23-21268_B1_cd_HE-CK.svs_aligned.ome.csv": "coeliac",
    "POST_IHC_PS23-21433_A1_PS23-21433_B1_PS23-22604_A1_cd_HE-CK.svs_aligned.ome.csv": "coeliac",
    "POST_IHC_PS23-22706_A1_PS23-22706_B1_PS23-24449_A1_cd_HE-CK.svs_aligned.ome.csv": "coeliac",
    "POST_IHC_PS23-24970_A1_PS23-09489_A1_carcinoma_HE-CK.svs_aligned.ome.csv": "carcinoma",
    "POST_IHC_PS23-25204_A1_PS23-17242_A1_normal_HE-CK.svs_aligned.ome.csv": "normal",
    "POST_IHC_PS23-25749_A1_PS23-28165_A1_cd_HE-CK.svs_aligned.ome.csv": "coeliac"
}

print("Loading CSV files...")
file_list = glob.glob("CK_distributions/*.csv")
df_list = []

for file in file_list:
    df = pd.read_csv(file)
    filename = file.split("/")[-1]  # Extract the filename
    df["Image Name"] = file_names.get(filename, filename)  # Replace with label if available
    df_list.append(df)

# Combine data
df_combined = pd.concat(df_list, ignore_index=True)
print(f"Loaded {len(file_list)} CSV files. Total data points: {df_combined.shape[0]}")

# Check if required column exists
if "Cytoplasm: DAB_vec OD mean" not in df_combined.columns:
    raise ValueError("Column 'Cytoplasm: DAB_vec OD mean' not found in CSV files.")

# Initialize FacetGrid
pal = sns.cubehelix_palette(len(df_combined["Image Name"].unique()), rot=-.25, light=.7)
print("Setting up the grid...")
g = sns.FacetGrid(df_combined, row="Image Name", hue="Image Name", aspect=15, height=1.2, palette=pal)

# Draw KDE plots
g.map(sns.kdeplot, "Cytoplasm: DAB_vec OD mean", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "Cytoplasm: DAB_vec OD mean", clip_on=False, color="w", lw=2, bw_adjust=.5)
g.set(xlim=(-0.2, 2))

# Reference line at y=0
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Label function
def label(x, color, label):
    ax = plt.gca()
    ax.text(-0.2, .65, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes, fontsize=8)

g.map(label, "Cytoplasm: DAB_vec OD mean")

# Adjust subplot overlap
g.figure.subplots_adjust(hspace=-0.25)

g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)

print("Displaying plot...")
plt.savefig("test.png")
