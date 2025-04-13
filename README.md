# part_II_project

## Overview
This repo contains all of the code used for my Pathology Part II Project. My project centred on using machine learning-based cell segmentation and classification approaches - specifically [Hovernet](https://www.sciencedirect.com/science/article/pii/S1361841519301045) (Graham et al., 2019) - to diagnose normal duodenal biopsies. This research was motivated by the assumption that densities of various cell types (e.g. lymphocyte, epithelial cell, neutrophil, plasma cell, eosinophil, connective cell) will characteristically vary from normal levels in pathology. This was achieved by training hovernet models on two types of data:

1. Publically available histopathologist paired-labelled datasets ([CPM17](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK), [PanNuke](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification), [Lizard](https://drive.google.com/drive/folders/1il9jG7uA4-ebQ_lNmXbbF2eOK9uNwheb))
2. Self-generated paired-labelled datasets (CD3, CK)

Diagnostic performance was then assessed on unseen clinically obtained, processed and histopathologist-diagnosed H&E stained duodenal tissue images exclusively available to the Soilleux Lab, University of Cambridge and its spinout company, Lyzeum. The test dataset included 15 normal cases, 15 coeliac cases, 5 adenomas, 2 adenocarcinomas, 2 ulcer, 2 neuroendocrine tumour. A binary classifier logistic regression model was fitted using densities of each cell type identified by each hovernet. The diagnostic results are as follows:

| Model                   | Precision (normal) | Precision (pathology) | Recall (normal) | Recall (pathology) | f1-score (normal) | f1-score (pathology) | Overall accuracy |
|-------------------------|--------------------|-----------------------|-----------------|--------------------|-------------------|----------------------|------------------|
| Lizard-trained hovernet | 0.83               | 1.00                  | 1.00            | 0.86               | 0.91              | 0.92                 | 0.9167           |
| CD3-trained hovernet    | 1.00               | 0.64                  | 0.20            | 1.00               | 0.33              | 0.78                 | 0.6667           |
| CK-trained hovernet     | 0.44               | 0.67                  | 0.80            | 0.29               | 0.57              | 0.40                 | 0.5000           |

As shown above, Lizard-trained hovernet accurately classifies duodenal biopsies as normal or pathological in ~91% of cases thereby supporting the hypothesis that machine learning-based cell segmentation and classification approaches have the potential to automate the diagnosis of normal duodenal biopsies.

Whilst the CD3-trained hovernet and CK-trained hovernet are poorer at biopsy diagnosis this is perhaps not surprising given the CD3 and CK dataset were generated **entirely automatically** (no human labelling involved). The development of an automated pipeline to generate training data for hovernet from scratch in this project is itself a success since it enabled, across both the CD3 and CK dataset, labelling of 9,809,543 cells; **approximately 20x more cells than in the largest publically available dataset (Lizard)**. It is anticipated that further development of this pipeline will enable the creation of accurately-labelled datasets that are orders of magnitude larger than those currently available for future machine learning applications.

NB this repo is not currently designed for external use (e.g. it contains hard-coded file paths and frequently lacks use of command-line arguments) however the end of this README.md contains a step-by-step guide of how to use this repo.

## Structure of repo
### hovernet directory
This directory contains all of the code used to train, validate, run inference and analyse both publically available datasets and my own datasets. It also contains output data from training, validation and inference (not synced). 
- ```hovernet/original-hovernet/``` contains the aforementioned files for use with the CPM17 and PanNuke dataset. Large parts of this directory were obtained from cloning the [hovernet master branch](https://github.com/vqdang/hover_net/tree/master)
- ```hovernet/hovernet-conic/``` contains the aforementioned files for use with my datasets and the Lizard dataset. Large parts of this directory were obtained from cloning the [hovernet conic branch](https://github.com/vqdang/hover_net/tree/conic)
- The only differences in implementation between ```hovernet-conic``` and ```original-hovernet``` are that (1) ```hovernet-conic``` uses ```resnet50-0676ba61.pth``` for its pretrained weights whereas ```original-hovernet``` uses ```ImageNet-ResNet50-Preact_pytorch.tar``` and (2) ```hovernet-conic ``` uses padded deconvolution in the decoders to result in the same output size as the input patches. By contrast ```original-hovernet``` outputs patches either 80x80 or 164x164 pixels.  

Despite some redundancy between these directories they were intentionally kept separate for ease of development.

### in-house directory
This directory contains all "in-house" data used for the project. This includes (1) hovernet-compatible self-generated paired-labelled dataset for training (2) Soilleux Lab-exclusive H&E stained tissue images for inference to assess Hovernet performance. It also contains all python code used to make the self-generated dataset. 

### opensource directory
This directory contains all publically available datasets used for training and validating hovernet models.

## Step-by-step guide to using repo
### Creating your own dataset
#### Wet-lab work 
1. Stain clinically diagnosed tissue with haematoxylin and eosin (H&E) and scan whole-slide image (WSI) to .tiff or .svs file
2. Remove H&E stain from tissue and conduct IHC stain for cell specific markers (e.g. CD3, CK) and scan WSI to .tiff or .svs file
#### Computational work
Unless stated otherwise please use the ```tiatoolbox``` conda environment for all scripts

3. Align H&E-stained and IHC-stained WSIs
- Upload H&E and IHC stained scans to ```in-house/align-wsis/all-unaligned-wsis``` directory and a csv with corresponding mapping between matching file names
- Create and activate conda environment for WSI alignment
```
cd in-house/align-wsis/
conda env create -f align_wsis.yaml
conda activate slide_overlay
conda install -c conda-forge openslide
```
-  Run alignment (edit file paths for ```CSV_FILE```, ```WSI_DIRECTORY``` and ```WSI_ALIGNED_DIRECTORY``` in ```run_align_wsis_for_all_wsis.py``` as required, code written by Dr Florian Jaeckle)
```
cd in-house/align-wsis/scripts/
python run_align_wsis_for_all_wsis.py
```
-  Check alignment worked by visualising output files in ```in-house/align-wsis/plots```
4. Segment cells in QuPath
- Download [QuPath 0.5.1](https://qupath.github.io/) (Bankhead et al., 2017). Other versions may also work.
- Install [StarDist extension for QuPath 0.5.0](https://github.com/qupath/qupath-extension-stardist) (Schmidt et al., 2018)
- Create QuPath project containing all IHC-stained WSIs
- Download (1) ```he_heavy_augment.pb``` (2) ```tissue_pixel_classifier.json``` (3) ```all_steps_universal_stardist_for_qupath.groovy``` [(Zaidi et al., 2021)](https://github.com/MarkZaidi/Universal-StarDist-for-QuPath) from my [Google Drive](https://drive.google.com/file/d/1qjYjfrHR4DdZCgTn2bSmcUVI4lA6lRqn/view?usp=drive_link) and change file paths as appropriate. You can alternatively create your own pixel classifier that accurately separates tissue from background
- Run segmentation for all IHC-stained WSIs in QuPath project via Automate -> Script Editor -> all_steps_universal_stardist_for_qupath.groovy -> Run for Project. I strongly recommend you run this overnight since this can take several hours!
5. Convert QuPath annotations into mask patches
- Upload QuPath .geojson and .txt files to ```in-house/create-masks/segmentation-annotation-data/```
- Create and activate conda environment for creating masks from QuPath .geojson files
```
cd in-house/create-maasks/
conda env create -f conda.requirements.yaml
conda activate tiatoolbox
conda install -c conda-forge openslide
```
- Convert QuPath .txt files to .csv files (edit file paths for ```input_dir``` and ```output_dir``` in ```convert_segmentation_annotations_txts_to_csvs.py``` as required)
```
cd in-house/create-masks/scripts/
python convert_segmentation_annotations_txts_to_csvs.py
```
- Calculate thresholds for classification of annotations using format provided in ```calculate_CK_wsi_classification_thresholds.ipynb```
- Create masks patches of size 256x256 pixels (format compatible with hovernet)
6. Match mask patches to H&E patches and IHC patches for each WSI
- Unfortunately the lyzeum-ml directory cannot be shared since it is proprietary IP owned by Lyzeum, the Soilleux Lab's spinout company. The below instructions detail how this would be done should it become available
- Create and activate conda environment for creating patches from WSIs
```
cd in-house/create=he-ihc-patches/
conda env create -f conda.requirements.yaml
conda activate lyzeum_patch_extractor
cd lyzeum-ml
pip install .
pip install --upgrade openslide-python
```
- Install [QuPath 0.3.2](https://github.com/qupath/qupath) and add a symbolic link ``"qupath"`` pointing to the QuPath executable binary in a folder added to your PATH. 
- Create patches for aligned H&E and IHC WSIs (edit file path for ```input_dir```, ```output_dir``` and ```script_path```in ```extract_patches_from_directory.py``` as required). The below script must be run on a GPU (can be changed out of the box).
```
cd in-house/create-he-ihc-patches/additional-scripts-for-github/
python extract_patches_from_directory.py
```
7. Visualise masks to assess quality of segmentation and classification (NB - Jupyter notebook found in ```hovernet/hovernet-conic/ihc-mask-overlay``` NOT within any ```in-house``` subdirectory. Sorry I know this is confusing!)
- Edit file paths for ```DIRECTORIES```, ```he_images```, ```masks``` and ```directory``` as required 
```
cd hovernet/hovernet-conic/ihc-mask-overlay/
overlay_check_folds.ipynb
```
8. Create hovernet-compatible training data (create-training-data directory)
- Create ```he-images.npy``` ```ihc-images.npy``` and ```masks.npy``` for each WSI. Each index of these .npy files matches with the other two .npy files at the same index (edit file paths for ```output_csv```, all keys in ```base_dirs``` and ```output_dir``` in ```create_training_data_per_wsi_for_all_wsis.py``` as required)
```
cd in-house/create-training-data/scripts/
python create_training_data_per_wsi_for_all_wsis.py
```
- Create dataset splits for training and validation (edit file paths in ```info```, ```fold1_files```, ```fold2_files``` and joblib output paths in ```create_CD3_split.py``` as required)
```
cd in-house/create-training-data/scripts/
python create_CD3_split.py
```
- Create final training data by aggregating together individual WSI he-images.npy files (edit file paths for DIRECTORIES, fold_1_files, fold_2_files, patch_info_path and numpy save paths ```create_final_training_data.py``` as required). Please note this script currently only works for files with a strict naming convention and therefore may need to be edited.
```
cd in-house/create-training-data/scripts/
python create_final_training_data.py
```
- Create csv file counting number of each nuclei in dataset (edit file paths for ```labels``` and ```df.to_csv``` ```create_counts_csv.py``` as required)
```
cd in-house/create-training-data/scripts/
python create_counts_csv.py
```
- (Optional) assess statistical significance in cell type density between different pathologies (edit file paths in ```analyse_summary_density_diagnosis_csv.ipynb``` as required)
You have now successfully created a Hovernet-compatible, paired-labelled dataset in a <ins>**fully automated fashion**</ins> (no requirement for human labelling).

### Training hovernet
Whilst this repo contains code enabling training of Hovernet on CPM17, PanNuke, Lizard and self-generated datasets, only training of Hovernet on Lizard and self-generated datasets is of functional relevance for diagnosing duodenal biopsies. This is because
- CPM17 labels only segment cells - it does not classify them. Therefore only total cell density can be used as a predictor of pathology
- PanNuke labels segment and classify cells - however its classification labels (neoplastic, non-neoplastic epithelial, inflammatory, connective, dead, non-nuclei) are highly specific to the diagnosis of cancer and were found to poorly generalise to classification of non-cancerous biopsies (normal, coeliac etc).
These datasets were used to first get hovernet working (training, inference) since the dataset specifically designed for use with hovernet, [CoNSeP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/), was (and still is) behind a University of Warwick login wall. By contrast CPM17 and PanNuke were referenced in both the Hovernet paper and GitHub as datasets that can work (after much tweaking!) with the repo.

As such this section will focus on how to train hovernet with Lizard and self-generated datasets. This is all done in ```hovernet-conic``` except where specified. For those that are interested there is a final endnote outlining use of hovernet with CPM17 and PanNuke.

1. Setup pretrained backbone for hovernet models
- Download [PyTorch ImageNet ResNet50](https://download.pytorch.org/models/resnet50-0676ba61.pth) and upload to ```pretrained/```
- Edit ```pretrained_backbone``` variable in ```param/template.yaml``` to point to the downloaded weights above
2. Download [Lizard](https://drive.google.com/drive/folders/1il9jG7uA4-ebQ_lNmXbbF2eOK9uNwheb) and upload to ```opensource/lizard/``` (NB - not in ```hovernet-conic``` directory!)
3. Set model hyperparameters and runtime parameters
- change number of number of training epochs and weights for each loss component as required in ```models/hovernet/opt.py```
- change batch_size etc in ```param/template.yaml```
4. Split Lizard dataset into training and validation folds. See section 8 above for how to do this with self-generated dataset
- Edit file path for ```info``` and relative proportion of training and testing folds via ```test_size``` and ```train_size``` as required
```
cd hovernet/hovernet-conic/
python generate_split.py
```
5. Train hovernet model (edit file paths in ```WORKSPACE_DIR```, ```DATA_DIR```, ```FOLD_IDX```, ```splits``` as required). This requires a GPU.
- Create and activate conda environment for training. This environment is designed to be compatible for training with the GPU's available through the University of Cambridge's HPC service.
```
cd hovernet/hovernet-conic/
conda env create -f conda.requirements.yaml
conda activate /home/mf774/rds/hpc-work/condaenvs/hovernet
```
```
python run.py
```
6. Check loss functions
7. Validation
8. Inference
9. Analysis

### Endnote - using hovernet with CPM17 and PanNuke
1. Highly simplified from original hovernet GitHub for easier use. Put name of dataset in
2. Convert PanNuke Dataset
3. Train
4. Validate
5. Inference

## Citation
If any part of this code is used please give appropriate citation to this GitHub

## Authors
- [Matthew Ferguson](https://github.com/mpf0909)