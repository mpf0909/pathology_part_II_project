# part_II_project

## Overview
This repo contains all of the code used for my Pathology Part II Project. My project centred on using machine learning-based cell segmentation and classification approaches - specifically [Hovernet](https://www.sciencedirect.com/science/article/pii/S1361841519301045) (Graham et al., 2019) - to diagnose normal duodenal biopsies. This research was motivated by the assumption that densities of various cell types (e.g. lymphocyte, epithelial cell, neutrophil, plasma cell, eosinophil, connective cell) will characteristically vary from normal levels in pathology. This was achieved by training hovernet models on two types of data:

1. Publically available histopathologist paired-labelled datasets ([CPM17](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK), [PanNuke](link), [Lizard](https://drive.google.com/drive/folders/1il9jG7uA4-ebQ_lNmXbbF2eOK9uNwheb))
2. Self-generated paired-labelled datasets (CD3, CK)

Diagnostic performance was then assessed on unseen clinically obtained, processed and histopathologist-diagnosed H&E stained duodenal tissue images exclusively available to the Soilleux Lab, University of Cambridge and its spinout company Lyzeum. The test dataset included 15 normal cases, 15 coeliac cases, 5 adenomas, 2 adenocarcinomas, 2 ulcer, 2 neuroendocrine tumour. The diagnostic results are as follows:

| Model                   | Precision (normal) | Precision (pathology) | Recall (normal) | Recall (pathology) | f1-score (normal) | f1-score (pathology) | Overall accuracy |
|-------------------------|--------------------|-----------------------|-----------------|--------------------|-------------------|----------------------|------------------|
| Lizard-trained hovernet | 0.83               | 1.00                  | 1.00            | 0.86               | 0.91              | 0.92                 | 0.9167           |
| CD3-trained hovernet    | 1.00               | 0.64                  | 0.20            | 1.00               | 0.33              | 0.78                 | 0.6667           |
| CK-trained hovernet     | 0.44               | 0.67                  | 0.80            | 0.29               | 0.57              | 0.40                 | 0.5000           |

As shown above, Lizard-trained hovernet accurately classifies duodenal biopsies as normal or pathological in ~91% of cases thereby supporting the hypothesis that machine learning-based cell segmentation and classification approaches have the potential to automate the diagnosis of normal duodenal biopsies.

Whilst the CD3-trained hovernet and CK-trained hovernet are poorer at biopsy diagnosis this is perhaps not surprising given the CD3 and CK dataset were generated **entirely automatically** (no human labelling involved). The development of an automated pipeline to generate training data for hovernet from scratch in this project is itself a success since it enabled, across both the CD3 and CK dataset, labelling of 9,809,543 cells; **approximately 20x more cells than in the largest publically available dataset (Lizard)**. It is anticipated that further development of this pipeline will result the ability to create accurately-labelled datasets orders of magnitude larger than those currently available for future machine learning applications.

NB this repo is not currently designed for external use (e.g. it contains hard-coded file paths and frequently lacks use of command-line arguments) however the end of this README.md contains a step-by-step guide of how to use this repo.

## Structure of repo
### hovernet directory
This directory contains all of the code used to train, validate, run inference and analyse both publically available datasets and my own datasets. It also contains output data from training, validation and inference (not synced). 
- The subdirectory 'original-hovernet' contains the aforementioned files for use with the CPM17 and PanNuke dataset. Large parts of this directory were obtained from cloning the [hovernet master branch](https://github.com/vqdang/hover_net/tree/master)
- The subdirectory 'hovernet-conic' contains the aforementioned files for use with my datasets and the Lizard dataset. Large parts of this directory were obtained from cloning the [hovernet conic branch](https://github.com/vqdang/hover_net/tree/conic)

### in-house directory
This directory contains all "in-house" data used for the project. This includes (1) hovernet-compatible self-generated paired-labelled dataset for training (2) Soilleux Lab-exclusive H&E stained tissue images for inference to assess Hovernet performance. It also contains all python code used to make the self-generated dataset. 

### opensource directory
This directory contains all publically available datasets used for training and validating Hovernet.

## Step-by-step guide to using repo
### Creating your own dataset
#### Wet-lab work 
1. Stain clinically diagnosed tissue with haematoxylin and eosin (H&E) and scan whole-slide image (WSI) to .tiff or .svs file
2. Remove H&E stain from tissue and conduct IHC stain for cell specific markers (e.g. CD3, CK) and scan WSI to .tiff or .svs file
#### Computational work
3. Align H&E-stained and IHC-stained WSIs (align-wsis directory)
 - Upload H&E and IHC stained scans to 'all-unaligned-wsis' directory and a csv with corresponding mapping between matching file names
 - Create and activate conda environment for WSI alignment
```
conda env create -f align_wsis.yaml
conda activate slide_overlay
conda install -c conda-forge openslide
```
-  Run alignment (edit file paths in run_align_wsis_for_all_wsis.py as required, code written by Dr Florian Jaeckle)
```
python run_align_wsis_for_all_wsis.py
```
-  Check alignment worked by visualising output files in 'plots' diirectory
4. Segment cells in QuPath
- Download [QuPath 0.5.1](https://qupath.github.io/) (Bankhead et al., 2017). Other versions may also work.
- Install [StarDist extension for QuPath 0.5.0](https://github.com/qupath/qupath-extension-stardist) (Schmidt et al., 2018)
- Create QuPath project containing all IHC-stained WSIs
- Download (1) 'he_heavy_augment.pb' (2) tissue_pixel_classifier.json (3) 'all_steps_universal_stardist_for_qupath.groovy' (Zaidi et al., 2021) from my [Google Drive](https://drive.google.com/file/d/1qjYjfrHR4DdZCgTn2bSmcUVI4lA6lRqn/view?usp=drive_link) and change file paths as appropriate. You can alternatively create your own pixel classifier that accurately separates tissue from background
- Run segmentation for all IHC-stained WSIs in QuPath project via Automate -> Script Editor -> all_steps_universal_stardist_for_qupath.groovy' -> Run for Project. I strongly recommend you run this overnight since this can take several hours!
5. Convert QuPath annotations into mask patches (create-masks directory)
- Upload QuPath .geojson and .txt files to 'segmentation-annotation-data' subdirectory
- Create and activate conda environment for creating masks from QuPath .geojson files
```
conda env create -f conda.requirements.yaml
conda activate tiatoolbox
conda install -c conda-forge openslide
```
- Convert QuPath .txt files to .csv files (edit file paths in convert_segmentation_annotations_txts_to_csvs.py as required)
```
python convert_segmentation_annotations_txts_to_csvs.py
```
- Calculate thresholds for classification of annotations using format provided in Jupyter Notebooks
- Create masks patches of size 256x256 pixels (format compatible with hovernet)
6. Match mask patches to H&E patches and IHC patches (create-he-ihc-patches directory)
- Create and activate conda environment for creating patches from WSIs
```
conda env create -f conda.requirements.yaml
conda activate lyzeum_patch_extractor
cd lyzeum-ml
pip install .
pip install --upgrade openslide-python
```
- Install [QuPath 0.3.2](https://github.com/qupath/qupath) and add a symbolic link ``"qupath"`` pointing to the QuPath executable binary in a folder added to your PATH. 
- Create patches for aligned H&E and IHC WSIs (edit file path in extract_patches_from_directory.py as required, code written by Dr Florian Jaeckle)
```
python extract_patches_from_directory.py
```
7. Create hovernet-compatible training data (create-training-data directory)
- Create he-images.npy ihc-images.npy and masks.npy for each WSI. Each index of these .npy files matches with the other two .npy files at the same index (edit file paths in create_training_data_per_wsi_for_all_wsis.py as required)
 ```
python create_training_data_per_wsi_for_all_wsis.py
```
- Create dataset splits for training and validation (edit file paths and number of folds in create_CD3_split.py as required)
```
python create_CD3_split.py
```
- Create final training data by aggregating together individual WSI he-images.npy files (edit file paths in create_final_training_data.py as required)
```
python create_final_training_data.py
```
- Create csv file counting number of each nuclei in dataset (edit file paths in create_counts_csv.py as required)
```
python create_counts_csv.py
```
- (Optional) assess statistical significance in cell type density between different pathologies (edit file paths in analyse_summary_density_diagnosis_csv.ipynb as required)
You have now successfully created a Hovernet-compatible, paired-labelled dataset in a <ins>**fully automated fashion**</ins> (no requirement for human labelling).

### Training hovernet
Whilst this repo contains code enabling training of Hovernet on CPM17, PanNuke, Lizard and self-generated datasets, only training of Hovernet on Lizard and self-generated datasets is of any functional relevance for diagnosing duodenal biopsies since CPM17 and PanNuke datasets contain paired-lablled data entirely from cancer tissue.

Whilst both hovernet-conic and original-hovernet directories can be used for training hovernet - most development has occured 



