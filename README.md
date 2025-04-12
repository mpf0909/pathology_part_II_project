# part_II_project

## Overview
This repo contains all code used for my Pathology Part II Project. My project centred on using machine-learning approaches - specifically Hovernet (Graham et al., 2017) to diagnose normal duodenal biopsies based on the underlying assumption that densities of various cell types (e.g. lymphocyte, epithelial cell, neutrophil, plasma cell, eosinophil, connective cell) will characteristically vary from normal levels in pathology. This was achieved by training Hovernet models on two types of data: 

1. Publically available histopathologist paired-labelled datasets (CPM17, PanNuke, Lizard)
2. My own paired-labelled datasets (CD3, CK) [more info later]

Diagnostic performance was then assessed on unseen clinically obtained, processed and histopathologist-diagnosed H&E stained tissue images. The diagnostic results are as follows
_____________________________________________________________________________________________________________________________________________________________________________
| Model                   | Precision (normal) | Precision (pathology) | Recall (normal) | Recall (pathology) | f1-score (normal) | f1-score (pathology) | Overall accuracy |
|-------------------------|--------------------|-----------------------|-----------------|--------------------|-------------------|----------------------|------------------|
| Lizard-trained Hovernet | 0.83               | 1.00                  | 1.00            | 0.86               | 0.91              | 0.92                 | 0.9167           |
| CD3-trained Hovernet    | 1.00               | 0.64                  | 0.20            | 1.00               | 0.33              | 0.78                 | 0.6667           |
| CK-trained Hovernet     | 0.44               | 0.67                  | 0.80            | 0.29               | 0.57              | 0.40                 | 0.5000           |
_____________________________________________________________________________________________________________________________________________________________________________

These results indicate that machine learning-based cell segmentation and classification approaches have the potential to automate the diagnosis of normal duodenal biopsies.

NB this repo has not currently been designed for external use e.g. hard-coded file paths, lack of command-line arguments - however the end of this README.md contains a step-by-step guide of how to use this repo.

## Structure of repo
### hovernet directory
This directory contains all of the code used to train, validate, run inference and analyse both publically available datasets and my own datasets. It also contains output data from training, validation and inference (not synced). 
- The original-hovernet subdirectory contains the aforementioned files for use with the CPM17 and PanNuke dataset. Large parts of this directory were obtained from cloning the Hovernet master branch: https://github.com/vqdang/hover_net/tree/master
- The hovernet-conic subdirectory contains the aforementioned files for use with my datasets and the Lizard dataset. Large parts of this directory were obtained from cloning the Hovernet conic branch: https://github.com/vqdang/hover_net/tree/conic

### in-house directory
This directory contains all of the code and data to create my own paired-labelled datasets. It also contains clinically-obtained H&E-stained data for final inference to assess performance of each Hovernet model


