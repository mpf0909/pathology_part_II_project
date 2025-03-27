# WSI overlay repository

This is a research and development repository containing work conducted by the Cambridge spin-out company Lyzeum Ltd. It has been adapted by Matthew Ferguson for use in his Part II Pathology Research Project.

## Getting started

#### Installation

To clone and install this repo:
```bash
git clone git@gitlab.com:lyzeum/overlay-wsis.git
cd overlay-wsis/
conda env create -f requirements.conda.yaml
conda activate slide_overlay
conda install -c conda-forge openslide
pre-commit install
```
*Please* make sure you run the ``pre-commit install`` to make sure automated quality control checks are turned on.


## Running code

In order to run the WSI overlaying code run:
```bash
python overlay_wsis.py $wsi_1_path $wsi_2_path --save_path_one $save_path_1 --save_path_two $save_path_2
```

Note that ``save_path_1`` and ``save_path_2`` need to be .ome.tif files.


## TODO
- Create the lower pyramid levels with a clevere averaging function.
- Add check for pixel size unit.
- Note add an option where we don't change image 1 and only align image 2 (would result in black edges of image 2 but may be helpful for certain applications)
- Maybe include loop that runs the code up to 3 times if the alignmemnt isn't perfect yet.



## QuPath vs OpenSlide:

Currently, the code can only either write .ome.tif files where OpenSlide can detect the pyramid structure, OR .ome.tif that work with QuPath.

I believe (and may be wrong here) that OpenSlide and QuPath expect different TIFF structures. In the code there are two branches:

For QuPath: We pass the keyword subifds=subresolutions when writing the base image. This causes the TIFF writer to reserve space in the base image’s IFD (Image File Directory) for the pyramid levels. QuPath depends on that SubIFDs pointer to locate the reduced‐resolution images.
For OpenSlide: We omit that parameter so that the pyramid levels are written as independent IFDs linked via the “next IFD” pointer chain. OpenSlide’s parser expects to find the pyramid levels this way rather than being “attached” via a SubIFDs array.
Because OpenSlide doesn’t (yet) support reading pyramid levels from a SubIFDs tag, when we include it (i.e. for QuPath) OpenSlide won’t detect the levels. On the other hand, if we remove the subifds parameter so that OpenSlide can see a linear chain of IFDs, QuPath—which requires the explicit SubIFDs pointer—won’t work.

In short, the TIFF file layout that QuPath expects is different from the one that OpenSlide expects. That’s why our file can be “OpenSlide‐compatible” only if we omit the SubIFDs pointer, and “QuPath‐compatible” only if we include it.

To support both viewers simultaneously, we would need to generate a TIFF pyramid structure that satisfies both readers. Unfortunately, there isn’t (currently) a single universally accepted OME-TIFF pyramid layout that both OpenSlide and QuPath parse correctly. We'll have to choose one structure or generate two files with the appropriate internal organization for each viewer.