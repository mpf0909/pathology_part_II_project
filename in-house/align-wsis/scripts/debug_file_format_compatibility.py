import openslide
import tifffile as tf
import xml.etree.ElementTree as ET
import os

def get_properties(slide: openslide.OpenSlide) -> float:
    """Get pixel size in µm from either a standard TIFF or OME-TIFF file.

    Parameters
    ----------
    slide : OpenSlide object
        OpenSlide object representing the image.

    Returns
    -------
    float
        Pixel size in µm or None if not found.
    """
    # Check if it's an OME-TIFF using the file name stored in slide._filename
    slide_path = slide._filename if hasattr(slide, '_filename') else None

    if slide_path and slide_path.lower().endswith('.ome.tif'):
        # Use tifffile for OME-TIFF
        try:
            with tf.TiffFile(slide_path) as tif:
                metadata = tif.ome_metadata
                if metadata:
                    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    root = ET.fromstring(metadata)
                    pixel_size = root.find('.//ome:Pixels', namespace).get('PhysicalSizeX')
                    return float(pixel_size) if pixel_size else None
        except Exception as e:
            print(f"OME-TIFF error: {e}")
            return None

    else:
        # Use OpenSlide for standard TIFF
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
        return float(mpp_x) if mpp_x is not None else None

# Usage example
image_path_1 = "/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/all-unaligned-wsis/POST_IHC_PS23-18669_A1_normal_HE-CD3.svs"
slide1 = openslide.OpenSlide(image_path_1)
mpp_x = get_properties(slide1)
print(mpp_x)


