"""Overlaying two WSIs without PyVips."""

# pylint: disable=no-member, too-many-locals, fixme
# pylint: disable=too-many-arguments, too-many-positional-arguments
import os
import gc
import argparse
from pathlib import Path
import openslide
import cv2
import numpy as np
import tifffile as tf
from tifffile import TiffWriter  # type: ignore
import xml.etree.ElementTree as ET

np.set_printoptions(suppress=True, precision=8)


def overlay_images_new(
    image_path_1: Path,
    image_path_2: Path,
    save_path_1: Path,
    save_path_2: Path,
    level: int,
    count: int,
    max_tries: int,
) -> None:
    """Align two WSIs and save as pyramidal TIFFs using OpenSlide and Tifffile.

    Parameters
    ----------
    image_path_1 : Path
        Path of first image.
    image_path_2 : Path
        Path of second image.
    save_path_1 : Path
        Save path of the first aligned image.
    save_path_2 : Path
        Save path of the second aligned image.
    level : int
        Level of the WSI pyramid to use for the affine matrix.
    """
    # Get the affine transformation matrix
    scaled_matrix = get_affine_mask(image_path_1, image_path_2, level=level)
    print(f"affine matrix before: {scaled_matrix}")

    # Open WSIs using OpenSlide
    slide1 = openslide.OpenSlide(str(image_path_1))
    slide2 = openslide.OpenSlide(str(image_path_2))

    mpp_x = get_properties(slide1)

    # Read the full-resolution image from WSI 2
    # TODO swamp the naming of height and width
    # Note: openslide does (width, height) whereas numpy does height, width
    width1, height1 = slide1.level_dimensions[0]
    width2, height2 = slide2.level_dimensions[0]
    print(f"Dimensions image 1: {width1=}, {height1=}")
    print(f"Dimensions image 2: {width2=}, {height2=}")

    # Apply affine transformation using OpenCV
    affine_matrix = np.array(
        [
            [scaled_matrix[0, 0], scaled_matrix[0, 1], scaled_matrix[0, 2]],
            [scaled_matrix[1, 0], scaled_matrix[1, 1], scaled_matrix[1, 2]],
        ]
    )

    # We're transforming image2 into image 1's coordinate system
    # Note we are also making it the same size as image1
    # We achieve that by adding black pixels on the required edges.
    # TODO Think about whether this should be (height1, width1) or really
    # (max(height1, height2), max(width1, width2))
    if max(width1, height1, width2, height2) > 32700:
        # We use our own implementation if the image is large as cv2 breaks otherwise
        transformed_image = warp_affine_tiled(slide2, affine_matrix, (height1, width1))
    else:
        # We use cv2 directly for small images
        image2 = slide2.read_region((0, 0), 0, (width2, height2)).convert("RGB")
        image2 = np.array(image2)
        transformed_image = cv2.warpAffine(
            image2, affine_matrix, (width1, height1), flags=cv2.INTER_CUBIC
        )

    ## Note: the two images are now aligned so we could save them now.
    ## However, image 2 includes black parts as a result of the rotation.
    ## Therefore we crop the images to get rid of any black edges.

    # Compute the positions of the four corners in the image1 coordinate system
    new_corners2 = (
        np.array([[0, 0], [width2, 0], [width2, height2], [0, height2]])
        @ scaled_matrix[:, :2].T
    ) + scaled_matrix[:, 2]

    width1, height1 = slide1.level_dimensions[0]

    # Dimensions a rectangle that fits inside both WSIs (using the image1 coordinate system).
    # Note that this is not optimal, once could compute larger rectangels to fit.
    # The resulting are the largest
    # TODO: i think we're potentially cropping too much at the bottom and the right edge
    #       especially if image 1 is smaller than image 2
    xmin = round(max(0, new_corners2[0, 0], new_corners2[3, 0]))
    ymin = round(max(0, new_corners2[0, 1], new_corners2[1, 1]))
    # xmax = round(min(width1, new_corners2[1, 0], new_corners2[2, 0]))
    # ymax = round(min(height1, new_corners2[2, 1], new_corners2[3, 1]))
    xmax = round(min(width1, new_corners2[1, 0], new_corners2[2, 0]))
    ymax = round(min(height1, new_corners2[2, 1], new_corners2[3, 1]))

    transformed_image_cropped = transformed_image[ymin:ymax, xmin:xmax]
    transformed_image = None
    gc.collect()

    # Saving the cropped image as a .ome.tif file
    save_ome_pyramidal_tiff(transformed_image_cropped, save_path_2, mpp_x)
    transformed_image_cropped = None
    gc.collect()

    image1 = get_image1(slide1, xmin, xmax, ymin, ymax)

    print(f"Dimensions of aligned images: width={xmax-xmin}, height={ymax-ymin}")
    # Saving image 1 as .ome.tif file
    save_ome_pyramidal_tiff(image1, save_path_1, mpp_x)

    image1 = None
    gc.collect()

    assessment = final_quality_check(save_path_1, save_path_2)
    if assessment == True:
        return
    if count == max_tries:
        with open("failed_alignments.txt", "a") as f:
            f.write(
                f"{image_path_1} AND {image_path_2}\n"
            )  # Or use image_path_2 if both should be saved
        return
    if assessment == False:
        count += 1
        overlay_images_new(
            save_path_1, save_path_2, save_path_1, save_path_2, level, count, max_tries
        )
    else:
        return


def final_quality_check(save_path_1: Path, save_path_2: Path):
    """Check whether created images are aligned.

    Parameters
    ----------
    save_path_1 : Path
        First aligned image path.
    save_path_2 : Path
        Second aligned image path.

    Raises
    ------
    RuntimeError
        Raise error if the two aligned images, aren't aligned well.
    """
    check_matrix = get_affine_mask(save_path_1, save_path_2, level=0)
    if abs(check_matrix[:, 2]).max() > 4:
        print(f"Poor alignment. Translation matrix: {check_matrix}. Trying again.")
        return False
    else:
        return True


def get_image1(slide1, xmin: int, xmax: int, ymin: int, ymax: int) -> np.ndarray:
    """Return cropped image 1 as a numpy ndarray.

    Parameters
    ----------
    slide1 : _type_
        Openslide slide.
    xmin : int
        xmin index.
    xmax : int
        xmax index.
    ymin : int
        ymin index.
    ymax : int
        ymax index.

    Returns
    -------
    np.ndarray
        Cropped image 1 as a numpy array or memory-map array.
    """
    # TODO: rather than read entire region from slide transforming to Image and then cropping
    # I should read it directly
    image1 = slide1.read_region((xmin, ymin), 0, (xmax - xmin, ymax - ymin)).convert(
        "RGB"
    )
    width, height = image1.size  # PIL gives (width, height)

    if width * height < 50000 * 50000:
        image_array = np.array(image1)

    else:
        # Define shape and dtype.
        shape = (
            height,
            width,
            3,
        )  # PIL uses (width, height); NumPy expects (rows, cols, channels)
        dtype = np.uint8

        memmap_path = "./aligned_images/image1_memmap.dat"

        # Create the memmap array in write-plus mode.
        image_array = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=shape)

        # Copy the data from the PIL image into the memmap.

        # chunk_size = int(170000000 / width)
        chunk_size = int(89000000 / width)

        for start in range(0, height, chunk_size):
            end = min(start + chunk_size, height)
            # Crop the current chunk from the PIL image.
            chunk = image1.crop((0, start, width, end))
            # Convert the chunk to a NumPy array and store it in the memmap.
            image_array[start:end, :, :] = np.array(chunk)

        # Flush the changes to disk.
        image_array.flush()

        # TODO check whether this works or whether I need to load the array in read mode.

    return image_array


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
    slide_path = slide._filename if hasattr(slide, "_filename") else None

    if slide_path and slide_path.lower().endswith(".ome.tif"):
        # Use tifffile for OME-TIFF
        try:
            with tf.TiffFile(slide_path) as tif:
                metadata = tif.ome_metadata
                if metadata:
                    namespace = {
                        "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                    }
                    root = ET.fromstring(metadata)
                    pixel_size = root.find(".//ome:Pixels", namespace).get(
                        "PhysicalSizeX"
                    )
                    return float(pixel_size) if pixel_size else None
        except Exception as e:
            print(f"OME-TIFF error: {e}")
            return None

    else:
        # Use OpenSlide for standard TIFF
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None)
        return float(mpp_x) if mpp_x is not None else None


def save_ome_pyramidal_tiff(
    image_array: np.ndarray, save_path: Path, mpp_x: float, version="QuPath"
):
    """Save numpy array as .ome.tif file with QuPath or OpenSlide-compatible pyramid structure.

    Parameters
    ----------
    image_array : np.ndarray
        The image array to save.
    save_path : Path
        The .ome.tif file path to use for saving.
    mpp_x : float
        Pixel size in µm.
    version : str
        Specify QuPath or OpenSlide compatibility.
    """
    assert version in ["QuPath", "OpenSlide"]

    assert (
        "".join(save_path.suffixes[-2:]) == ".ome.tif"
    ), "Save path needs to be a .ome.tif file"

    subresolutions = 5
    pixelsize = mpp_x  # Micrometers per pixel

    with TiffWriter(save_path, bigtiff=True) as tif:
        metadata = {
            "axes": "YXS",
            "SignificantBits": 8,
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["R", "G", "B"]},
        }
        options = dict(
            photometric="rgb",
            tile=(256, 256),  # Larger tile size is better for OpenSlide
            compression="jpeg",
            resolutionunit="CENTIMETER",
            maxworkers=2,
        )

        # Write base image with subIFDs
        if version == "QuPath":
            tif.write(
                image_array,
                subifds=subresolutions,  # Explicitly create SubIFDs
                resolution=(1e4 / pixelsize, 1e4 / pixelsize),
                metadata=metadata,
                **options,
            )
        elif version == "OpenSlide":
            tif.write(
                image_array,
                resolution=(1e4 / pixelsize, 1e4 / pixelsize),
                metadata=metadata,
                **options,
            )

        # Generate and write pyramid levels (SubIFDs)
        # downsampled_images = []
        for level in range(subresolutions):
            scale = 2 ** (level + 1)
            new_size = (image_array.shape[1] // scale, image_array.shape[0] // scale)
            downsampled = cv2.resize(
                image_array, new_size, interpolation=cv2.INTER_AREA
            )

            tif.write(
                downsampled,
                subfiletype=1,  # Marks as a reduced-resolution image
                resolution=(1e4 / (pixelsize * scale), 1e4 / (pixelsize * scale)),
                **options,
            )

        # Add a thumbnail image as a separate series for QuPath
        thumbnail = (image_array[::16, ::16] >> 2).astype("uint8")
        tif.write(thumbnail, metadata={"Name": "thumbnail"})

    print(f"OME-TIFF with {version}-compatible pyramid saved successfully: {save_path}")


def get_affine_mask(
    image_path_1: Path, image_path_2: Path, level: int, max_size=50000
) -> np.ndarray:
    """Compute the affine transformation matrix to align two WSIs.

    Parameters
    ----------
    image_path_1 : Path
        Path of first image.
    image_path_2 : Path
        Path of second image.

    Returns
    -------
    np.ndarray
        scaled_matrix

    Raises
    ------
    FileNotFoundError
        Raises error if either of the two images aren't found.
    """
    if not os.path.exists(image_path_1) or not os.path.exists(image_path_2):
        raise FileNotFoundError("One or both image paths are invalid.")

    slide1 = openslide.OpenSlide(str(image_path_1))
    slide2 = openslide.OpenSlide(str(image_path_2))

    # TODO turn this into an argparse argument
    level = min(
        level, len(slide1.level_dimensions) - 1, len(slide1.level_dimensions) - 1
    )
    print("level for affine matrix", level)

    width1, height1 = slide1.level_dimensions[level]
    width2, height2 = slide2.level_dimensions[level]
    if max(width1 * height1, width2 * height2) > max_size * max_size:
        left = max(0, (min(width1, width2) - max_size) // 2)
        top = max(0, (min(height1, height2) - max_size) // 2)
        dim1 = dim2 = (max_size, max_size)
    else:
        left = top = 0
        dim1 = slide1.level_dimensions[level]
        dim2 = slide2.level_dimensions[level]

    # if we want to support other levels, we need to assert that the magnification is the same
    # also add is as an argparse
    # region1 = slide1.read_region((0, 0), level, slide1.level_dimensions[level])
    region1 = slide1.read_region((left, top), level, dim1)
    region1_gray = cv2.cvtColor(np.asarray(region1.convert("RGB")), cv2.COLOR_RGB2GRAY)
    region1 = None
    gc.collect()
    print("computed gray scale 1")

    # region2 = slide2.read_region((0, 0), level, slide2.level_dimensions[level])
    region2 = slide2.read_region((left, top), level, dim2)
    region2_gray = cv2.cvtColor(np.asarray(region2.convert("RGB")), cv2.COLOR_RGB2GRAY)
    region2 = None
    gc.collect()
    print("computed gray scale 2")

    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(region1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(region2_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(descriptors1, descriptors2), key=lambda x: x.distance
    )

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    matrix, _ = cv2.estimateAffinePartial2D(points2, points1, method=cv2.RANSAC)
    scale_factor = slide1.level_downsamples[level]

    scaled_matrix = matrix.copy()
    scaled_matrix[0, 2] *= scale_factor
    scaled_matrix[1, 2] *= scale_factor

    return scaled_matrix


def warp_affine_tiled(
    slide,
    transform_matrix: np.ndarray,
    dsize: tuple,
    max_tile_size: int = 10000,
) -> np.ndarray:
    """
    Apply an affine transformation to a large image in a tiled fashion.

    Parameters
    ----------
    transform_matrix : np.ndarray
        The 2x3 affine transformation matrix mapping destination coordinates to source coordinates.
    dsize : tuple
        Overall output (destination) size as (height, width).
    max_tile_size : int, optional
        Maximum allowed width or height for the (central) destination tile (default is 10000).

    Returns
    -------
    np.ndarray
        The fully warped image.
    """
    # TODO: only use memmap file if the image is large, otherwise use numpy array directly for speed
    out_height, out_width = dsize
    # Create the output image container.
    # transformed = np.zeros((out_height, out_width, image.shape[2]), dtype=image.dtype)
    # Create a memory-mapped file
    # TODO: create folder if it doesn't exist
    memmap_filename = "./aligned_images/transformed_memmap.dat"
    transformed_memmap = np.memmap(
        memmap_filename, dtype="uint8", mode="w+", shape=(out_height, out_width, 3)
    )

    # We'll process the destination image in tiles,
    # each with a central (core) area of size at most max_tile_size.
    dest_tile_w = max_tile_size
    dest_tile_h = max_tile_size

    for ymin in range(0, out_height, dest_tile_h):
        # Determine the height of the central (core) tile.
        ymax = min(ymin + dest_tile_h, out_height)
        for xmin in range(0, out_width, dest_tile_w):
            # Determine the width of the central (core) tile.
            xmax = min(xmin + dest_tile_w, out_width)
            print(f"{xmin=}, {xmax=}, {ymin=}, {ymax=}")

            warped_tile = get_affine_tile(
                slide, transform_matrix, xmin, ymin, xmax, ymax
            )

            # Assign values
            transformed_memmap[ymin:ymax, xmin:xmax] = warped_tile

    # Flush changes to disk
    transformed_memmap.flush()
    # Load transformed array
    transformed = np.memmap(
        memmap_filename, dtype="uint8", mode="r+", shape=(out_height, out_width, 3)
    )

    return transformed


def get_affine_tile(
    slide,
    transform_matrix: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> np.ndarray:
    """Exctracting transformed patch from image using coordinates from original image.

    We have a source image (parameter image). We want to extract the patch from image
    that corresponds to xmin, xmax, ymin, ymax in the target image coordinates.
    We use the transform_matrix to map from source image to target image.

    TODO: add a test that xmax-xmin and ymax-ymin aren't over 32k
        as otherwise cv2.warpAffine breaks.

    Parameters
    ----------
    transform_matrix : np.ndarray
        affine transformation matrix (2x3 matrix, first 2x2 rotation, last column translation)
    xmin : int
        xmin coordinates of patch in target image.
    ymin : int
        ymin coordinates of patch in target image.
    xmax : int
        xmax coordinates of patch in target image.
    ymax : int
        ymax coordinates of patch in target image.

    Returns
    -------
    np.ndarray
        The patch from source image that will lie in the given coordinates in the target image.
    """
    # Determine a crop region that covers the ROI plus a margin.
    # (Make sure the crop is within image bounds.)

    points = np.array(
        [
            print_position_in_im1_coordinate_system(
                xmin, ymin, transform_matrix, invert=True
            ),
            print_position_in_im1_coordinate_system(
                xmax, ymin, transform_matrix, invert=True
            ),
            print_position_in_im1_coordinate_system(
                xmin, ymax, transform_matrix, invert=True
            ),
            print_position_in_im1_coordinate_system(
                xmax, ymax, transform_matrix, invert=True
            ),
        ]
    )

    x_crop = max(0, int(points.squeeze()[:, 0].min()) - 100)
    y_crop = max(0, int(points.squeeze()[:, 1].min()) - 100)

    # updated as
    source_width, source_height = slide.level_dimensions[0]
    x_max_crop = min(source_width, int(points.squeeze()[:, 0].max()) + 100)
    y_max_crop = min(source_height, int(points.squeeze()[:, 1].max()) + 100)

    # Crop the large image to a smaller sub-image.
    cropped = np.array(
        slide.read_region(
            (x_crop, y_crop), 0, (x_max_crop - x_crop, y_max_crop - y_crop)
        ).convert("RGB")
    )

    # Compute the desired ROI dimensions.
    roi_width = xmax - xmin
    roi_height = ymax - ymin

    # Adjust the transformation matrix to the cropped coordinate system.
    # The original matrix maps from destination (tile) to full image coordinates.
    # We need to subtract the crop offset (x0, y0).
    m_roi = transform_matrix.copy().astype("float32")
    m_roi[:, 2] += m_roi[:, :2] @ np.array([x_crop, y_crop])
    m_roi[:, 2] += np.array([-xmin, -ymin])

    # return zero vector in case the cropped image is empty
    if 0 in cropped.shape:
        return np.zeros((roi_height, roi_width, 3), dtype=np.dtype("uint8"))

    # Now apply warpAffine to the cropped image.
    roi_image = cv2.warpAffine(
        cropped, m_roi, (roi_width, roi_height), flags=cv2.INTER_CUBIC
    )

    return roi_image


def print_position_in_im1_coordinate_system(
    x: float, y: float, scaled_matrix, invert=False
) -> np.ndarray:
    """Map image 2 coordinates to image 1 coordinates or vice versa.

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    scaled_matrix : _type_
        affine transformation matrix (2x3 matrix, first 2x2 rotation, last column translation)
    invert : bool, optional
        if True we map image 2 to image 1, if false map from image 2 to image 1, by default False

    Returns
    -------
    np.array
        New point as a numpy array.
    """
    if invert:
        # Convert the 2x3 matrix to a 3x3 matrix.
        affine_m_extended = np.vstack([scaled_matrix, [0, 0, 1]])
        # Invert the 3x3 matrix.
        affine_m_extended_inv = np.linalg.inv(affine_m_extended)
        # Return the upper 2x3 part, which is the affine matrix mapping image1 -> image2.
        scaled_matrix = affine_m_extended_inv[:2, :]

    new_point = (np.array([[x, y]]) @ scaled_matrix[:, :2].T) + scaled_matrix[:, 2]

    return new_point


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        argparse
    """
    parser = argparse.ArgumentParser(
        description="Overlay two images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("wsi_file_one", help="Path to the first WSI", type=str)
    parser.add_argument("wsi_file_two", help="Path to the second WSI", type=str)
    parser.add_argument(
        "--save_path_one",
        help="Path to save the aligned first WSI",
        type=str,
        default="./wsi_one_aligned.tif",
    )
    parser.add_argument(
        "--save_path_two",
        help="Path to save the aligned second WSI",
        type=str,
        default="./wsi_two_aligned.tif",
    )
    parser.add_argument(
        "--level",
        help="WSI level to use for calculating the affine matrix",
        type=int,
        default=0,
    )

    return parser.parse_args()


count = 1

if __name__ == "__main__":
    # TODO: remove temporary .dat files if they exist
    # TODO: add argparse for max crop for affine matrix
    args = parse_command_line_arguments()
    overlay_images_new(
        image_path_1=Path(args.wsi_file_one),
        image_path_2=Path(args.wsi_file_two),
        save_path_1=Path(args.save_path_one),
        save_path_2=Path(args.save_path_two),
        level=args.level,
        count=count,
        max_tries=3,
    )
