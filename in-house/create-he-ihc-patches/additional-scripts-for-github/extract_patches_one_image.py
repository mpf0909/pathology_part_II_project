#!/usr/bin/env python3
"""Obtain patches from whole slide images (WSIs) to prepare for experiments."""
import argparse
from pathlib import Path
from typing import List, Union

from lyzeum_ml.patch_extraction import PatchExtractor
from numpy import isnan, maximum
from pandas import DataFrame, concat, read_csv  # type: ignore

PathLike = Union[str, Path]


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate tiles from Whole slide images (WSIs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "wsi_file",
        help="Path to the WSIs",
        type=str,
    )

    parser.add_argument(
        "parent_output_dir",
        help="Top directory in which the patches should be generated.",
        type=str,
    )

    parser.add_argument(
        "--patch-size",
        help="Length (in pixels) of the square patches",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--stride",
        help="The step to shift by when generating patches (sliding window).",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--min-mag",
        help="Minimum magnifcation to take patches at",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--max-mag",
        help="Maximum magnifcation to take patches at",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--num-workers",
        help="Number of process to spawn when cleaning up the patches.",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--generate-patches",
        help="Should we generate patches? If not we simply generate a thumbnail overview.",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--software",
        help="Which software to use for reading WSI and extracing patches.",
        type=str,
        default="OpenSlide",
        choices=["QuPath", "OpenSlide"],
    )

    parser.add_argument(
        "--zip-patches",
        help="Should each patch directory be zipped? You may generate huge"
        + " numbers of files if false.",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--patch-mask-threshold",
        help="Minimum overlap needed between patches and tissue mask, for patch to be kept.",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--wsi-magnification",
        help="Magnification of the image. Should be able to infer it from the WSI really.",
        type=float,
        default=None,
    )

    return parser.parse_args()


def _list_coordinate_columns() -> List[str]:
    """Return a list of the coordinate columns the user may specify.

    Returns
    -------
    List[str]
        The coordinate the user may optionally specify.

    """
    return ["left", "right", "top", "bottom"]


def _load_metadata_from_sources(sources: List[str]) -> DataFrame:
    """Load a data frame of scan metadata based on user-requested sources.

    Parameters
    ----------
    sources : List[str]
        List of training sources to use.

    Returns
    -------
    DataFrame
        Metadata for listed scans.

    """
    wsi_metadata_dir = Path("../wsi-metadata/").resolve()
    metadata_list = []
    for source in sources:
        csv_path = wsi_metadata_dir / f"{source}.csv"
        metadata_list.append(read_csv(csv_path))
    return concat(metadata_list).reset_index(drop=True)


def _check_test_or_train_col(speadsheet: DataFrame) -> None:
    """Check the `'test_or_train'` column has only allowed values.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.

    Raises
    ------
    ValueError
        If not all entries in the test or train column are in
        `["test", "train"]`.

    """
    if not speadsheet.test_or_train.isin(["test", "train"]).all():
        msg = "'test_or_train' col "
        msg += f"contains fields '{speadsheet.test_or_train.unique()}'. "
        msg += f"Should only contain {['test', 'train']}."
        raise ValueError(msg)


def _add_unspecified_coord_cols(spreadsheet: DataFrame) -> None:
    """Add coordinate columns of zero if they are unspecified.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.

    Raises
    ------
    ValueError
        If the coordinate calls are not all missing or all present.

    """
    cols = _list_coordinate_columns()
    all_specified = all(map(lambda x: x in spreadsheet, cols))
    none_specified = all(map(lambda x: x not in spreadsheet, cols))

    if not (all_specified or none_specified):
        msg = f"Either all of the coord fields '{cols}' should be specified, "
        msg += "or none."
        raise ValueError(msg)

    if none_specified:
        spreadsheet[cols] = 0


def _add_magnification_if_unspecified(spreadsheet: DataFrame) -> None:
    """Add the magnification column (value None) if unspecified.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.

    """
    if "magnification" not in spreadsheet:
        spreadsheet["magnification"] = None


def _add_optionally_unspecified_metadata(spreadsheet: DataFrame) -> None:
    """Add optionally unspecified columns to `spreadsheet`.

    Parameters
    ----------
    spreadsheet
        Scan-level metadata.

    Notes
    -----
    The user may have opted to omit the columns 'left', 'right', 'bottom' and
    'top', because in all cases they want to use the entire WSI and don't need
    to specifiy a region.

    They may also have ommited 'magnification', as they want qupath to
    determine it.

    """
    _add_unspecified_coord_cols(spreadsheet)
    _add_magnification_if_unspecified(spreadsheet)


def _add_width_and_height_to_metadata(spreadsheet: DataFrame) -> None:
    """Add the columns `'width'` and `'height'` to `spreadsheet`.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.

    """
    spreadsheet["width"] = spreadsheet["right"] - spreadsheet["left"]
    spreadsheet["height"] = spreadsheet["bottom"] - spreadsheet["top"]


def _coordinate_value_checks(spreadsheet: DataFrame):
    """Make sure either all or none of the coord entries are missing.

    Parameters
    ----------
    spreadsheet : DataFrame
        scan-level metadata

    Raises
    ------
    ValueError
        If any of the scans have

    """
    coord_cols = _list_coordinate_columns()

    all_nan = isnan(spreadsheet[coord_cols].to_numpy()).all(axis=1)
    none_nan = (~isnan(spreadsheet[coord_cols].to_numpy())).all(axis=1)

    all_or_none = maximum(all_nan, none_nan)

    if not all_or_none.all():
        msg = "Some scans have incomplete coordinate information. The fields"
        msg += f"{coord_cols} should all be specifed, or not included in "
        msg += "the source spreadsheets."
        raise ValueError(msg)


def _zero_coordinate_nans(spreadsheet: DataFrame) -> None:
    """Zero the nans in the coordinate columns of `spreadsheet`.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.

    """
    cols = _list_coordinate_columns()
    values = dict(zip(cols, len(cols) * [0]))
    spreadsheet.fillna(value=values, inplace=True)


def _set_coords_as_integers(spreadsheet: DataFrame) -> None:
    """Set the coordinate fields to integers.

    Parameters
    ----------
    spreadsheet : DataFrame
        scan-level metadata.

    """
    cols = _list_coordinate_columns()
    spreadsheet[cols] = spreadsheet[cols].astype(int)


def _replace_forbidden_case_id_characters(spreadsheet: DataFrame) -> None:
    """Replace forbidden characters in the 'case_id' column.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.

    """
    spreadsheet.case_id = spreadsheet.case_id.apply(
        lambda case_id: case_id.replace("/", "_")
    )


def _add_wsi_path_col(spreadsheet: DataFrame, wsi_dir: str) -> None:
    """Apply `Path.resolve` to the WSI paths.

    Parameters
    ----------
    spreadsheet : DataFrame
        Scan-level metadata.
    wsi_dir : str
        Path to the directory containing the WSIs.

    """
    spreadsheet["wsi_path"] = spreadsheet.apply(
        lambda x: Path(wsi_dir, x.source, x.scan),
        axis=1,
    )


def _prepare_scan_level_metadata(
    sources: List[str],
    wsi_dir: str,
) -> DataFrame:
    """Prepare a scan-level metadata data frame to extract patches with.

    Parameters
    ----------
    sources
        List of the scan sources (i.e. 'heeartlands', 'addenbrookes', etc.) to
        use.
    wsi_dir : str
        The root directory containing the WSIs.

    """
    metadata = _load_metadata_from_sources(sources)

    _check_test_or_train_col(metadata)
    _add_optionally_unspecified_metadata(metadata)
    _coordinate_value_checks(metadata)
    _zero_coordinate_nans(metadata)
    _set_coords_as_integers(metadata)
    _add_width_and_height_to_metadata(metadata)
    _replace_forbidden_case_id_characters(metadata)
    _add_wsi_path_col(metadata, wsi_dir)
    return metadata.sort_values(
        by=["source", "test_or_train", "label", "case_id", "scan"]
    )


def extract_patches_from_single_case_id(
    command_line_args: argparse.Namespace,
) -> None:
    """Extract patches from a single WSI.

    Parameters
    ----------
    command_line_args : argparse.Namespace
        The command line arguments.

    """
    extractor = PatchExtractor(
        patch_size=command_line_args.patch_size,
        stride=command_line_args.stride,
        mag_range=(command_line_args.min_mag, command_line_args.max_mag),
        top_dir=command_line_args.parent_output_dir,
        cleanup_workers=command_line_args.num_workers,
        zip_patches=command_line_args.zip_patches,
        patch_mask_threshold=command_line_args.patch_mask_threshold,
        software=command_line_args.software,
    )

    region_dict = {"left": 0, "top": 0, "width": 0, "height": 0}
    subdirs = Path(Path(command_line_args.wsi_file).stem)

    # TODO Check whether patches for an image of the same image already
    # exist and if so rename the image maybe?

    extractor(
        Path(command_line_args.wsi_file),
        subdirs,
        region=region_dict,
        user_supplied_mag=command_line_args.wsi_magnification,
        generate_patches=command_line_args.generate_patches,
    )


if __name__ == "__main__":
    command_line_arguments = parse_command_line_arguments()
    extract_patches_from_single_case_id(command_line_arguments)
