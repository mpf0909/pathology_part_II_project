#!/usr/bin/env python
"""Summarise the case counts we in `"../wsi_metadata/"`."""
from pathlib import Path

from pandas import DataFrame, read_csv, concat  # type: ignore


def _load_scan_metadata() -> DataFrame:
    """Load the metadata for each scan source.

    Returns
    -------
    DataFrame
        The scan-level metadata for every source.

    """
    metadata_dir = Path(__file__).resolve().parent.parent / "wsi-metadata/"
    files = list(metadata_dir.glob("*.csv"))
    return concat(map(read_csv, files), axis=0, ignore_index=True)


df = _load_scan_metadata()

keys = ["source", "label", "test_or_train"]
source_label = df.groupby(keys).case_id.nunique()

print("Summary by cases")
print(79 * "-")
print("Total cases per source and set")
print(source_label.groupby(level=(0, 2)).sum())

print("\n\n")
print("Total cases per set")
print(source_label.groupby(level=2).sum())


print("\n\n")
print("Full case summary")
print(source_label)


print("\n\n\n\n\n")
print("Summary by scans")
print(79 * "-")


keys = ["source", "label", "file_format", "test_or_train"]
src_label_format = df.groupby(keys).scan.count()


print("By source and split")
print(src_label_format.groupby(level=(0, 3)).sum())

print("\n\n")
print("By source, label and split")
print(src_label_format.groupby(level=(0, 1, 3)).sum())

# print(src_label_format.sum(level=(0, 3)))


print("\n\n")
print("Full summary")
print(src_label_format)
# print(f"Total scans = {src_label_format.sum()}")
