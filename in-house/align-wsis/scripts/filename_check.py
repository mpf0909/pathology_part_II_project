import os
import pandas as pd


def check_filenames(csv_path, directory_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Ensure the expected columns are present
    if "pre_IHC_file_name" not in df.columns or "post_IHC_file_name" not in df.columns:
        print(
            "Error: The CSV file must contain 'pre_IHC_file_name' and 'post_IHC_file_name' columns."
        )
        return

    # Get the list of files in the directory
    dir_files = set(os.listdir(directory_path))

    # Check for mismatches
    mismatches = []
    for index, row in df.iterrows():
        pre_file = row["pre_IHC_file_name"]
        post_file = row["post_IHC_file_name"]
        line_number = index + 2  # Account for header row

        if pre_file not in dir_files:
            mismatches.append((line_number, pre_file, "Missing pre_IHC_file"))
        if post_file not in dir_files:
            mismatches.append((line_number, post_file, "Missing post_IHC_file"))

    # Output results
    if mismatches:
        print("Mismatched or missing files:")
        for line_number, file, issue in mismatches:
            print(f"- Line {line_number}: {file} - {issue}")
    else:
        print("All file names match the directory.")


# Example usage:
check_filenames(
    "/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/wsi_alignment_mapping.csv", "/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/all-unaligned-wsis/"
)
