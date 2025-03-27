import subprocess
import os
from pathlib import Path

def run_on_all_images(input_dir, output_dir, script_path, **kwargs):
    """
    Run the WSI patch extraction script on all images in a directory.

    Parameters:
        input_dir (str): Path to the directory containing WSI images.
        output_dir (str): Path to the directory where output will be saved.
        script_path (str): Path to the Python script to execute.
        **kwargs: Additional arguments to pass to the script using equals sign (e.g., stride=256).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    script_path = Path(script_path)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop through all files in input directory
    for image_file in input_dir.glob("*.tif"):  # Adjust extension if needed
        # Construct the command
        command = ["python3", str(script_path), str(image_file), str(output_dir)]
        
        # Add optional arguments with equals sign
        for key, value in kwargs.items():
            cli_key = key.replace("_", "-")  # Replace underscores with hyphens
            if isinstance(value, bool):
                # Boolean flags as --flag=true or --flag=false
                command.append(f"--{cli_key}={'true' if value else 'false'}")
            else:
                command.append(f"--{cli_key}={value}")

        print(f"Running command: {' '.join(command)}")
        
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Print output or handle errors
        if result.returncode == 0:
            print(f"Success: {image_file.name}")
        else:
            print(f"Error: {image_file.name}")
            print(result.stderr)


# Example Usage
run_on_all_images(
    input_dir="/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/aligned-wsis/post-ihc/",
    output_dir="/rds/user/mf774/hpc-work/part_II_project/in-house/create-he-ihc-patches/he-ihc-patches/post-ihc/",
    script_path="/rds/user/mf774/hpc-work/part_II_project/in-house/create-he-ihc-patches/patch-extractor/scripts/extract_patches_one_image.py",
    stride=256,
    min_mag=20.0,
    max_mag=20.0,
    software="QuPath"
)

