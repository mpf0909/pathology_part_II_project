"""plotting function."""

import os
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # type: ignore
import openslide


# pylint: disable=too-many-locals


def plot_aligned_images(
    image_path_1: Path, image_path_2: Path, save_folder: Path
) -> None:
    """Plot 10 random corresponding patches to visualise how well the images are aligned.

    Parameters
    ----------
    image_path_1 : Path
        First WSI path.
    image_path_2 : Path
        Second WSI path.
    save_folder : Path
        Folder to save plots in.

    """
    # TODO check that both images exist

    if not os.path.exists(save_folder):
        # Create the folder
        os.makedirs(save_folder)

    # Load the aligned images
    slide1 = openslide.OpenSlide(image_path_1)
    slide2 = openslide.OpenSlide(image_path_2)

    # Function to check if a patch has a white background
    def is_non_white_patch(patch, threshold=200):
        # Convert RGBA to grayscale approximation
        patch_array = np.array(patch)[:, :, :3]
        gray_patch = np.mean(patch_array[..., :3], axis=-1)
        # Check if the mean intensity of all pixels is below the threshold (not white)
        return np.mean(gray_patch) < threshold

    # Extract random patches
    def get_random_patch(image1, image2, patch_size, level=0):
        width, height = image1.level_dimensions[0]
        while True:
            # Randomly select the top-left corner
            top = random.randint(0, height - patch_size)
            left = random.randint(0, width - patch_size)
            patch1 = image1.read_region(
                location=(left, top), level=level, size=(patch_size, patch_size)
            )
            patch2 = image2.read_region(
                location=(left, top), level=level, size=(patch_size, patch_size)
            )
            if is_non_white_patch(patch1, threshold=230) and is_non_white_patch(
                patch2, threshold=230
            ):
                # patch1.save(f"patches_plotting/patch_x_{left}_y_{top}.png", format="PNG")
                return patch1, patch2

    # Parameters
    num_rows = 5  # Number of rows in the plot
    patch_size = 512  # Size of each patch

    # Create the figure
    fig, axes = plt.subplots(num_rows, 4, figsize=(10, num_rows))
    fig.suptitle("Random Patches from Aligned Images", fontsize=16)

    for i in range(2 * num_rows):
        patch_size = 512 if i < num_rows else 64
        # Get a random patch from both images
        patch1, patch2 = get_random_patch(slide1, slide2, patch_size)

        col = 2 if i >= 5 else 0

        # Display patches in the plot
        patch_rgb = Image.alpha_composite(
            Image.new("RGBA", patch1.size, "white"), patch1
        ).convert("RGB")
        axes[i % 5, 0 + col].imshow(patch_rgb)
        axes[i % 5, 0 + col].axis("off")
        axes[i % 5, 0 + col].set_title(f"Image 1 Patch {i+1}")

        patch_rgb = Image.alpha_composite(
            Image.new("RGBA", patch2.size, "white"), patch2
        ).convert("RGB")
        axes[i % 5, 1 + col].imshow(patch_rgb)
        axes[i % 5, 1 + col].axis("off")
        axes[i % 5, 1 + col].set_title(f"Image 2 Patch {i+1}")

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(
        f"{save_folder}/random_patches_comparison_{os.path.basename(image_path_1)[:-4]}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def print_new_image_corner_patches(
    image_path: Path,
    save_folder: Path,
):
    """Plot the four corners of a WSI.

    Parameters
    ----------
    image_path : Path
        _description_, by default "./transformed_image_affine_output.tif"
    save_folder : Path, optional
        _description_, by default "corner_patches.png"
    """
    slide = openslide.OpenSlide(image_path)

    width, height = slide.level_dimensions[0]

    patch_size = 1000

    patch_coordinates = [
        (0, 0),
        (width - patch_size, 0),
        (0, height - patch_size),
        (width - patch_size, height - patch_size),
    ]
    patch_names = ["top left", "top right", "bottom left", "bottom right"]
    plt_coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    fig.suptitle("Corner Figures", fontsize=16)

    for idx in range(4):
        left, top = patch_coordinates[idx]
        level = 0
        patch = slide.read_region(
            location=(left, top), level=level, size=(patch_size, patch_size)
        )

        # Display patches in the plot
        patch_rgb = Image.alpha_composite(
            Image.new("RGBA", patch.size, "white"), patch
        ).convert("RGB")
        plt1, plt2 = plt_coordinates[idx]
        axes[plt1, plt2].imshow(patch_rgb)
        axes[plt1, plt2].axis("off")
        axes[plt1, plt2].set_title(patch_names[idx])

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(
        f"{save_folder}/corners_comparison{os.path.basename(image_path)[:-4]}.png", dpi=300, bbox_inches="tight"
    )
    # plt.show()


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Overlay two images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "wsi_file_one",
        help="Path to the first WSIs",
        type=str,
    )

    parser.add_argument(
        "wsi_file_two",
        help="Path to the second WSIs",
        type=str,
    )

    parser.add_argument(
        "--save-folder",
        help="Path to save the aligned first WSIs",
        type=str,
        default="./plots/",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    plot_aligned_images(
        image_path_1=args.wsi_file_one,
        image_path_2=args.wsi_file_two,
        save_folder=args.save_folder,
    )
    print_new_image_corner_patches(image_path=args.wsi_file_one,save_folder=args.save_folder)
    print_new_image_corner_patches(image_path=args.wsi_file_two,save_folder=args.save_folder)
