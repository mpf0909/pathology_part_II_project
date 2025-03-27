import os

def replace_spaces_in_filenames(directory):
    try:
        for filename in os.listdir(directory):
            old_path = os.path.join(directory, filename)
            if os.path.isfile(old_path) and " " in filename:
                new_filename = filename.replace(" ", "_")
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: "{filename}" -> "{new_filename}"')
        print("Filename replacement complete.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    directory = input("/rds/user/mf774/hpc-work/part_II_project/in-house/align-wsis/all-unaligned-wsis/").strip()
    replace_spaces_in_filenames(directory)
