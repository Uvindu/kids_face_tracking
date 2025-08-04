
import os
import shutil


from pathlib import Path
# Configuration
repo_root = Path(__file__).resolve().parents[2]
SOURCE_IMAGE_DIR = repo_root / "kids_face_recognition" / "raw_datasets" / "UTKFace"
DESTINATION_DIR = repo_root / "kids_face_recognition" / "filtered_datasets" / "filtered_malaysian_kids_0_to_12"
MIN_AGE = 0
MAX_AGE = 12
RACE_CODES_TO_INCLUDE = [2, 3]  # 2: Asian (Malay/Chinese), 3: Indian

def filter_utkface_dataset():
    """
    Cleans the source directory by removing non-JPEG files, then filters and copies images by age and race.
    """

    if not SOURCE_IMAGE_DIR.is_dir():
        print(f"Error: Source directory not found at '{SOURCE_IMAGE_DIR}'")
        print("Please update the 'SOURCE_IMAGE_DIR' variable with the correct path.")
        return

    if not DESTINATION_DIR.exists():
        print(f"Creating destination directory: '{DESTINATION_DIR}'")
        DESTINATION_DIR.mkdir(parents=True)
    else:
        print(f"Destination directory '{DESTINATION_DIR}' already exists. Files will be added/overwritten.")

    print("\n--- WARNING: Permanently removing non-image files from source directory ---")
    files_removed = 0
    for filename in os.listdir(SOURCE_IMAGE_DIR):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
            file_to_remove_path = SOURCE_IMAGE_DIR / filename
            try:
                file_to_remove_path.unlink()
                print(f"  - Removed: {filename}")
                files_removed += 1
            except OSError as e:
                print(f"  - Error removing file {filename}: {e}")
    print(f"Cleanup complete. Removed {files_removed} non-image file(s).")

    total_files_scanned = 0
    files_copied = 0
    print("\nStarting the filtering process...")
    all_files = os.listdir(SOURCE_IMAGE_DIR)
    for filename in all_files:
        if not filename.lower().endswith('.jpg'):
            continue
        total_files_scanned += 1
        try:
            parts = filename.split('_')
            age = int(parts[0])
            race = int(parts[2])
            if (MIN_AGE <= age <= MAX_AGE) and (race in RACE_CODES_TO_INCLUDE):
                source_path = SOURCE_IMAGE_DIR / filename
                destination_path = DESTINATION_DIR / filename
                shutil.copy2(str(source_path), str(destination_path))
                files_copied += 1
                if files_copied % 100 == 0:
                    print(f"  ...copied {files_copied} matching images so far.")
        except (IndexError, ValueError):
            print(f"  - Skipping file with unexpected name format: {filename}")
            continue

    # --- 4. Final Report ---
    
    print("\n---------------------------------")
    print("Filtering Complete!")
    print(f"Total images scanned after cleanup: {total_files_scanned}")
    print(f"Total matching images copied: {files_copied}")
    print(f"Filtered dataset saved in: '{DESTINATION_DIR.resolve()}'")
    print("---------------------------------")


# --- Run the script ---
if __name__ == '__main__':
    # Before running, ensure you have set the SOURCE_IMAGE_DIR variable above.
    filter_utkface_dataset()
