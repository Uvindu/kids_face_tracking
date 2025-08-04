import os
import shutil

# --- Configuration ---

# IMPORTANT: Set the path to the folder containing the UTKFace dataset images.
# This folder should contain all the individual .jpg files.
# Example: "C:/Users/YourUser/Downloads/UTKFace"
SOURCE_IMAGE_DIR = '/home/uvi/kids_face_recognition/UTKFace'

# Set the path to the folder where you want to save the filtered images.
# The script will create this folder if it doesn't exist.
# Example: "C:/Users/YourUser/Documents/Filtered_Kid_Faces"
DESTINATION_DIR = 'filtered_malaysian_kids_0_to_12'

# Define the filtering criteria
MIN_AGE = 0
MAX_AGE = 12
# Race codes: 2 (Asian for Malay/Chinese), 3 (Indian)
RACE_CODES_TO_INCLUDE = [2, 3] 

# --- Main Script Logic ---

def filter_utkface_dataset():
    """
    First, cleans the source directory by removing non-JPEG files.
    Then, filters the UTKFace dataset based on age and race criteria and
    copies the matching images to a new directory.
    """
    # --- 1a. Initial Setup ---
    
    # Check if the source directory exists
    if not os.path.isdir(SOURCE_IMAGE_DIR):
        print(f"Error: Source directory not found at '{SOURCE_IMAGE_DIR}'")
        print("Please update the 'SOURCE_IMAGE_DIR' variable with the correct path.")
        return

    # Create the destination directory if it doesn't already exist
    if not os.path.exists(DESTINATION_DIR):
        print(f"Creating destination directory: '{DESTINATION_DIR}'")
        os.makedirs(DESTINATION_DIR)
    else:
        print(f"Destination directory '{DESTINATION_DIR}' already exists. Files will be added/overwritten.")

    # --- 1b. Cleanup Phase (New Addition) ---
    
    print("\n--- WARNING: Permanently removing non-image files from source directory ---")
    files_removed = 0
    
    # Iterate over a copy of the list of files to safely remove items
    for filename in os.listdir(SOURCE_IMAGE_DIR):
        # Check if the file is NOT a jpg or jpeg
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
            file_to_remove_path = os.path.join(SOURCE_IMAGE_DIR, filename)
            try:
                # This action is permanent.
                os.remove(file_to_remove_path)
                print(f"  - Removed: {filename}")
                files_removed += 1
            except OSError as e:
                print(f"  - Error removing file {filename}: {e}")

    print(f"Cleanup complete. Removed {files_removed} non-image file(s).")


    # --- 2. Filtering and Copying ---
    
    total_files_scanned = 0
    files_copied = 0
    
    print("\nStarting the filtering process...")
    
    # Get a fresh list of all files in the source directory after cleanup
    all_files = os.listdir(SOURCE_IMAGE_DIR)

    for filename in all_files:
        # This check is now slightly redundant but safe to keep
        if not filename.lower().endswith('.jpg'):
            continue
            
        total_files_scanned += 1
        
        try:
            # The filename format is [age]_[gender]_[race]_[date&time].jpg
            # We split the string by the underscore '_' delimiter
            parts = filename.split('_')
            
            # Extract age and race, then convert them to integers
            age = int(parts[0])
            race = int(parts[2])
            
            # --- 3. Apply Filtering Logic ---
            
            # Check if the image meets our criteria:
            # 1. Age must be between MIN_AGE and MAX_AGE (inclusive)
            # 2. Race code must be in our list of desired races
            if (MIN_AGE <= age <= MAX_AGE) and (race in RACE_CODES_TO_INCLUDE):
                
                # If criteria are met, construct the full source and destination paths
                source_path = os.path.join(SOURCE_IMAGE_DIR, filename)
                destination_path = os.path.join(DESTINATION_DIR, filename)
                
                # Copy the file to the new directory
                shutil.copy2(source_path, destination_path)
                files_copied += 1
                
                # Optional: Print progress for every 100 files copied
                if files_copied % 100 == 0:
                    print(f"  ...copied {files_copied} matching images so far.")

        except (IndexError, ValueError):
            # This handles cases where a filename does not match the expected format.
            print(f"  - Skipping file with unexpected name format: {filename}")
            continue

    # --- 4. Final Report ---
    
    print("\n---------------------------------")
    print("Filtering Complete!")
    print(f"Total images scanned after cleanup: {total_files_scanned}")
    print(f"Total matching images copied: {files_copied}")
    print(f"Filtered dataset saved in: '{os.path.abspath(DESTINATION_DIR)}'")
    print("---------------------------------")


# --- Run the script ---
if __name__ == '__main__':
    # Before running, ensure you have set the SOURCE_IMAGE_DIR variable above.
    filter_utkface_dataset()
