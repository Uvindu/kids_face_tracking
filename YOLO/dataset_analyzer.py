import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---

# Set the path to your source dataset containing the original image folders.
# This should be the same source directory as your data preparation script.
SOURCE_DATA_DIR = Path("/home/uvi/kids_face_recognition/filtered_datasets")

# --- Main Analysis Logic ---

def analyze_dataset_dimensions():
    """
    Analyzes all images in the source directory to find their
    average and median dimensions.
    """
    print("--- Starting Dataset Size Analysis ---")

    # --- 1. Collect all image paths from the source directories ---
    all_image_paths = list(SOURCE_DATA_DIR.glob("**/*.jpg"))
    
    if not all_image_paths:
        print(f"[ERROR] No images found in {SOURCE_DATA_DIR}. Please check the path.")
        return

    print(f"[INFO] Found {len(all_image_paths)} total images to analyze.")

    # --- 2. Iterate through images and collect their dimensions ---
    widths = []
    heights = []

    # Using tqdm for a nice progress bar
    for img_path in tqdm(all_image_paths, desc="Analyzing images"):
        try:
            # Read image headers to get dimensions without loading the full image
            # This is faster than cv2.imread()
            image = cv2.imread(str(img_path))
            if image is not None:
                h, w = image.shape[:2]
                widths.append(w)
                heights.append(h)
        except Exception as e:
            print(f"\n[WARNING] Could not process {img_path}. Error: {e}")

    if not widths or not heights:
        print("[ERROR] Could not read dimensions from any images.")
        return

    # --- 3. Calculate statistics ---
    avg_width = int(np.mean(widths))
    avg_height = int(np.mean(heights))
    
    median_width = int(np.median(widths))
    median_height = int(np.median(heights))
    
    min_width = int(np.min(widths))
    min_height = int(np.min(heights))

    max_width = int(np.max(widths))
    max_height = int(np.max(heights))


    # --- 4. Print the final report ---
    print("\n--- Dataset Size Analysis Report ---")
    print(f"Total images analyzed: {len(widths)}")
    print("\n--- Averages ---")
    print(f"Average Dimensions: (Width: {avg_width} px, Height: {avg_height} px)")
    
    print("\n--- Medians (often better for skewed data) ---")
    print(f"Median Dimensions:  (Width: {median_width} px, Height: {median_height} px)")

    print("\n--- Extremes ---")
    print(f"Min Dimensions:     (Width: {min_width} px, Height: {min_height} px)")
    print(f"Max Dimensions:     (Width: {max_width} px, Height: {max_height} px)")

    print("\n--- Recommendation ---")
    print("Based on this analysis, consider setting the `IMAGE_SIZE` in your training script")
    print(f"to a value close to the median or average, such as {median_width} or {avg_width}.")
    print("Remember to choose a value that is a multiple of 32 for optimal performance.")


if __name__ == "__main__":
    # Ensure you have numpy and tqdm installed:
    # pip3 install numpy tqdm
    analyze_dataset_dimensions()
