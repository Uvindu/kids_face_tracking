import os
import cv2
import random
import shutil
from pathlib import Path

# --- Configuration ---

# Path to the directory containing your source image folders.
SOURCE_DATA_DIR = Path("/home/uvi/kids_face_recognition/filtered_datasets")

# Path where the final, structured YOLO dataset will be created.
# This is now inside your YOLO folder.
OUTPUT_DATA_DIR = Path("/home/uvi/kids_face_recognition/YOLO/yolo_dataset")

# Path to the face detection model files.
# --- UPDATED FILENAME ---
# The script now looks for 'deploy.prototxt' directly.
PROTOTXT_PATH = Path("/home/uvi/kids_face_recognition/filtered_datasets/face_model/deploy.prototxt")
MODEL_PATH = Path("/home/uvi/kids_face_recognition/filtered_datasets/face_model/res10_300x300_ssd_iter_140000.caffemodel")

# Defines the ratio for the validation set (e.g., 0.2 = 20%).
VALIDATION_SPLIT = 0.2

# The class ID for your object. It's 0 since you only have one class ('kid_face').
CLASS_ID = 0

# --- Main Script Logic ---

def create_yolo_dataset():
    """
    Main function to process images, create annotations, and structure the
    dataset for YOLO training.
    """
    print("--- Starting Dataset Preparation ---")

    # Load the pre-trained face detection model from disk
    print(f"[INFO] Loading face detector model...")
    try:
        face_detector = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(MODEL_PATH))
    except cv2.error as e:
        print(f"[ERROR] Could not load model. Please ensure the paths are correct and files are downloaded.")
        print(f"  - Prototxt Path: {PROTOTXT_PATH}")
        print(f"  - Model Path: {MODEL_PATH}")
        return

    # --- 1. Collect all image paths from the source directories ---
    all_image_paths = list(SOURCE_DATA_DIR.glob("**/*.jpg"))
    random.shuffle(all_image_paths) # Shuffle to ensure random distribution
    
    if not all_image_paths:
        print(f"[ERROR] No images found in {SOURCE_DATA_DIR}. Please check the path.")
        return

    print(f"[INFO] Found {len(all_image_paths)} total images.")

    # --- 2. Create the output directory structure ---
    print(f"[INFO] Creating output directory structure at: {OUTPUT_DATA_DIR}")
    path_images_train = OUTPUT_DATA_DIR / "images/train"
    path_images_val = OUTPUT_DATA_DIR / "images/val"
    path_labels_train = OUTPUT_DATA_DIR / "labels/train"
    path_labels_val = OUTPUT_DATA_DIR / "labels/val"

    for p in [path_images_train, path_images_val, path_labels_train, path_labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    # --- 3. Split the dataset ---
    split_index = int(len(all_image_paths) * (1 - VALIDATION_SPLIT))
    train_paths = all_image_paths[:split_index]
    val_paths = all_image_paths[split_index:]

    print(f"[INFO] Splitting dataset: {len(train_paths)} training images, {len(val_paths)} validation images.")

    # --- 4. Process images and create annotations ---
    # This list will store details of all skipped files for the final report.
    skipped_files_log = []
    
    process_split(train_paths, path_images_train, path_labels_train, face_detector, "Training", skipped_files_log)
    process_split(val_paths, path_images_val, path_labels_val, face_detector, "Validation", skipped_files_log)

    print("\n--- Dataset Preparation Complete! ---")
    print(f"Your YOLO-ready dataset is located at: {OUTPUT_DATA_DIR}")

    # --- 5. Print a detailed report of all skipped files ---
    if skipped_files_log:
        print("\n--- Skipped Files Report ---")
        print(f"A total of {len(skipped_files_log)} images were skipped.")
        for file_path, reason in skipped_files_log:
            print(f"  - File: {file_path}\n    Reason: {reason}")
    else:
        print("\n[INFO] No images were skipped during processing. Great!")


def process_split(image_paths, img_output_dir, label_output_dir, face_detector, split_name, skipped_files_log):
    """
    Processes a list of images for a given split (train/val).
    Detects faces, creates YOLO annotations, and saves the files.
    Logs any skipped files to the provided log list.
    """
    print(f"\n[INFO] Processing {split_name} set...")
    images_processed = 0
    images_skipped_in_split = 0

    for img_path in image_paths:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                reason = "Could not read image file (file might be corrupted or not a valid image)."
                print(f"  [WARNING] Skipping {img_path.name}: {reason}")
                skipped_files_log.append((str(img_path), reason))
                images_skipped_in_split += 1
                continue

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_detector.setInput(blob)
            detections = face_detector.forward()

            best_detection_idx = detections[0, 0, :, 2].argmax()
            confidence = detections[0, 0, best_detection_idx, 2]

            if confidence > 0.5:
                box = detections[0, 0, best_detection_idx, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                yolo_x_center = (startX + (endX - startX) / 2) / w
                yolo_y_center = (startY + (endY - startY) / 2) / h
                yolo_width = (endX - startX) / w
                yolo_height = (endY - startY) / h

                yolo_annotation = f"{CLASS_ID} {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_width:.6f} {yolo_height:.6f}"

                base_filename = img_path.stem
                output_img_path = img_output_dir / f"{base_filename}.jpg"
                output_label_path = label_output_dir / f"{base_filename}.txt"

                cv2.imwrite(str(output_img_path), image)
                with open(output_label_path, "w") as f:
                    f.write(yolo_annotation)
                
                images_processed += 1
            else:
                reason = f"Low face detection confidence: {confidence:.2f} (threshold > 0.5)"
                # This warning is optional as it will be in the final report
                # print(f"  [INFO] Skipping {img_path.name}: {reason}")
                skipped_files_log.append((str(img_path), reason))
                images_skipped_in_split += 1

        except Exception as e:
            reason = f"An unexpected error occurred during processing: {e}"
            print(f"  [ERROR] Skipping {img_path.name}: {reason}")
            skipped_files_log.append((str(img_path), reason))
            images_skipped_in_split += 1
            
    print(f"[INFO] {split_name} processing finished. {images_processed} images processed. {images_skipped_in_split} images skipped.")

if __name__ == "__main__":
    create_yolo_dataset()
