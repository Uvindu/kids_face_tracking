import os
import deeplake
from PIL import Image
from tqdm import tqdm

def save_images_from_deeplake(save_path):
    """
    Loads the FGNET dataset, filters for ages 0-12, and saves 
    the images to a specified directory. This version is compatible
    with deeplake version 3.x and includes error handling for corrupt data.

    Args:
        save_path (str): The absolute path to the directory where the 
                         images will be saved.
    """
    print("Attempting to load FGNET dataset from Activeloop Hub...")
    
    try:
        print(f"Deep Lake version: {deeplake.__version__}")

        # Using deeplake.load with read_only=True, which is the
        # correct syntax for deeplake v3.x.
        ds = deeplake.load("hub://activeloop/fgnet", read_only=True)

        subject_id_names = ds.image_ids.info['class_names']
        age_names = ds.ages.info['class_names']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")

        print(f"Dataset loaded successfully. Found {len(ds)} images.")
        print("Filtering for ages 0-12...")
        print(f"Images will be saved in: {os.path.abspath(save_path)}")

        # --- Iterate through the dataset and save each image ---
        saved_count = 0
        skipped_count = 0
        filtered_out_count = 0
        for i, sample in enumerate(tqdm(ds, desc="Processing images")):
            try:
                age_index = sample['ages'].numpy()[0]
                age = int(age_names[age_index])

                # --- Filter by age ---
                if 0 <= age <= 12:
                    image_array = sample['images'].numpy()
                    
                    # 1. Get the subject ID string via index lookup
                    subject_id_index = sample['image_ids'].numpy()[0]
                    subject_id = subject_id_names[subject_id_index]
                    
                    # 2. Format the filename like the original FGNET dataset: {subject}A{age}.jpg
                    age_formatted = f"{age:02d}"
                    
                    # 3. To prevent overwriting files (e.g. multiple photos at the same age),
                    #    we append the loop index to make every filename unique.
                    filename = f"{subject_id}A{age_formatted}_{i}.jpg"
                    
                    # --- Save the image ---
                    
                    img = Image.fromarray(image_array)
                    output_filepath = os.path.join(save_path, filename)
                    img.save(output_filepath, 'JPEG')
                    saved_count += 1
                else:
                    # This age is outside our desired range
                    filtered_out_count += 1

            except Exception as e:
                # This error handling will catch problematic samples in the dataset
                # and allow the script to continue.
                print(f"\nWarning: Could not process sample {i}. Error: {e}. Skipping.")
                skipped_count += 1

        print(f"\nProcessing complete.")
        print(f"Total images saved (ages 0-12): {saved_count}")
        print(f"Total images filtered out (age > 12): {filtered_out_count}")
        if skipped_count > 0:
            print(f"Total images skipped due to data errors: {skipped_count}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("\nIf this fails, the issue is likely with the specific version of 'deeplake' installed.")
        print("You could try forcing a version 3 install with: pip install \"deeplake<4\"")


if __name__ == "__main__":
    # The output directory is set to your specified path.
    output_directory = "/home/uvi/kids_face_recognition/filtered_datasets/FGNET_age_0_12"
    save_images_from_deeplake(save_path=output_directory)
