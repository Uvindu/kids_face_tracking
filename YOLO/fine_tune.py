
from ultralytics import YOLO
from pathlib import Path

# Configuration
MODEL_PATH = Path("/home/uvi/kids_face_recognition/models/yolov12l-face.pt")
DATASET_YAML_PATH = Path("/home/uvi/kids_face_recognition/YOLO/dataset.yaml")
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 352

def train_model():
    """
    Loads a pre-trained YOLO model and starts the fine-tuning process.
    """
    print("--- Starting Model Fine-Tuning ---")
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    if not DATASET_YAML_PATH.exists():
        print(f"[ERROR] Dataset configuration file not found at: {DATASET_YAML_PATH}")
        return

    # Load the pre-trained model.
    print(f"[INFO] Loading pre-trained model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Start the training process.
    print(f"[INFO] Starting training for {EPOCHS} epochs...")
    results = model.train(
        data=str(DATASET_YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name='kid_face_yolov12n_finetuned' # A custom name for this training run
    )

    print("\n--- Fine-Tuning Complete! ---")
    print("The trained model and results are saved in the 'runs/detect/kid_face_yolov12n_finetuned' directory.")

if __name__ == "__main__":
    train_model()
