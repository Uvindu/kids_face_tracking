
import ultralytics
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
MODEL_PATH = repo_root / "kids_face_recognition" / "models" / "yolov12l-face.pt"
#MODEL_PATH = repo_root / "kids_face_recognition" / "YOLO" / "runs" / "detect" / "yolov12l_100ep_352imgsize" / "weights" / "best.pt"
#MODEL_PATH = repo_root / "kids_face_recognition" / "YOLO" / "runs" / "detect" / "yolov12l_10ep_352imgsize" / "weights" / "last.pt"

SOURCE_VIDEO = repo_root / "kids_face_recognition" / "YOLO" / "inference" / "input_video.mp4"
OUTPUT_PROJECT_DIR = repo_root / "kids_face_recognition" / "YOLO" / "inference"
OUTPUT_RUN_NAME = 'output'


if not MODEL_PATH.exists():
    print(f"Model file not found at '{MODEL_PATH}'.")
    print("Please ensure you have downloaded the correct model file and placed it in the correct directory.")
    exit()

if not SOURCE_VIDEO.exists():
    print(f"Input source video file do not exist.")

model = YOLO(str(MODEL_PATH))
print(f"Processing video: {SOURCE_VIDEO}")
print(f"Using model: {MODEL_PATH}")

CONFIDENCE_THRESHOLD= 0.4
IOU_THRESHOLD = 0.7
"""
Confidence Threshold (conf): You have this set to 0.2, which is quite low. This will find many potential faces but may also include a lot of false positives.
To reduce false alarms: Increase the confidence threshold (e.g., conf=0.4 or conf=0.5).
To find more, hard-to-detect faces: Keep the threshold low, but combine this with tracking (point #2) to filter out the noise.

IoU Threshold (iou): This threshold is used for Non-Maximum Suppression (NMS), which merges multiple overlapping boxes for the same object. The default is 0.7. If you find you're getting several boxes on a single face, you can lower this value.
"""
# Run tracking on the video
results = model.track(
    source=str(SOURCE_VIDEO),
    stream=True,
    save=True,
    project=str(OUTPUT_PROJECT_DIR),
    name=OUTPUT_RUN_NAME,
    exist_ok=True,
    conf=CONFIDENCE_THRESHOLD,
    iou=IOU_THRESHOLD,
    tracker="bytetrack.yaml"
)
for r in results:
    pass
        
print("---------------------------------")
print(f"Processing complete!")
print(f"Output video saved in: {OUTPUT_PROJECT_DIR / OUTPUT_RUN_NAME}")