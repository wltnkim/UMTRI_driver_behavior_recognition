import os
import cv2
import re
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
from ultralytics import YOLO

# Load the YOLO-Pose model
model = YOLO("yolov8x-pose.pt")

# Directory to save the output
save_dir = "saved_frames_yolopose_align_1frame"
os.makedirs(save_dir, exist_ok=True)

# Log file to store paths of completed directories
completed_log_path = "completed_paths_frames_yolopose_align_1frame.txt"
if os.path.exists(completed_log_path):
    with open(completed_log_path, "r") as f:
        completed_paths = set(line.strip() for line in f.readlines())
else:
    completed_paths = set()

# Function to extract 2D pose from an image using YOLO-Pose
def extract_pose_from_image(img_path):
    """
    Extracts 2D pose keypoints from a single image file.
    Returns a flattened numpy array of (x, y) coordinates for 17 keypoints.
    """
    results = model(img_path, verbose=False)
    if len(results[0].keypoints.xy) == 0:
        # Return a zero vector if no keypoints are detected
        return np.zeros((17, 2), dtype=np.float32).flatten()  # 34 dimensions (x, y)
    else:
        # Return the detected keypoints
        return results[0].keypoints.xy[0].cpu().numpy().flatten()

# Function to extract the behavior class code from a folder name
def extract_behavior_class(folder_name):
    """
    Parses the folder name to find the behavior class code using regex.
    Example: "OBC_H001_WALKING_001_" -> "WALKING"
    """
    match = re.match(r"OBC_H\d{3}_(.+?)_\d{3}_", folder_name)
    if match:
        return match.group(1)
    return None

# Root directory for the dataset
root_dir = "/home/superman/data/jkim/work/datasets/OBC_STRUCTURE_COPY"
label_map = {}
label_counter = 0

# Load the label map if it exists
label_map_path = "pose_label_map_frames_yolopose_align_1frame.pkl"
if os.path.exists(label_map_path):
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
    label_counter = len(label_map)

# Process the data
for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
    # Check if the directory contains color images
    if "Color" in dirpath and any(fname.lower().endswith(".jpeg") for fname in filenames):
        if dirpath in completed_paths:
            continue

        parts = dirpath.split(os.sep)
        folder_name = next((p for p in parts if p.startswith("OBC_H")), None)
        if not folder_name:
            continue

        class_code = extract_behavior_class(folder_name)
        if not class_code:
            continue

        # Assign a numerical label to the class if it's new
        if class_code not in label_map:
            label_map[class_code] = label_counter
            label_counter += 1
        label = label_map[class_code]

        try:
            frame_files = sorted([f for f in filenames if f.lower().endswith(".jpeg")])
            # Create a deterministic file name using a hash of the directory path
            dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]

            for idx, frame_file in enumerate(frame_files):
                img_path = os.path.join(dirpath, frame_file)
                pose = extract_pose_from_image(img_path)
                save_path = os.path.join(save_dir, f"{dir_hash}_frame{idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump((pose, label), f)

            # Log the completed directory path
            with open(completed_log_path, "a") as f:
                f.write(dirpath + "\n")

        except Exception as e:
            print(f"❌ Error processing directory: {dirpath} → {e}")

# Save the final label map
with open(label_map_path, "wb") as f:
    pickle.dump(label_map, f)

print("✅ Frame-by-frame data saving based on YOLO-Pose is complete!")
