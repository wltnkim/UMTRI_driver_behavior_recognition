import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
import hashlib
from ultralytics import YOLO
from models_vit import VisionTransformer
from tqdm import tqdm
import re

# --- Settings ---
input_root_dir = "/home/superman/data/jkim/work/datasets/OBC_STRUCTURE_COPY"
save_dir = "saved_frames_au_features"
os.makedirs(save_dir, exist_ok=True)

completed_log_path = "completed_paths_au_features.txt"
if os.path.exists(completed_log_path):
    with open(completed_log_path, "r") as f:
        completed_paths = set(line.strip() for line in f.readlines())
else:
    completed_paths = set()

au_names = [f"AU{i}" for i in [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]]

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AU Model
model = VisionTransformer(img_size=224, patch_size=16, num_classes=12,
                          embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True)
ckpt = torch.load('checkpoints/FMAE_ViT_base.pth', map_location='cpu')
model.load_state_dict(ckpt, strict=False)
model.eval()
model.to(device)

# YOLOv8-face model
face_detector = YOLO("yolov8x-face-lindevs.pt")  # Path to the pre-trained face model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def extract_behavior_class(folder_name):
    match = re.match(r"OBC_H\d{3}_(.+?)_\d{3}_", folder_name)
    if match:
        return match.group(1)
    return None

label_map = {}
label_counter = 0
label_map_path = "au_label_map.pkl"
if os.path.exists(label_map_path):
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
    label_counter = len(label_map)

# --- Start Processing ---
for dirpath, _, filenames in tqdm(os.walk(input_root_dir)):
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

        if class_code not in label_map:
            label_map[class_code] = label_counter
            label_counter += 1
        label = label_map[class_code]

        frame_files = sorted([f for f in filenames if f.lower().endswith(".jpeg")])
        dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]

        for idx, frame_file in enumerate(frame_files):
            try:
                img_path = os.path.join(dirpath, frame_file)
                frame = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_detector(img_rgb, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                au_vector = np.zeros(12, dtype=np.float32)
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
                    face_crop = img_rgb[y1:y2, x1:x2]

                    input_tensor = transform(face_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        scores = torch.sigmoid(output).squeeze().cpu().numpy()
                        au_vector = scores.astype(np.float32)

                save_path = os.path.join(save_dir, f"{dir_hash}_frame{idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump((au_vector, label), f)

            except Exception as e:
                print(f"❌ Error: {img_path} → {e}")

        with open(completed_log_path, "a") as f:
            f.write(dirpath + "\n")

# --- Save Label Map ---
with open(label_map_path, "wb") as f:
    pickle.dump(label_map, f)

print("✅ AU feature saving complete (based on YOLOv8-face)")
