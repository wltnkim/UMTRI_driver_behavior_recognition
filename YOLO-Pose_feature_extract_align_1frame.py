import os
import cv2
import re
import numpy as np
import pickle
import hashlib
from tqdm import tqdm
from ultralytics import YOLO

# YOLO-Pose 모델 불러오기
model = YOLO("yolov8x-pose.pt")

# 저장 디렉토리
save_dir = "saved_frames_yolopose_align_1frame"
os.makedirs(save_dir, exist_ok=True)

# 완료된 경로를 저장할 로그 파일
completed_log_path = "completed_paths_frames_yolopose_align_1frame.txt"
if os.path.exists(completed_log_path):
    with open(completed_log_path, "r") as f:
        completed_paths = set(line.strip() for line in f.readlines())
else:
    completed_paths = set()

# YOLO-Pose 기반 2D 포즈 추출 함수
def extract_pose_from_image(img_path):
    results = model(img_path, verbose=False)
    if len(results[0].keypoints.xy) == 0:
        return np.zeros((17, 2), dtype=np.float32).flatten()  # 34차원 (x, y)
    else:
        return results[0].keypoints.xy[0].cpu().numpy().flatten()

# 폴더명에서 behavior class 코드 추출
def extract_behavior_class(folder_name):
    match = re.match(r"OBC_H\d{3}_(.+?)_\d{3}_", folder_name)
    if match:
        return match.group(1)
    return None

# 루트 디렉토리
root_dir = "/home/superman/data/jkim/work/datasets/OBC_STRUCTURE_COPY"
label_map = {}
label_counter = 0

# label_map 불러오기
label_map_path = "pose_label_map_frames_yolopose_align_1frame.pkl"
if os.path.exists(label_map_path):
    with open(label_map_path, "rb") as f:
        label_map = pickle.load(f)
    label_counter = len(label_map)

# 데이터 처리
for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
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

        try:
            frame_files = sorted([f for f in filenames if f.lower().endswith(".jpeg")])
            dir_hash = hashlib.md5(dirpath.encode()).hexdigest()[:12]  # 결정적 파일 이름

            for idx, frame_file in enumerate(frame_files):
                img_path = os.path.join(dirpath, frame_file)
                pose = extract_pose_from_image(img_path)
                save_path = os.path.join(save_dir, f"{dir_hash}_frame{idx}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump((pose, label), f)

            with open(completed_log_path, "a") as f:
                f.write(dirpath + "\n")

        except Exception as e:
            print(f"❌ 오류 발생: {dirpath} → {e}")

# label map 저장
with open(label_map_path, "wb") as f:
    pickle.dump(label_map, f)

print("✅ YOLO-Pose 기반 프레임 단위 저장 완료!")
