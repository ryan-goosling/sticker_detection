import os
import random
import cv2
import matplotlib.pyplot as plt

# --- CONFIG ---
CONFIG = {
    "dataset_dir": "dataset",
    "subset": "train",  # or 'val', 'test'
    "num_samples": 20,
    "class_names": ["happy", "sad", "dead"]
}

image_dir = os.path.join(CONFIG["dataset_dir"], "images", CONFIG["subset"])
label_dir = os.path.join(CONFIG["dataset_dir"], "labels", CONFIG["subset"])

image_files = sorted(os.listdir(image_dir))
random.shuffle(image_files)
samples = image_files[:CONFIG["num_samples"]]

output_dir = os.path.join(CONFIG["dataset_dir"], "visualized", CONFIG["subset"])
os.makedirs(output_dir, exist_ok=True)

for filename in samples:
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls_id, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                class_id = int(cls_id)
                class_name = CONFIG["class_names"][class_id]
                color = [(255, 0, 0), (0, 255, 0), (0, 128, 255)][class_id % 3]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness=1, lineType=cv2.LINE_AA)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
