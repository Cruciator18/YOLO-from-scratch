import os
import shutil
import re
import yaml
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from ultralytics import YOLO

SOURCE_ROOT = "/content/drive/MyDrive/VOC2005_2"
DEST_ROOT = "/content/dataset_yolo"
MODEL_CONFIG_NAME = "custom_yolo.yaml"

CLASS_MAP = {
    'bicycle': 0, 'bike': 0, 'velo': 0,
    'car': 1, 'voiture': 1,
    'motorbike': 2, 'motocyclette': 2,
    'pedestrian': 3, 'person': 3, 'pieton': 3
}

def create_custom_model_config():
    yolo_config = {
        'nc': 4,
        'scales': {
            'n': [0.50, 0.25, 1024]
        },
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [-1, 1, 'Conv', [256, 3, 1]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [-1, 1, 'Conv', [512, 3, 1]],
            [-1, 1, 'Conv', [1024, 3, 2]],
            [-1, 1, 'Conv', [1024, 3, 1]],
            [-1, 1, 'SPPF', [1024, 5]],
            [-1, 1, 'C2PSA', [1024]],
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],
            [-1, 1, 'C3k2', [512, False, 0.25]],

            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],
            [-1, 1, 'C3k2', [256, False, 0.25]],

            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 13], 1, 'Concat', [1]],
            [-1, 1, 'C3k2', [512, False, 0.25]],

            [-1, 1, 'Conv', [512, 3, 2]],
            [[-1, 10], 1, 'Concat', [1]],
            [-1, 1, 'C3k2', [1024, True, 0.25]],

            [[16, 19, 22], 1, 'Detect', ['nc']]
        ]
    }

    with open(MODEL_CONFIG_NAME, 'w') as f:
        yaml.dump(yolo_config, f, sort_keys=False)

    print(f"Fixed Architecture Saved: {MODEL_CONFIG_NAME}")

def convert_to_yolo():
    if os.path.exists(f"{DEST_ROOT}/data.yaml"):
        print("Data exists, skipping conversion.")
        return

    if os.path.exists(DEST_ROOT): shutil.rmtree(DEST_ROOT)
    for split in ['train', 'val']:
        os.makedirs(f"{DEST_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DEST_ROOT}/labels/{split}", exist_ok=True)

    ann_dir = os.path.join(SOURCE_ROOT, "Annotations")
    img_dir = os.path.join(SOURCE_ROOT, "PNGImages")
    samples = []

    print(f"Scanning: {SOURCE_ROOT}")
    if not os.path.exists(ann_dir): return

    for class_folder in os.listdir(ann_dir):
        if class_folder in CLASS_MAP:
            cls_id = CLASS_MAP[class_folder]
            folder_path = os.path.join(ann_dir, class_folder)
            if not os.path.isdir(folder_path): continue
            for txt_file in os.listdir(folder_path):
                if txt_file.endswith(".txt"):
                    base_name = os.path.splitext(txt_file)[0]
                    img_name = base_name + ".png"
                    img_src = os.path.join(img_dir, class_folder, img_name)
                    txt_src = os.path.join(folder_path, txt_file)
                    if os.path.exists(img_src): samples.append((img_src, txt_src, cls_id))

    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    def process_samples(sample_list, split):
        for img_path, txt_path, cls_id in tqdm(sample_list, desc=f"Converting {split}"):
            try:
                with Image.open(img_path) as img: w, h = img.size
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                boxes = re.findall(r':\s*\(\s*(\d+),\s*(\d+)\s*\)\s*-\s*\(\s*(\d+),\s*(\d+)\s*\)', content)
                yolo_data = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = map(float, box)
                    xmin = max(0, min(xmin, w)); ymin = max(0, min(ymin, h))
                    xmax = max(0, min(xmax, w)); ymax = max(0, min(ymax, h))
                    if xmax <= xmin or ymax <= ymin: continue
                    x_c = ((xmin + xmax) / 2) / w; y_c = ((ymin + ymax) / 2) / h
                    bw = (xmax - xmin) / w; bh = (ymax - ymin) / h
                    yolo_data.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
                if yolo_data:
                    shutil.copy(img_path, f"{DEST_ROOT}/images/{split}/{os.path.basename(img_path)}")
                    label_name = os.path.splitext(os.path.basename(txt_path))[0] + ".txt"
                    with open(f"{DEST_ROOT}/labels/{split}/{label_name}", 'w') as f: f.write("\n".join(yolo_data))
            except Exception: continue

    process_samples(train_samples, 'train')
    process_samples(val_samples, 'val')

    data_yaml = {'path': DEST_ROOT, 'train': 'images/train', 'val': 'images/val', 'nc': 4, 'names': ['bicycle', 'car', 'motorbike', 'pedestrian']}
    with open(f"{DEST_ROOT}/data.yaml", 'w') as f: yaml.dump(data_yaml, f)
    print("Done.")

