import os
import shutil
import re
import yaml
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


SOURCE_ROOT = "/content/drive/MyDrive/VOC2005_2"


DEST_ROOT = "/content/dataset_yolo"


CLASS_MAP = {
    'bicycle': 0, 'bike': 0, 'velo': 0,
    'car': 1, 'voiture': 1,
    'motorbike': 2, 'motocyclette': 2,
    'pedestrian': 3, 'person': 3, 'pieton': 3
}

def convert_to_yolo():
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)

    for split in ['train', 'val']:
        os.makedirs(f"{DEST_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{DEST_ROOT}/labels/{split}", exist_ok=True)

    ann_dir = os.path.join(SOURCE_ROOT, "Annotations")
    img_dir = os.path.join(SOURCE_ROOT, "PNGImages")

    samples = []

    print(f"Scanning files in {SOURCE_ROOT}...")
    if not os.path.exists(ann_dir):
        print(f"CRITICAL ERROR: Could not find {ann_dir}")
        return

    for class_folder in os.listdir(ann_dir):
        if class_folder in CLASS_MAP:
            cls_id = CLASS_MAP[class_folder]
            folder_path = os.path.join(ann_dir, class_folder)

            if not os.path.isdir(folder_path):
                continue

            for txt_file in os.listdir(folder_path):
                if txt_file.endswith(".txt"):
                    base_name = os.path.splitext(txt_file)[0]
                    img_name = base_name + ".png"
                    img_src = os.path.join(img_dir, class_folder, img_name)
                    txt_src = os.path.join(folder_path, txt_file)

                    if os.path.exists(img_src):
                        samples.append((img_src, txt_src, cls_id))

    print(f"Found {len(samples)} valid samples.")
    if len(samples) == 0:
        print("No samples found. Check your SOURCE_ROOT path!")
        return


    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    def process_samples(sample_list, split):
        for img_path, txt_path, cls_id in tqdm(sample_list, desc=f"Converting {split}"):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size

                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                boxes = re.findall(r':\s*\(\s*(\d+),\s*(\d+)\s*\)\s*-\s*\(\s*(\d+),\s*(\d+)\s*\)', content)

                yolo_data = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = map(float, box)

                    xmin = max(0, min(xmin, w))
                    ymin = max(0, min(ymin, h))
                    xmax = max(0, min(xmax, w))
                    ymax = max(0, min(ymax, h))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    x_center = ((xmin + xmax) / 2) / w
                    y_center = ((ymin + ymax) / 2) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h

                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)

                    yolo_data.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                if yolo_data:
                    shutil.copy(img_path, f"{DEST_ROOT}/images/{split}/{os.path.basename(img_path)}")
                    label_name = os.path.splitext(os.path.basename(txt_path))[0] + ".txt"
                    with open(f"{DEST_ROOT}/labels/{split}/{label_name}", 'w') as f:
                        f.write("\n".join(yolo_data))

            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    process_samples(train_samples, 'train')
    process_samples(val_samples, 'val')

    data_yaml = {
        'path': DEST_ROOT,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 4,
        'names': ['bicycle', 'car', 'motorbike', 'pedestrian']
    }

    with open(f"{DEST_ROOT}/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f)

    print("\nSUCCESS: Data preparation complete.")
    print(f"YAML saved at: {DEST_ROOT}/data.yaml")
    print("You can now start training with: model.train(data='/content/dataset_yolo/data.yaml', ...)")

if __name__ == "__main__":
    convert_to_yolo()