from ultralytics import YOLO
import shutil
import os
from data_converter import create_custom_model_config , convert_to_yolo
from data_converter import MODEL_CONFIG_NAME ,DEST_ROOT

def train_custom():
    print(f"Training: {MODEL_CONFIG_NAME}")
    model = YOLO(MODEL_CONFIG_NAME)
    results = model.train(
        data=f"{DEST_ROOT}/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        project='/content/drive/MyDrive/My_Object_Detection_Project',
        name='custom_cnn_scratch',
        device=0,
        plots=True
    )
    source_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    destination_path = '/content/drive/MyDrive/My_Object_Detection_Project/final_custom_model.pt'
    try:
        shutil.copy(source_path, destination_path)
        print(f"Saved: {destination_path}")
    except Exception:
        print(f"Safe: {source_path}")

if __name__ == "__main__":
    create_custom_model_config()
    convert_to_yolo()
    train_custom()