from ultralytics import YOLO
import shutil
import os

def train_long():
    model = YOLO('yolo11n.yaml') 

    results = model.train(
        data='/content/dataset_yolo/data.yaml',
        epochs=150,            
        imgsz=640,
        batch=16,
        project='/content/drive/MyDrive/My_Object_Detection_Project',
        name='train_scratch_150e',
        device=0,
    )


    
    source_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    destination_path = '/content/drive/MyDrive/My_Object_Detection_Project/final_model_150e.pt'
    
    try:
        shutil.copy(source_path, destination_path)
        print(f"Model saved to: {destination_path}")
    except Exception as e:
        print(f"Could not copy file: {e}")
        print(f"Model is still at: {source_path}")

if __name__ == '__main__':
    train_long()