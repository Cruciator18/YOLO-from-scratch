import os
from ultralytics import YOLO

model_path = '/content/drive/MyDrive/My_Object_Detection_Project/final_model_150e.pt'
model = YOLO(model_path)

print(f"Loading model from: {model_path}")
print("Processing video...")
results = model.predict(source='test_traffic.mp4', save=True, conf=0.20, iou=0.45)

latest_run = max([os.path.join('runs/detect', d) for d in os.listdir('runs/detect')], key=os.path.getmtime)
saved_video_path = os.path.join(latest_run, 'test_video_traffic.mp4')

print(f" video saved at: {saved_video_path}")
