from ultralytics import YOLO

# load a pretrained model (recommended for training)
model = YOLO("yolov8n-cls.pt")  
# Train the model
results = model.train(data="./data", epochs=20, imgsz=64)
