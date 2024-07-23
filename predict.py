from ultralytics import YOLO
import numpy as np


# load a custom model
model = YOLO("./runs/classify/train26/weights/best.pt")

# Predict with the model
results = model("./data/train/sunrise/sunrise7.jpg")  # predict on an image

#Print predictions with scores
names_dict = results[0].names

probs = results[0].probs.data.tolist()

# print(names_dict)
# print(probs)

print(names_dict[np.argmax(probs)])