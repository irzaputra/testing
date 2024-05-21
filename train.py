import os
import torch
from ultralytics import YOLO, settings

MODEL_PATH ="yolov8n.pt" # nano model
# MODEL_PATH = "yolov8s.pt" # small model
# MODEL_PATH = "yolov8m.pt" # medium model
# MODEL_PATH = "yolov8l.pt" # large model
# MODEL_PATH = "yolov8x.pt" # huge model

if __name__ == "__main__":
    torch.cuda.empty_cache()

    settings.update({"datasets_dir": os.path.join(os.getcwd(), "./training_data_preprocessed/")})
    settings.save()

    #load model
    model = YOLO(MODEL_PATH)
    model.train(data= "./data.yaml", epochs= 30, patience=50)