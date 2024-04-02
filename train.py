from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8l-obb.pt') # load a pretrained model (recommended for training)
model = YOLO(r"D:\yolo_training\runs\obb\train\weights\best.pt")  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data='coco128.yaml', epochs=10, imgsz=640)

model.predict(r"C:\Users\user\Downloads\DJI_20230402134748_0045_T.JPG", save=True, conf=0.69)