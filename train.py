from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.train(
    data="/Users/kimsonghyeon/Desktop/project/dataset",
    epochs=20
)