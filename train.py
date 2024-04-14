from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
results = model.train(data='data.yaml', epochs=5, imgsz=640, device='mps', project = './runs')
