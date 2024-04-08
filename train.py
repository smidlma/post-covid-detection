from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
results = model.train(data='data.yaml', epochs=1, imgsz=640, device='mps', project = './runs')
