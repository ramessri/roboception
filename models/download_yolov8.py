from pathlib import Path
from ultralytics import YOLO
import os

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

model = YOLO('yolov8n.pt')

model.export(
    format='onnx',
    imgsz=640,
    opset=12,
    simplify=True
)