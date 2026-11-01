from ultralytics import YOLO
import numpy as np

MODELS = [
    ("YOLOv8n", "PyTorch", "models/yolov8n.pt"),
    ("YOLOv8n", "NCNN",    "models/yolov8n_ncnn_model"),
    ("YOLO11n", "PyTorch", "models/yolo11n.pt"),
    ("YOLO11n", "NCNN",    "models/yolo11n_ncnn_model"),
    ("YOLO26n", "PyTorch", "models/yolo26n.pt"),
    ("YOLO26n", "NCNN",    "models/yolo26n_ncnn_model"),
    ("YOLO26n", "ONNX",    "models/yolo26n.onnx"),
]
IMG = "images/bus.jpg"

print(f"{'Model':8s} {'Format':8s} {'Inference (ms)':>16s} {'Postprocess (ms)':>17s}  Tespitler")
for name, fmt, path in MODELS:
    model = YOLO(path, task="detect")
    for _ in range(3):                                   # warm-up
        model.predict(IMG, imgsz=640, verbose=False)
    inf, post = [], []
    for _ in range(10):                                  # ölçüm
        r = model.predict(IMG, imgsz=640, verbose=False)[0]
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    names = [r.names[int(c)] for c in r.boxes.cls]
    summary = ", ".join(f"{names.count(n)} {n}" for n in sorted(set(names)))
    print(f"{name:8s} {fmt:8s} {np.mean(inf):9.1f} ± {np.std(inf):4.1f} {np.mean(post):10.1f} ± {np.std(post):4.1f}  {summary}")
