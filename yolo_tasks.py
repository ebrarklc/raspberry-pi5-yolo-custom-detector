from ultralytics import YOLO
import time

IMG = "images/bus.jpg"

# 🎭 Segmentasyon
seg = YOLO("yolo11n-seg.pt")
r = seg.predict(IMG, imgsz=640, conf=0.5, verbose=False)[0]
t0 = time.perf_counter(); r = seg.predict(IMG, imgsz=640, conf=0.5, verbose=False)[0]
print(f"🎭 SEGMENT  | {(time.perf_counter()-t0)*1000:6.0f} ms | {len(r.boxes)} nesne | masks.data shape: {tuple(r.masks.data.shape)}")
r.save(filename="images/bus_seg.jpg")

# 🕺 Poz tahmini
pose = YOLO("yolo11n-pose.pt")
r = pose.predict(IMG, imgsz=640, conf=0.5, verbose=False)[0]
t0 = time.perf_counter(); r = pose.predict(IMG, imgsz=640, conf=0.5, verbose=False)[0]
print(f"🕺 POSE     | {(time.perf_counter()-t0)*1000:6.0f} ms | {len(r.boxes)} kişi  | keypoints.data shape: {tuple(r.keypoints.data.shape)}")
print("   1. kişinin burnu (x, y, görünürlük):", r.keypoints.data[0][0].tolist())
r.save(filename="images/bus_pose.jpg")

# 🏷️ Sınıflandırma (ImageNet)
cls = YOLO("yolo11n-cls.pt")
r = cls.predict(IMG, verbose=False)[0]
t0 = time.perf_counter(); r = cls.predict(IMG, verbose=False)[0]
top5 = ", ".join(f"{r.names[i]} %{r.probs.data[i]*100:.0f}" for i in r.probs.top5)
print(f"🏷️ CLASSIFY | {(time.perf_counter()-t0)*1000:6.0f} ms | top-5: {top5}")
