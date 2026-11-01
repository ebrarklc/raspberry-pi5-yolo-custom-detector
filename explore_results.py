from ultralytics import YOLO

model = YOLO("models/yolo11n_ncnn_model", task="detect")
r = model.predict("images/bus.jpg", imgsz=640, conf=0.5, verbose=False)[0]

print("Tespit sayısı      :", len(r.boxes))
print("\nboxes.data (tablo) :\n", r.boxes.data)
print("\nboxes.xyxy (piksel):\n", r.boxes.xyxy)
print("\nboxes.xyxyn (0–1)  :\n", r.boxes.xyxyn)
print("\nboxes.conf (skor)  :", r.boxes.conf)
print("boxes.cls (sınıf)  :", r.boxes.cls)
print("names[0], names[5] :", r.names[0], ",", r.names[5])
print("orig_shape (H, W)  :", r.orig_shape)

r.save(filename="images/bus_yolo11n_ncnn.jpg")
print("\n✅ Kutulu resim kaydedildi: images/bus_yolo11n_ncnn.jpg")
