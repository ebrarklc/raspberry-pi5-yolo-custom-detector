"""manual_ncnn_decode.py: Ultralytics OLMADAN, sadece ncnn + numpy + OpenCV ile YOLO çıkarımı.

"Sıfırdan Kahramana" dokümanının 9. seviyesi: Ultralytics'in predict() içinde yaptığı her şeyi
(letterbox → normalize → CHW → NCNN → (1,6,2100) çıkışını çözme → eşik → NMS → koordinatları geri ölçekleme)
kendimiz yazıyoruz. Sonuçlar Ultralytics ile aynı olmalı.

Kullanım:  python manual_ncnn_decode.py [resim_klasörü]      (varsayılan: images/box_wheel_test)
Çıktı:     runs/manual_ncnn/  (kutulu resimler)
"""
import sys
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np

MODEL_DIR = Path("models/box_wheel_320_yolo11n_ncnn_model")
IMG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/box_wheel_test")
OUT_DIR = Path("runs/manual_ncnn")
IMGSZ, CONF, IOU = 320, 0.5, 0.7             # imgsz = eğitimdeki boyut! IOU=0.7 Ultralytics predict varsayılanı
NAMES = {0: "box", 1: "wheel"}
COLORS = {0: (255, 60, 0), 1: (255, 220, 0)}  # BGR


# ── 1) Ön işleme: letterbox ──────────────────────────────────────
def letterbox(bgr, new=IMGSZ):
    """En-boy oranını koruyarak new×new kareye sığdır, boşlukları gri (114) doldur."""
    h, w = bgr.shape[:2]
    r = min(new / h, new / w)                         # ölçek: uzun kenar new olacak
    nh, nw = round(h * r), round(w * r)
    pad_x, pad_y = (new - nw) / 2, (new - nh) / 2     # iki yana eşit boşluk
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    img = cv2.copyMakeBorder(img, top, new - nh - top, left, new - nw - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def preprocess(bgr):
    img, r, px, py = letterbox(bgr)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # model RGB ile eğitildi
    x = rgb.astype(np.float32) / 255.0                # 0–255 → 0–1
    x = np.ascontiguousarray(x.transpose(2, 0, 1))    # HWC → CHW (NCNN kanal önde)
    return x, r, px, py


# ── 2) IoU ve NMS ────────────────────────────────────────────────
def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_thr=IOU):
    """En yüksek skorlu kutuyu tut, onunla çok örtüşenleri sil, tekrarla."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        order = order[1:][iou(boxes[i], boxes[order[1:]]) < iou_thr]
    return keep


# ── 3) Çıkışı çöz: (1, 6, 2100) → kutular ────────────────────────
def postprocess(out, r, px, py, orig_hw):
    p = out.T                                         # (6, 2100) → (2100, 6): her satır bir aday
    xywh, cls_scores = p[:, :4], p[:, 4:]             # [xc, yc, w, h] (320'lik giriş pikselinde) + 2 skor
    cls_id = cls_scores.argmax(1)                     # en yüksek skorlu sınıf
    score = cls_scores.max(1)
    m = score > CONF                                  # 2100 adaydan eşiği geçenler
    xywh, cls_id, score = xywh[m], cls_id[m], score[m]

    xyxy = np.empty_like(xywh)                        # merkez+boyut → köşeler
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    keep = []                                         # sınıf başına NMS
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        keep += [idx[k] for k in nms(xyxy[idx], score[idx])]
    keep = np.array(keep, dtype=int)
    xyxy, cls_id, score = xyxy[keep], cls_id[keep], score[keep]

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r      # letterbox'ı geri al: boşluğu çıkar, ölçeği böl
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
    h, w = orig_hw
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return xyxy, cls_id, score


# ── 4) Ana akış ──────────────────────────────────────────────────
net = ncnn.Net()
net.opt.num_threads = 4
net.load_param(str(MODEL_DIR / "model.ncnn.param"))
net.load_model(str(MODEL_DIR / "model.ncnn.bin"))
IN, OUT = net.input_names()[0], net.output_names()[0]   # Ultralytics export'unda: in0, out0
print(f"Giriş: {IN} | Çıkış: {OUT}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def run(x):
    ex = net.create_extractor()
    ex.input(IN, ncnn.Mat(x))
    _, out = ex.extract(OUT)
    return np.array(out)                              # (6, 2100)


x0, *_ = preprocess(cv2.imread(str(images[0])))
for _ in range(5):                                    # ısınma
    run(x0)

times = []
for path in images:
    bgr = cv2.imread(str(path))
    x, r, px, py = preprocess(bgr)
    t0 = time.perf_counter()
    out = run(x)
    times.append((time.perf_counter() - t0) * 1000)
    boxes, cls_id, score = postprocess(out, r, px, py, bgr.shape[:2])

    for (x1, y1, x2, y2), c, s in zip(boxes.astype(int), cls_id, score):
        cv2.rectangle(bgr, (x1, y1), (x2, y2), COLORS[int(c)], 2)
        cv2.putText(bgr, f"{NAMES[int(c)]} {s:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[int(c)], 1)
    cv2.imwrite(str(OUT_DIR / path.name), bgr)

    names = [NAMES[int(c)] for c in cls_id]
    print(f"{path.name[:28]:28s} box={names.count('box')} wheel={names.count('wheel')}  "
          f"{times[-1]:5.1f} ms  skorlar: {' '.join(f'{s:.2f}' for s in sorted(score, reverse=True))}")

print(f"\nOrtalama NCNN çıkarımı: {sum(times) / len(times):.1f} ms  →  {OUT_DIR}/")
