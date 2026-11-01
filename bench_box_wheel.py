"""bench_box_wheel.py: PyTorch (.pt) ve NCNN modellerini aynı test resimlerinde karşılaştırır.
Hem HIZ hem DOĞRULUK: export sonucu değiştirmiş mi?"""
from pathlib import Path
from ultralytics import YOLO

IMG_DIR = Path("images/box_wheel_test")
IMGSZ, CONF = 320, 0.5
TUR = 3                                     # her resim 3 tur → 30 ölçüm, daha kararlı ortalama
MODELS = {
    "PyTorch": "models/box_wheel_320_yolo11n.pt",
    "NCNN":    "models/box_wheel_320_yolo11n_ncnn_model",
}

images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
hiz, tespit = {}, {}

for ad, yol in MODELS.items():
    model = YOLO(yol, task="detect")

    # Isınma
    for _ in range(5):
        model.predict(images[0], imgsz=IMGSZ, conf=CONF, verbose=False)

    pre, inf, post = [], [], []
    tespit[ad] = {}
    for _ in range(TUR):
        for img in images:
            r = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
            pre.append(r.speed["preprocess"])
            inf.append(r.speed["inference"])
            post.append(r.speed["postprocess"])
            siniflar = [model.names[int(c)] for c in r.boxes.cls]
            skorlar = sorted((round(s, 2) for s in r.boxes.conf.tolist()), reverse=True)
            tespit[ad][img.name] = (siniflar.count("box"), siniflar.count("wheel"), skorlar)

    ort = lambda xs: sum(xs) / len(xs)
    hiz[ad] = (ort(pre), ort(inf), ort(post))

# ── 1) Hız tablosu ───────────────────────────────────────────────
print(f"\n{'Model':8s} {'pre':>7s} {'inf':>8s} {'post':>7s} {'toplam':>8s} {'FPS':>6s}")
for ad, (p, i, o) in hiz.items():
    t = p + i + o
    print(f"{ad:8s} {p:6.1f}ms {i:7.1f}ms {o:6.1f}ms {t:7.1f}ms {1000 / t:6.1f}")
hizlanma = hiz["PyTorch"][1] / hiz["NCNN"][1]
print(f"\nNCNN çıkarım hızlanması: ×{hizlanma:.1f}")

# ── 2) Doğruluk: her resimde aynı sonuç mu? ──────────────────────
print(f"\n{'Resim':28s} {'PyTorch':>12s} {'NCNN':>12s}  durum")
ayni = 0
for img in images:
    a, b = tespit["PyTorch"][img.name], tespit["NCNN"][img.name]
    sayi_ayni = a[:2] == b[:2]
    fark = max((abs(x - y) for x, y in zip(a[2], b[2])), default=0.0)
    ayni += sayi_ayni
    durum = f"✅ skor farkı ≤ {fark:.2f}" if sayi_ayni else "❌ SAYI FARKLI"
    print(f"{img.name[:28]:28s} {f'{a[0]}b {a[1]}w':>12s} {f'{b[0]}b {b[1]}w':>12s}  {durum}")
print(f"\n{ayni}/{len(images)} resimde nesne sayıları aynı")
