"""predict_box_wheel.py: Eğittiğimiz box/wheel modelini test resimlerinde çalıştırır.
Kullanım: python predict_box_wheel.py [imgsz]     (varsayılan: 320)"""
import sys
from pathlib import Path
from ultralytics import YOLO

# ── Ayarlar ──────────────────────────────────────────────────────
MODEL   = "models/box_wheel_320_yolo11n.pt"
IMG_DIR = Path("images/box_wheel_test")
IMGSZ   = int(sys.argv[1]) if len(sys.argv) > 1 else 320
CONF    = 0.5
OUT_DIR = Path("runs").resolve()          # MUTLAK yol → runs/detect/ altına gitmez

# ── Modeli yükle ─────────────────────────────────────────────────
model = YOLO(MODEL, task="detect")
print("Sınıflar:", model.names)

images = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print(f"{len(images)} resim | imgsz={IMGSZ} | conf={CONF}\n")

# ── Isınma (warm-up): ilk çağrılar ölçüme karışmasın ─────────────
for _ in range(3):
    model.predict(images[0], imgsz=IMGSZ, conf=CONF, verbose=False)

# ── Her resimde tahmin ───────────────────────────────────────────
toplam_box = toplam_wheel = 0
sureler = []
for img in images:
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False,
                      save=True, project=str(OUT_DIR), name=f"box_wheel_pt_{IMGSZ}", exist_ok=True)[0]

    siniflar = [model.names[int(c)] for c in r.boxes.cls]
    n_box, n_wheel = siniflar.count("box"), siniflar.count("wheel")
    toplam_box += n_box
    toplam_wheel += n_wheel
    sureler.append(r.speed["inference"])

    skorlar = " ".join(f"{s:.2f}" for s in r.boxes.conf.tolist())
    print(f"{img.name[:28]:28s}  box={n_box} wheel={n_wheel}  "
          f"{r.speed['inference']:5.1f} ms  skorlar: {skorlar}")

# ── Özet ─────────────────────────────────────────────────────────
print(f"\nTOPLAM: {toplam_box} box, {toplam_wheel} wheel")
print(f"Ortalama çıkarım: {sum(sureler) / len(sureler):.1f} ms")
print(f"Çizilmiş resimler: {OUT_DIR / f'box_wheel_pt_{IMGSZ}'}")
