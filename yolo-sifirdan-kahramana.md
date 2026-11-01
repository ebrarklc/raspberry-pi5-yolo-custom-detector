# 🦸 YOLO: Sıfırdan Kahramana

> **Kimin için?** Python bilen, "nesne tespiti" lafını duymuş ama YOLO'nun içinde ne olduğunu bilmeyen herkes. Hedef: bu dokümanın sonunda bir YOLO modelini **eğitebilmek, çıktısını elle çözebilmek, edge cihaza deploy edebilmek ve bir mülakatta anlatabilmek**.
>
> **Nasıl okunur?** 16 seviye var. Her seviye bir öncekinin üzerine kurulur. Her seviyenin sonunda **✅ Kendini test et** soruları var; cevapları tıklayınca açılır. Takıldığın seviyede dur, tekrar oku, sonra devam et.
>
> **Uygulama:** Bu repodaki kodlar (Raspberry Pi 5 + YOLO11n, box/wheel veri seti) ve [Bölüm 6 ders notu](06-yolo-raspberry-pi.md) seviyelerdeki her kavramın **gerçek bir örneğini** içeriyor. Örneklerdeki sayılar bizim kendi ölçümlerimiz.
>
> **Hazırlayan:** Ebrar Kılıç

---

## 🗺️ Yol haritası

| Seviye | Konu | Bunu bitirince… |
|---|---|---|
| 🥚 0 | Görüntü görevleri haritası | Sınıflandırma, tespit, segmentasyon, poz arasındaki farkı anlatırsın |
| 🐣 1 | Bir kutuyu sayılarla anlatmak | 3 kutu formatını birbirine çevirirsin |
| 🐣 2 | IoU | İki kutunun ne kadar örtüştüğünü hesaplarsın |
| 🐥 3 | YOLO'nun büyük fikri | "Neden *You Only Look Once*?" sorusunu cevaplarsın |
| 🐥 4 | Anchor'lar ve anchor-free | Modern YOLO'nun kutuyu nasıl tahmin ettiğini bilirsin |
| 🐥 5 | Mimari: backbone, neck, head | Eğitim log'undaki katman tablosunu okursun |
| 🦅 6 | Çıkış tensörünün matematiği | `(1, 84, 8400)` gibi bir şekli görünce anlamını söylersin |
| 🦅 7 | NMS ve NMS'siz YOLO | Kopya kutuların nasıl temizlendiğini bilirsin |
| 🦅 8 | Ön işleme: letterbox | Kutuları orijinal resme doğru geri taşırsın |
| 🦅 9 | Kendi decode'unu yaz | Ultralytics olmadan YOLO çalıştırırsın |
| 🦸 10 | Eğitim: veri, loss, augmentation | Bir eğitim log'unu baştan sona okursun |
| 🦸 11 | Metrikler | P, R, mAP50, mAP50-95 ve eşik seçimini savunursun |
| 🦸 12 | Veri seti stratejisi | Kendi veri setini doğru kurarsın |
| 🦸 13 | Export ve deploy | Modeli doğru formatta, doğru boyutta, doğrulayarak taşırsın |
| 🦸 14 | Edge'de hız | 100 ms'lik modeli 15 ms'ye indirirsin |
| 🦸 15 | Takip ve sayma | Bölüm 7'ye hazır olursun |
| 📜 16 | YOLO soy ağacı | v1'den YOLO26'ya kim ne getirdi, bilirsin |
| 🏆 | Final | Mülakat soruları, hata ayıklama tablosu, proje fikirleri |

---

## 🥚 Seviye 0 — Görüntü görevleri haritası

Bilgisayarlı görüdeki dört temel görev, **"resim hakkında ne kadar detaylı bilgi istiyorum?"** sorusuna göre sıralanır:

| Görev | Soru | Çıktı | YOLO modeli |
|---|---|---|---|
| **Sınıflandırma** (classify) | Bu resimde **ne** var? | 1 etiket + olasılık | `yolo11n-cls.pt` |
| **Nesne tespiti** (detect) | **Neler** var ve **nerede**? | Her nesne için kutu + etiket + skor | `yolo11n.pt` |
| **Örnek segmentasyonu** (segment) | Her nesnenin **tam şekli** ne? | Her nesne için piksel maskesi | `yolo11n-seg.pt` |
| **Poz tahmini** (pose) | İnsanın **eklemleri** nerede? | Kişi başına 17 nokta | `yolo11n-pose.pt` |

```
Sınıflandırma:  "otobüs"                          ← en az bilgi, en hızlı
Tespit:         [otobüs 📦] [kişi 📦] [kişi 📦]
Segmentasyon:   otobüsün piksel piksel silueti
Poz:            kişinin iskeleti (burun, omuz, dirsek…)  ← en çok bilgi
```

**Neden önemli?** Doğru görevi seçmek, projenin yarısıdır. "Bantta kaç kutu geçti?" sorusu için segmentasyon gereksiz pahalıdır; tespit yeter. "Meyvenin ne kadarı çürük?" sorusu için ise kutu yetmez, maske gerekir.

> **📝 Mülakat notu:** Ultralytics'te dört görev de **aynı API**'yi kullanır: `YOLO("model.pt").predict(...)`. Değişen tek şey modelin head'i ve `Results` nesnesindeki alan (`boxes`, `masks`, `keypoints`, `probs`).

<details><summary>✅ Kendini test et</summary>

1. Bir AGV'nin önündeki engelin **ne olduğunu ve nerede olduğunu** bilmesi gerekiyor. Hangi görev? → **Tespit.**
2. Bir fabrikada kumaştaki lekenin **alanını** ölçmek istiyoruz. Hangi görev? → **Segmentasyon** (kutu alanı lekenin alanı değildir).
3. Spor salonunda squat'ın doğru yapılıp yapılmadığını kontrol eden bir uygulama? → **Poz tahmini** (diz ve kalça açıları).
</details>

---

## 🐣 Seviye 1 — Bir kutuyu sayılarla anlatmak

Bir **sınırlayıcı kutu** (*bounding box*) dört sayıyla tanımlanır. Ama hangi dört sayı? Üç yaygın format var:

```
(0,0) ──────────────── x ──▶
  │        (x1,y1)
  │          ┌────────────┐
  │          │            │ h
  y          │     ●(xc,yc)│
  │          │            │
  ▼          └────────────┘
                   w      (x2,y2)
```

| Format | Sayılar | Kim kullanır? |
|---|---|---|
| **xyxy** | sol-üst köşe + sağ-alt köşe `[x1, y1, x2, y2]` | YOLO çıktısı (`r.boxes.xyxy`), çizim fonksiyonları |
| **xywh (merkez)** | merkez + boyut `[xc, yc, w, h]` | YOLO etiket dosyaları, YOLO'nun ham çıktısı |
| **yxyx** | `[ymin, xmin, ymax, xmax]` | TensorFlow / SSD / EfficientDet (Bölüm 4) |

Bunlara bir de **ölçek** boyutu eklenir:
- **Piksel:** `[48, 230, 245, 795]` — resmin gerçek boyutunda.
- **Normalize (0–1):** `[0.06, 0.21, 0.30, 0.74]` — genişliğe ve yüksekliğe bölünmüş. Resim boyutu değişse de geçerli.

### Çeviri formülleri
```python
# xywh (merkez) → xyxy
x1, y1 = xc - w/2, yc - h/2
x2, y2 = xc + w/2, yc + h/2

# xyxy → xywh (merkez)
xc, yc = (x1 + x2)/2, (y1 + y2)/2
w,  h  = x2 - x1,     y2 - y1

# normalize → piksel  (W, H = resmin genişliği ve yüksekliği)
x_px = x_norm * W
y_px = y_norm * H
```

### YOLO etiket dosyası
Her resim için aynı adlı bir `.txt`; her satır bir nesne:
```
0 0.4312 0.5207 0.1984 0.3125
│   │      │      │      └── yükseklik (resim yüksekliğine oranla)
│   │      │      └── genişlik (resim genişliğine oranla)
│   │      └── merkez y
│   └── merkez x
└── sınıf numarası (data.yaml'daki names sırasına göre: 0 = box)
```
**Boş dosya** = resimde nesne yok (**background resmi**).

> **📝 Mülakat notu:** Tespit projelerindeki en sinsi hatalar **koordinat sırası** (x mi önce, y mi?) ve **ölçek** (piksel mi, normalize mi?) karışıklığıdır. Kod çökmez, kutular sadece yanlış yere çizilir. Yeni bir modelle çalışırken ilk iş: tek bir kutuyu elle çizip doğru yerde olduğunu görmek.

<details><summary>✅ Kendini test et</summary>

640×480'lik bir resimde YOLO etiketi `1 0.5 0.5 0.25 0.5`. Kutunun piksel köşeleri?

→ Merkez (320, 240), boyut (160, 240) → **x1=240, y1=120, x2=400, y2=360**.
</details>

---

## 🐣 Seviye 2 — IoU (Intersection over Union)

İki kutu **ne kadar örtüşüyor?** Bu sorunun cevabı, YOLO'nun her yerinde kullanılır: eğitimde (tahmin doğru mu?), NMS'te (bu iki kutu aynı nesne mi?) ve metriklerde (mAP).

```
IoU = Kesişim alanı / Birleşim alanı

  ┌───────┐                 0.0 → hiç örtüşmüyor
  │ A ┌───┼───┐             0.5 → "makul" eşleşme
  │   │ ∩ │   │             1.0 → birebir aynı kutu
  └───┼───┘ B │
      └───────┘
```

### Elle örnek
A = `[0, 0, 10, 10]`, B = `[5, 5, 15, 15]`
- Kesişim: `[5, 5, 10, 10]` → 5 × 5 = **25**
- Birleşim: 100 + 100 − 25 = **175**
- IoU = 25 / 175 = **0.143**

İki kutu çeyrek kare kayınca IoU **0.14'e** düşüyor. IoU, küçük kaymalara karşı **çok hassastır**; bu yüzden mAP50-95 gibi yüksek IoU eşikleri "kutu milimetrik mi oturmuş?" sorusunu ölçer.

### Kod
```python
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)          # örtüşme yoksa 0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)
```

### IoU'nun kardeşleri (kayıp fonksiyonlarında)
| Ad | Ek olarak neyi ölçer? |
|---|---|
| **GIoU** | Kutular hiç örtüşmese bile ne kadar uzak olduklarını |
| **DIoU** | Merkezler arası mesafeyi |
| **CIoU** | Merkez mesafesi + **en-boy oranı** farkını. YOLOv8/11 kutu kaybı bunu kullanır |

<details><summary>✅ Kendini test et</summary>

1. Bir kutu diğerinin tam içinde ve alanının yarısı kadarsa IoU kaç? → İç kutunun alanı = kesişim = S/2, birleşim = S → **0.5**.
2. Neden düz IoU kayıp fonksiyonu olarak tek başına yetmez? → Kutular hiç örtüşmüyorsa IoU = 0 ve **gradyan da 0** olur; model "hangi yöne kaymalıyım?" bilgisini alamaz. GIoU/DIoU/CIoU bu sorunu çözer.
</details>

---

## 🐥 Seviye 3 — YOLO'nun büyük fikri

### Önce: iki aşamalı dedektörler (R-CNN ailesi, 2014–2016)
```
Resim → 1) "Nesne olabilecek ~2000 bölge öner" → 2) Her bölgeyi ayrı ayrı sınıflandır
```
Doğruydu ama **yavaştı**: Aynı resme yüzlerce kez bakılıyordu.

### YOLO (2016): "Resme bir kez bak"
Joseph Redmon'un fikri: Tespiti **tek bir regresyon problemi** olarak gör. Resmi bir **ızgaraya** böl; her hücre, **merkezi kendi içine düşen** nesneyi tahmin etsin. Tek bir ileri geçişte (*forward pass*) bütün kutular çıksın.

```
YOLOv1: 448×448 resim → 7×7 ızgara
┌──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │   Her hücre:
├──┼──┼──┼──┼──┼──┼──┤   • 2 kutu tahmini (x, y, w, h, güven)
│  │  │🐶│  │  │  │  │   • 20 sınıf olasılığı (Pascal VOC)
├──┼──┼──┼──┼──┼──┼──┤
│  │  │  │  │🚲│  │  │   Çıkış: 7 × 7 × (2×5 + 20) = 7×7×30
└──┴──┴──┴──┴──┴──┴──┘
```

| | İki aşamalı (Faster R-CNN) | Tek aşamalı (YOLO, SSD) |
|---|---|---|
| Akış | Bölge öner → sınıflandır | Tek geçişte hepsi |
| Hız | Yavaş | **Hızlı** (YOLOv1: 45 FPS, 2016'da) |
| Doğruluk | Tarihsel olarak daha yüksek | Bugün aradaki fark büyük ölçüde kapandı |
| Edge'e uygun mu? | Zor | ✅ |

**YOLOv1'in zayıflıkları** (sonraki sürümlerin çözdükleri):
- Her hücre en fazla 2 nesne → **kalabalık, küçük nesnelerde** (sürü kuşlar) başarısız.
- Tek ölçek → **küçük nesneler** kaçıyor.

> **📝 Mülakat notu:** "YOLO neden hızlı?" → Tespiti tek bir sinir ağı geçişinde, ızgara tabanlı **yoğun tahmin** (*dense prediction*) olarak yapar; ayrı bir bölge önerme aşaması yoktur.

<details><summary>✅ Kendini test et</summary>

YOLOv1 13 küçük kuşun aynı hücreye düştüğü bir resimde en fazla kaç kuş bulabilir? → Hücre başına 2 kutu ama **tek sınıf olasılığı** var; pratikte o hücreden **en fazla 2 kuş**.
</details>

---

## 🐥 Seviye 4 — Anchor'lar ve anchor-free

### Anchor tabanlı dönem (YOLOv2–v7)
**Anchor (çapa):** Her hücreye önceden tanımlanmış birkaç **hazır kutu şablonu** (örneğin uzun-ince, kare, yatay-geniş). Model sıfırdan kutu üretmek yerine "bu şablonu ne kadar kaydırıp büyüteyim?" sorusunu cevaplar.

```
Hücre:   ┌─┐  ┌───┐  ┌──────┐     ← 3 anchor (veri setinden k-means ile seçilir)
         │ │  │   │  └──────┘
         └─┘  └───┘
Tahmin:  anchor + küçük düzeltmeler (Δx, Δy, Δw, Δh)
```
**Sorun:** Anchor boyutları veri setine özel ayarlanmalı. Kendi veri setin farklı oranlarda nesneler içeriyorsa (örneğin çok ince kablolar) performans düşer. Ayrıca ekstra hiperparametre demek.

### Anchor-free dönem (YOLOX, YOLOv8, YOLO11, YOLO26)
Her ızgara hücresinin merkezi bir **anchor noktası**dır (şablon kutu yok). Model, o noktadan kutunun **dört kenarına olan mesafeleri** tahmin eder:

```
             t (üst)
             ↑
  l (sol) ←  ●  → r (sağ)        Kutu = [xc - l, yc - t, xc + r, yc + b] (stride ile çarpılarak)
             ↓
             b (alt)
```
- Kutu şablonu yok → veri setine göre ayar gerekmez.
- Daha basit, daha genellenebilir.

> **📝 Mülakat notu:** "YOLOv5 ile YOLOv8 arasındaki en büyük mimari fark?" → v8 **anchor-free** ve **decoupled head** (kutu ve sınıf için ayrı dallar) kullanır; v5 anchor tabanlı ve birleşik head'lidir.

<details><summary>✅ Kendini test et</summary>

Anchor tabanlı bir modeli, eğitildiği veri setinden çok farklı en-boy oranlarında nesneler içeren bir veri setinde kullanırsan ne olur? → Anchor'lar nesnelere uymaz; model büyük düzeltmeler yapmak zorunda kalır, **recall düşer**. Çözüm: anchor'ları yeni veriye göre yeniden hesaplamak ya da anchor-free model.
</details>

---

## 🐥 Seviye 5 — Mimari: backbone, neck, head

```
GİRİŞ 320×320×3
   │
┌──▼──────────────────────────┐
│ BACKBONE (gövde)            │  "Ne görüyorum?"
│ Conv(s2) → C3k2 → ... SPPF  │  Çözünürlük ↓ (320→160→80→40→20→10)
│ → C2PSA                     │  Kanal ↑ (16→32→64→128→256)
└──┬────────┬────────┬────────┘
   P3 (40×40) P4 (20×20) P5 (10×10)    ← üç farklı ölçekte özellik haritası
┌──▼────────▼────────▼────────┐
│ NECK (boyun) — FPN + PAN    │  Ölçekleri birleştir:
│ Upsample ↑ + Concat         │  derin "anlam" bilgisi ↔ sığ "detay" bilgisi
│ Conv(s2) ↓ + Concat         │
└──┬────────┬────────┬────────┘
┌──▼────────▼────────▼────────┐
│ HEAD (baş) — Detect         │  Her ölçekte, her hücre için:
│ kutu dalı ║ sınıf dalı      │  kutu (4×16 DFL) + sınıf (nc)
└─────────────────────────────┘
```

### Yapı taşları (YOLO11)
| Blok | Görevi | Benzetme |
|---|---|---|
| **Conv** | Konvolüsyon + BatchNorm + SiLU aktivasyon | Temel tuğla |
| **Conv stride 2** | Boyutu yarıya indir | Uzaklaşıp daha geniş bakmak |
| **C3k2** | Bölünmüş akış + birkaç Bottleneck + birleştirme (CSP fikri) | Bilgiyi iki yoldan taşıyıp birleştiren kavşak |
| **Bottleneck** | Residual (atlama) bağlantılı küçük blok | Bilginin kaybolmaması için "kısa yol" |
| **SPPF** | Art arda max-pool'larla farklı alan boyutlarında bakmak | Aynı sahneye farklı zoom'larla bakmak |
| **C2PSA** | Pozisyonel **self-attention** | "Resmin hangi bölgesi önemli?" |
| **Upsample + Concat** | Haritayı büyütüp sığ katmanla yan yana koymak | Kuşbakışı bilgi + yakın plan detay |
| **Detect** | Decoupled head: kutu ve sınıf ayrı | Biri "nerede?", diğeri "ne?" uzmanı |

### Neden üç ölçek?
| Harita | Stride | Hücre (320'de) | Bir hücre resmin ne kadarını görür? | İyi olduğu nesne |
|---|---|---|---|---|
| P3 | 8 | 40×40 | 8×8 piksel | **Küçük** |
| P4 | 16 | 20×20 | 16×16 piksel | **Orta** |
| P5 | 32 | 10×10 | 32×32 piksel | **Büyük** |

### Model boyutları (n / s / m / l / x)
Aynı mimari, farklı **derinlik** ve **genişlik** çarpanları. Ultralytics dokümantasyonuna göre YOLO11 (COCO, 640):

| Model | Parametre | mAP50-95 (COCO) | Edge için |
|---|---|---|---|
| **YOLO11n** | 2.6 M | 39.5 | ✅ Pi, mikrodenetleyiciye yakın cihazlar |
| YOLO11s | 9.4 M | 47.0 | ✅ Pi 5 (yavaş), Jetson |
| YOLO11m | 20.1 M | 51.5 | Jetson / GPU |
| YOLO11l | 25.3 M | 53.4 | GPU |
| YOLO11x | 56.9 M | 54.7 | GPU sunucu |

> **📝 Mülakat notu:** Edge cihazda doğruluk yetmiyorsa ilk düşünülecekler: daha büyük model (n→s), daha büyük imgsz (yeniden eğiterek!), daha iyi veri. Hız yetmiyorsa: daha küçük imgsz, daha verimli format (NCNN/TensorRT), quantization, NPU.

<details><summary>✅ Kendini test et</summary>

1. Uzaktan çekilmiş bir resimde 10 piksellik dronları tespit etmek istiyorsun. Hangi ölçek haritası en önemli? → **P3 (stride 8)**. P5'te bir hücre 32 piksel; 10 piksellik nesne bir hücrenin içinde kaybolur.
2. Neck neden var; backbone'un son katmanı yetmez mi? → Son katman güçlü **anlam** bilgisi taşır ama çözünürlüğü çok düşüktür (10×10). Küçük nesneler için sığ katmanların **detayı** gerekir. Neck ikisini birleştirir.
</details>

---

## 🦅 Seviye 6 — Çıkış tensörünün matematiği

Bu seviye, YOLO'yu "kullanan" ile "anlayan" arasındaki farktır.

### Aday sayısı
Her ölçek haritasının her hücresi **1 aday** üretir (anchor-free):
```
imgsz = 640:  (640/8)² + (640/16)² + (640/32)² = 80² + 40² + 20² = 6400 + 1600 + 400 = 8400
imgsz = 320:  (320/8)² + (320/16)² + (320/32)² = 40² + 20² + 10² = 1600 + 400 + 100 = 2100
```

### Her adayın sayıları
**Head'in ham çıktısı** (eğitimde) her aday için:
- **Kutu:** 4 kenar × **16** olasılık kutusu (DFL, aşağıda) = **64** kanal
- **Sınıf:** `nc` kanal (sigmoid öncesi)

**Export edilmiş modelin çıktısı** (çıkarımda) bunlar zaten çözülmüş halde gelir:
```
(1, 4 + nc, aday)
     │   │
     │   └── her sınıf için 0–1 skor (sigmoid uygulanmış)
     └── [xc, yc, w, h]  — giriş resminin (320×320) piksel cinsinden
```
| Model | Çıkış şekli |
|---|---|
| COCO YOLO11n @640 | `(1, 84, 8400)` → 4 + 80 sınıf |
| Bizim box/wheel @320 | `(1, 6, 2100)` → 4 + 2 sınıf |

### DFL (Distribution Focal Loss): kenarı bir sayı değil, dağılım olarak tahmin etmek
Klasik yaklaşım: "Sol kenar 5.3 hücre uzakta" diye **tek bir sayı** tahmin etmek. DFL yaklaşımı: Uzaklığı 0, 1, 2, … 15 arasında **16 kutucuğa** dağıtılmış bir **olasılık** olarak tahmin etmek, sonra **beklenen değeri** almak:

```
Olasılıklar: [0, 0, 0, 0, 0, 0.7, 0.3, 0, ...]   (softmax)
Beklenen değer: 5×0.7 + 6×0.3 = 5.3 → sol kenar 5.3 × stride piksel uzakta
```
**Neden?** Kenarın tam yeri belirsizse (bulanık kenar, gölge) model bu belirsizliği dağılımla ifade edebilir. Eğitim log'undaki **`Freezing layer 'model.23.dfl.conv.weight'`** satırı bu beklenen değeri hesaplayan **sabit ağırlıklı** (0, 1, …, 15) küçük konvolüsyondur; öğrenilmez, bu yüzden dondurulur.

> YOLO26, DFL'yi **kaldırdı**: Daha basit bir kutu çıkışı, edge cihazlarda daha kolay export ve quantization demek.

### Tensör dizilimi: NCHW ve NHWC
| Dizilim | Anlamı | Kim kullanır? |
|---|---|---|
| **NCHW** (BCHW) | `[batch, kanal, yükseklik, genişlik]` | PyTorch, NCNN, ONNX (genelde) |
| **NHWC** | `[batch, yükseklik, genişlik, kanal]` | TensorFlow, LiteRT, OpenCV resimleri (HWC) |

OpenCV resmi `(H, W, 3)` → YOLO'ya vermeden önce `transpose(2, 0, 1)` → `(3, H, W)`.

> **📝 Mülakat notu:** *"(1, 84, 8400) ne demek?"* → batch 1; her aday için 4 kutu koordinatı + 80 COCO sınıf skoru; 640 girişte üç ölçekten 8400 aday. Bu tek cümle, YOLO'yu anladığını kanıtlar.

<details><summary>✅ Kendini test et</summary>

1. 10 sınıflı bir YOLO11 modelini imgsz=480 ile export ettin. Çıkış şekli? → 480/8=60, 480/16=30, 480/32=15 → 3600 + 900 + 225 = **4725**; şekil **(1, 14, 4725)**.
2. imgsz=500 verirsen? → 32'nin katı değil; Ultralytics **512**'ye yuvarlar (stride 32).
</details>

---

## 🦅 Seviye 7 — NMS ve NMS'siz YOLO

### Sorun
Bir nesnenin çevresindeki **komşu hücrelerin çoğu** aynı nesne için kutu üretir. 2100 adaydan eşiği geçenler arasında aynı tekerlek için 5–10 kutu olabilir.

### NMS (Non-Maximum Suppression) algoritması
```
1. Kutuları skora göre büyükten küçüğe sırala
2. En yüksek skorluyu AL (kesin tespit)
3. Onunla IoU'su eşikten (örneğin 0.7) büyük olan bütün kutuları SİL (kopya)
4. Kalan kutularla 2. adıma dön
```
```python
def nms(boxes, scores, iou_thr=0.7):
    order = scores.argsort()[::-1]                 # büyükten küçüğe
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)                             # en iyisini tut
        rest = order[1:]
        order = rest[iou(boxes[i], boxes[rest]) < iou_thr]   # çok örtüşenleri at
    return keep
```
- **Sınıf başına NMS:** Bir "box" kutusu bir "wheel" kutusunu silmemeli. Bu yüzden NMS genelde her sınıf için ayrı yapılır.
- **Ultralytics varsayılanı:** `predict(..., iou=0.7)`.

### NMS'in zayıflıkları
1. **Sıralı (seri) bir algoritma**: GPU/NPU'da paralelleştirmesi zor, edge'de gecikme ekler.
2. **Ayar gerektirir**: IoU eşiği yüksekse kopyalar kalır; düşükse birbirine yakın gerçek nesneler (yan yana tekerlekler) silinir.
3. **Parça kutuları göremez**: Bizim imgsz deneyinde bir tekerleğin üst ve alt yarısına çizilen kutular az örtüştüğü için NMS onları "iki farklı nesne" sandı.

### NMS'siz (end-to-end) YOLO: YOLOv10 ve YOLO26
Eğitimde **bire-bir atama** (*one-to-one assignment*) kullanılır: Her gerçek nesneye sadece **tek bir** aday eşlenir. Model böylece her nesne için **zaten tek kutu** üretmeyi öğrenir; NMS'e gerek kalmaz.
- `metadata.yaml`'da `end2end: true` (YOLO26) / `false` (YOLO11).
- Avantaj: Daha basit ve öngörülebilir **son işleme**, kolay export.

<details><summary>✅ Kendini test et</summary>

Kalabalık bir otoparkta yan yana park etmiş arabaları sayıyorsun ve bazı arabalar sayılmıyor. NMS'in hangi ayarına bakarsın? → **IoU eşiği çok düşük** olabilir: Komşu arabaların kutuları biraz örtüşünce NMS birini "kopya" sanıp siliyor. Eşiği yükselt.
</details>

---

## 🦅 Seviye 8 — Ön işleme: letterbox ve geri ölçekleme

Model **sabit kare** bir giriş bekler (örneğin 320×320). Ama kamera görüntüsü 640×480. Düz `resize` yaparsan nesneler **ezilir** (tekerlek oval olur). Çözüm: **Letterbox.**

```
640×480 resim
  │ 1) Ölçek: r = min(320/480, 320/640) = 0.5 → 320×240
  │ 2) Kalan boşluğu gri (114,114,114) ile doldur: üst 40 + alt 40 piksel
  ▼
┌──────────────────┐ ← gri şerit (40 px)
│                  │
│   resim 320×240  │
│                  │
└──────────────────┘ ← gri şerit (40 px)
```

### Kutuyu geri taşımak
Model, 320×320'lik letterbox'lı resimde koordinat üretir. Orijinal resme dönmek için **tersini** yap:
```python
x_orig = (x_model - pad_x) / r
y_orig = (y_model - pad_y) / r
```
Örnek: Model `y1 = 80` dedi → (80 − 40) / 0.5 = **80** (orijinalde).

### Tam ön işleme zinciri
```python
img, r, pad_x, pad_y = letterbox(bgr, 320)   # kare yap
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # OpenCV BGR verir, model RGB ile eğitildi
x = rgb.astype(np.float32) / 255.0           # 0–255 → 0–1
x = x.transpose(2, 0, 1)                     # HWC → CHW
x = x[None]                                  # → (1, 3, 320, 320)
```

> ⚠️ **Training/serving skew'in en yaygın 3 kaynağı:** (1) BGR/RGB karışıklığı, (2) normalize etmeyi unutmak ya da yanlış yapmak (0–1 yerine −1…1), (3) letterbox yerine düz resize. Üçünde de model çalışır ama sonuçlar kötüleşir.

<details><summary>✅ Kendini test et</summary>

1280×720 bir kareyi 640'lık modele veriyorsun. r, pad_x, pad_y? → r = min(640/720, 640/1280) = **0.5**; yeni boyut 640×360; **pad_x = 0, pad_y = 140**.
</details>

---

## 🦅 Seviye 9 — Kendi decode'unu yaz: Ultralytics olmadan YOLO

Seviye 6–8'deki her şeyi birleştirdik: **[`manual_ncnn_decode.py`](manual_ncnn_decode.py)**. Bu betik Ultralytics kullanmıyor; sadece `ncnn`, `numpy` ve `OpenCV`.

```
Resim (BGR) ─▶ letterbox ─▶ RGB ─▶ /255 ─▶ CHW ─▶ NCNN (in0) ─▶ out0: (6, 2100)
                                                                    │
     kutular ◀─ geri ölçekle ◀─ sınıf başına NMS ◀─ xywh→xyxy ◀─ eşik ◀─ transpose (2100, 6)
```

Çekirdek kısım (çıkışı çözmek):
```python
p = out.T                                  # (6, 2100) → (2100, 6): her satır bir aday
xywh, cls_scores = p[:, :4], p[:, 4:]
cls_id = cls_scores.argmax(1)              # en olası sınıf
score  = cls_scores.max(1)                 # o sınıfın skoru
m = score > 0.5                            # 2100 adaydan eşiği geçenler (genelde 10–50)
# → xywh'yi xyxy'ye çevir → sınıf başına NMS → letterbox'ı geri al
```

**Neden yapıyoruz?**
1. Ultralytics'i **kuramayacağın** cihazlar var (mikrodenetleyici, C++ uygulaması, Android). Orada bu adımları kendin yazarsın.
2. "Model çalışıyor ama kutular yanlış" hatalarını ancak her adımı bilen biri ayıklayabilir.
3. Mülakatlarda "YOLO çıktısını nasıl işlersin?" sorusuna kodla cevap verebilirsin.

**Doğrulama (Pi 5'te gerçek sonuç):** 10 resmin **10'unda** kutu sayıları Ultralytics ile aynı; skorlar aynı ya da ±0.01 farklı. Ortalama çıkarım **16.0 ms** (Ultralytics NCNN: 14.6 ms, aradaki fark iki resimdeki anlık gecikme sıçramalarından). Yani Ultralytics'in `predict()` içinde yaptığı her şeyi ~100 satırlık kendi kodumuzla birebir yeniden ürettik. 🦸

| Resim | Ultralytics (`.pt`, 320) | Kendi decode'umuz (NCNN) |
|---|---|---|
| 00012 | 3 box + 1 wheel · 0.94 0.93 0.92 0.92 | 3 box + 1 wheel · 0.94 0.93 0.92 0.92 |
| 00046 | 3 box + 4 wheel · 0.94 … 0.91 | 3 box + 4 wheel · 0.94 … 0.91 |
| 120418 | 1 wheel · 0.91 | 1 wheel · 0.90 |
| 121009 | — | — |

<details><summary>✅ Kendini test et</summary>

Betikte `x = rgb / 255.0` satırını silersen ne olur? → Model 0–255 arası değerler görür; eğitimde hiç görmediği bir dağılım. Muhtemelen **hiç tespit olmaz ya da saçma kutular** çıkar. Sessiz hata örneği.
</details>

---

## 🦸 Seviye 10 — Eğitim: veri, loss, augmentation

### 10.1 Veri seti yapısı
```
dataset/
├── data.yaml
├── train/images/*.jpg   train/labels/*.txt
├── valid/images/        valid/labels/
└── test/images/         test/labels/
```
```yaml
path: /content/dataset      # mutlak yol önerilir
train: train/images
val: valid/images
test: test/images
nc: 2
names: ['box', 'wheel']     # sıra = sınıf numarası
```

### 10.2 Eğitim komutu
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")                         # önceden eğitilmiş → transfer learning
model.train(data="data.yaml", epochs=50, imgsz=320, batch=16)
```
`yolo11n.yaml` ile başlasaydın model **rastgele ağırlıklarla** (sıfırdan) başlardı; küçük veri setinde bu felaket olur.

### 10.3 Etiket atama (label assignment): "Hangi aday hangi nesneden sorumlu?"
2100 adaydan hangileri "pozitif" (bir nesneyi tahmin etmeli), hangileri "negatif" (arka plan)? YOLOv8/11, **Task-Aligned Assigner (TAL)** kullanır: Bir adayın puanı, **sınıf skoru** ile **kutu IoU**'sunun birleşimidir; her nesneye en iyi uyumlu birkaç aday atanır. Böylece "sınıfı iyi bilen" ile "kutuyu iyi yerleştiren" aday aynı olur.

### 10.4 Kayıp fonksiyonları (log'daki üç sütun)
| Kayıp | Ne ölçer? | Yöntem | Varsayılan ağırlık |
|---|---|---|---|
| `box_loss` | Kutu ne kadar doğru yerde? | **CIoU** | 7.5 |
| `cls_loss` | Sınıf doğru mu? | **BCE** (her sınıf için ayrı evet/hayır) | 0.5 |
| `dfl_loss` | Kenarların dağılımı ne kadar keskin ve doğru? | **DFL** | 1.5 |

**Neden softmax değil BCE?** Her sınıf bağımsız bir "evet/hayır" sorusu. Bu, bir nesnenin birden fazla etikete uyduğu (örneğin "kişi" ve "yaya") veri setlerinde de çalışır.

### 10.5 Augmentation (Ultralytics varsayılanları)
| Ayar | Varsayılan | Ne yapar? |
|---|---|---|
| `mosaic` | 1.0 | 4 resmi 2×2 kolajda birleştirir: çeşitlilik, küçük nesne, kesik nesne |
| `close_mosaic` | 10 | Son 10 epoch'ta mosaic kapalı: gerçek resimlere uyum |
| `fliplr` | 0.5 | %50 yatay çevirme |
| `hsv_h/s/v` | 0.015 / 0.7 / 0.4 | Renk tonu / doygunluk / parlaklık: ışık koşulları |
| `scale` | 0.5 | ±%50 ölçek |
| `translate` | 0.1 | %10 kaydırma |
| `erasing` | 0.4 | Rastgele bir parçayı silme (kısmi kapanma) |
| `mixup`, `copy_paste` | 0.0 | İki resmi karıştırma / nesneleri başka resme yapıştırma (büyük modellerde açılır) |

⚠️ **Augmentation'ın sınırı:** `scale=0.5`, ±%50 boyut değişimini kapsar. Bizim imgsz deneyinde ×2 (+%100) bu aralığın dışında kaldı ve model bozuldu.

⚠️ **Yanlış augmentation zararlıdır:** Rakam tanımada `flipud` (dikey çevirme) 6'yı 9 yapar. Sol/sağ şerit tespitinde `fliplr`, "sol şerit" etiketini yanlış yapar.

### 10.6 Hiperparametre kopya kağıdı
| Ayar | Ne zaman değiştir? |
|---|---|
| `epochs` | mAP50-95 hâlâ artıyorsa ↑ |
| `patience` | Erken durdurma: val iyileşmezse N epoch sonra dur |
| `imgsz` | Küçük nesneler kaçıyorsa ↑ (ve **aynı** imgsz ile deploy et) |
| `batch` | GPU belleği yetmiyorsa ↓ (`-1` = otomatik) |
| `lr0`, `optimizer` | Genelde `auto` bırak |
| `freeze` | Çok az veri varsa backbone'u dondur (`freeze=10`) |
| `cache` | Veriyi RAM'e al, eğitimi hızlandır |

### 10.7 Eğitim log'u okuma kontrol listesi
- [ ] Doğru cihaz mı? (`CUDA:0` mı, `CPU` mu?)
- [ ] `nc` doğru mu? (`Overriding ... nc=80 with nc=2`)
- [ ] Transfer var mı? (`Transferred 493/499 items`)
- [ ] Veri doğru okundu mu? (resim sayısı, `backgrounds`, `corrupt`, uyarılar)
- [ ] Kayıplar düşüyor mu? val kaybı yükselmeye başladı mı? (overfitting)
- [ ] mAP50 ve mAP50-95 ne zaman doydu?
- [ ] `best.pt` hangi epoch'tan?

<details><summary>✅ Kendini test et</summary>

1. 1. epoch'ta mAP50 = 0.96. Bu neyi gösterir? → **Transfer learning** çalışıyor; görev COCO'dan öğrenilen özelliklerle kolayca çözülebiliyor.
2. train loss sürekli düşüyor, val loss 30. epoch'tan sonra yükseliyor. Hangi ağırlıkları kullanırsın? → **best.pt** (val skorunun en iyi olduğu epoch); bir sonraki eğitimde daha fazla veri/augmentation veya `patience`.
</details>

---

## 🦸 Seviye 11 — Metrikler

### 11.1 Tespitte TP, FP, FN
| | Tanım |
|---|---|
| **TP** (doğru pozitif) | Tahmin edilen kutu, aynı sınıftan bir gerçek kutuyla **IoU ≥ eşik** (örneğin 0.5) |
| **FP** (yanlış pozitif) | Eşleşecek gerçek kutu yok (hayalet kutu) ya da aynı nesneye **ikinci** kutu |
| **FN** (yanlış negatif) | Gerçek kutu var ama hiçbir tahmin eşleşmedi (kaçırma) |
| ~~TN~~ | Tespitte **tanımsız**: "Kutu olmayan her yer" sayılamaz |

### 11.2 Precision, Recall, F1
```
Precision = TP / (TP + FP)    → "Kutu çizdiğimde ne kadar haklıyım?"
Recall    = TP / (TP + FN)    → "Var olan nesnelerin ne kadarını buldum?"
F1        = 2·P·R / (P + R)   → İkisinin dengesi
```

### 11.3 PR eğrisi, AP, mAP
Güven eşiğini 1'den 0'a indirdikçe recall artar, precision genelde azalır. Bu noktaların çizgisi **PR eğrisi**dir.
```
precision
 1.0 ├──────────────╮
     │   AP = alan  ╰──╮
     │                 ╰─╮
     └───────────────────┴── recall
```
| Metrik | Tanım |
|---|---|
| **AP** | Tek sınıfın PR eğrisinin altındaki alan |
| **mAP** | Tüm sınıfların AP ortalaması |
| **mAP50** | IoU eşiği 0.5 → "nesneyi **buldu** mu?" |
| **mAP50-95** | IoU 0.50, 0.55, …, 0.95 (10 eşik) ortalaması → "kutu **ne kadar sıkı**?" (COCO'nun ana metriği) |

### 11.4 Eşik seçimi ve bizim deneyimiz
| | conf=0.001 | conf=0.5 |
|---|---|---|
| Yanlış alarm | ~70 | 0 |
| Kaçırma | 0 | 1 |
| mAP50-95 | 0.897 | 0.879 (PR eğrisi kesildi) |

- **mAP'i raporlarken** düşük eşik (Ultralytics val varsayılanı 0.001).
- **Uygulamada** kendi eşiğini seç: genelde F1'i maksimum yapan eşik iyi bir başlangıçtır (Ultralytics `F1_curve.png` çizer).
- **Maliyete göre karar ver:** Güvenlik sisteminde kaçırma pahalıdır → eşik ↓. Sayma sisteminde yanlış alarm sayıyı şişirir → eşik ↑.

### 11.5 Confusion matrix (tespit versiyonu)
| | Gerçek: A | Gerçek: B | Gerçek: **background** |
|---|---|---|---|
| Tahmin: A | ✅ | A↔B karışıklığı | **FP** (hayalet A) |
| Tahmin: B | B↔A karışıklığı | ✅ | **FP** (hayalet B) |
| Tahmin: **background** | **FN** (kaçan A) | **FN** (kaçan B) | — |

<details><summary>✅ Kendini test et</summary>

Model mAP50 = 0.99, mAP50-95 = 0.60. Ne söylersin? → Model nesneleri **buluyor** ama kutuları **gevşek** yerleştiriyor. Daha uzun eğitim, daha iyi etiket kalitesi (etiketler de gevşek olabilir!) veya daha büyük imgsz düşünülebilir.
</details>

---

## 🦸 Seviye 12 — Veri seti stratejisi

> "Modelden önce veri." Bir YOLO projesinin başarısının çoğu veride belirlenir.

| Kural | Neden? |
|---|---|
| **Kullanılacak ortamda topla** | Domain shift: Laboratuvarda eğitilen model sahada bozulur. Aynı kamera, ışık, açı, mesafe |
| **Sınıf başına ≥ 100–200 nesne örneği** ile başla | Bizim 225 resim / 675 kutu küçük ama kolay bir görev için yetti |
| **%5–10 background resmi** ekle | "Boş sahneye kutu çizme" öğretir; bizim 12 background resmi, klavyeyi box sanmayı engelledi |
| **Etiketler sıkı ve tutarlı** olsun | Gevşek etiket → gevşek kutu → düşük mAP50-95. Kısmen görünen nesneleri ya hep etiketle ya hiç |
| **Her nesneyi etiketle** | Etiketlenmemiş bir nesne, modele "bu bir arka plan" diye öğretilir |
| **Split'i sızıntısız yap** | Aynı videonun ardışık kareleri hem train'de hem val'de olursa metrikler yalan söyler |
| **Val/test yeterince büyük** olsun | 13 resimlik val'de tek hata metrikleri %1.5 oynatır |
| **Hataları tek tek aç** (error analysis) | FP/FN örneklerine bakmak, hangi verinin eksik olduğunu söyler |

**Döngü:** Topla → etiketle → eğit → hataları incele → eksik durumların verisini topla → tekrar (**active learning**'in temeli).

<details><summary>✅ Kendini test et</summary>

Model gündüz mükemmel, gece kötü. İlk ne yaparsın? → Gece verisi topla ve etiketle. HSV augmentation bir miktar yardım eder ama **gerçek veriyi** yerine koyamaz.
</details>

---

## 🦸 Seviye 13 — Export ve deploy

### 13.1 Format haritası
| Format | Komut | Nerede parlar? |
|---|---|---|
| PyTorch `.pt` | — | Eğitim, hızlı deneme |
| **NCNN** | `format=ncnn` | **ARM CPU**: Raspberry Pi, Android |
| ONNX | `format=onnx` | Taşınabilir ara format; ONNX Runtime |
| OpenVINO | `format=openvino` | Intel CPU/GPU/NPU |
| TensorRT | `format=engine` | NVIDIA GPU, Jetson |
| LiteRT (TFLite) | `format=tflite` | Mobil, Coral Edge TPU (`edgetpu`) |
| CoreML | `format=coreml` | Apple cihazlar |
| ExecuTorch | — | PyTorch'un edge çözümü (kitabın Bölüm 8'i) |

### 13.2 Altın kurallar
1. **imgsz kuralı:** `eğitim = export = çıkarım`. NCNN/ONNX/TFLite'ta giriş boyutu modele **gömülür**.
2. **Metadata'yı kontrol et:** `imgsz`, `names`, `task`, `end2end`.
3. **Export sonrası doğrula:** Aynı resimlerde **aynı tespitler** (bizim `bench_box_wheel.py`: 10/10 aynı, skor farkı ≤ 0.01).

### 13.3 Hassasiyet ve quantization
| Tip | Boyut (YOLO11n) | Hız | Doğruluk |
|---|---|---|---|
| fp32 | ~10 MB | Referans | Referans |
| fp16 (`half=True`) | ~5 MB | GPU'da ↑ | Neredeyse aynı |
| int8 (`int8=True`, kalibrasyon verisi ile) | ~3 MB | NPU/Edge TPU'da çok ↑ | Biraz ↓ — **mutlaka ölç** |

**Kalibrasyon:** int8'e çevirirken her katmanın değer aralığını (scale, zero_point) bulmak için birkaç yüz temsilî resim kullanılır. Kalibrasyon verisi gerçek kullanım verisine ne kadar benzerse doğruluk kaybı o kadar az olur.

<details><summary>✅ Kendini test et</summary>

Ekip arkadaşın modeli `yolo export model=best.pt format=ncnn` ile export etmiş (imgsz yok), model 320 ile eğitilmiş. Ne beklersin? → 640'ta export edilir: çıkış `(1, 6, 8400)`; ×4 yavaş, düşük güven, **parça/hayalet kutular**. Kitaptaki "113 box, 164 wheel" örneği.
</details>

---

## 🦸 Seviye 14 — Edge'de hız

### Bizim Pi 5 yolculuğumuz (YOLO11n, box/wheel)
| Adım | Çıkarım | FPS |
|---|---|---|
| PyTorch, imgsz=640 (yanlış!) | 385 ms | 2.6 |
| PyTorch, imgsz=320 | 103 ms | 9.6 |
| **NCNN, imgsz=320** | **14.6 ms** | **61** |

### Hızlandırma kontrol listesi (etkiye göre sıralı)
1. **Doğru imgsz:** Piksel ×4 → hesap ×4. Küçük imgsz ile **eğit**.
2. **Doğru format/motor:** Pi'de NCNN (bizde ×7.1).
3. **Küçük model:** n < s < m.
4. **Thread sayısı:** Pi 5'te 4 çekirdek.
5. **Quantization:** int8, özellikle NPU/TPU varsa.
6. **Pipeline:** Kamera yakalama, çıkarım ve çizimi ayrı thread'lere böl.
7. **Profiling:** Önce ölç. Model 15 ms iken sistem 15 FPS ise darboğaz model **değil**.

### Amdahl yasası
```
Toplam hızlanma = 1 / ((1 − p) + p / s)
p: hızlandırılan kısmın oranı,  s: o kısmın hızlanması
```
Bizde: inference ×7.1 hızlandı ama toplam ×6.4, çünkü pre+post (1.7 ms) hızlanmadı. Model hızlandıkça **diğer her şey** önem kazanır.

### Isınma ve adil ölçüm
İlk çağrılar bellek ayırma, önbellek ve en iyi kernel seçimi yüzünden yavaştır. Kural: **3–5 ısınma**, sonra **çok sayıda** ölçüm, ortalama ± standart sapma.

<details><summary>✅ Kendini test et</summary>

Model 15 ms, ama canlı web akışı 15 FPS. Neden ve ne yaparsın? → Darboğaz model değil (15 ms → 66 FPS yeterdi). Kamera yakalama, renk çevirme, çizim, JPEG kodlama, ağ. **Profiling** yap, yakalama ile çıkarımı paralelleştir, JPEG kalitesini/çözünürlüğü düşür.
</details>

---

## 🦸 Seviye 15 — Takip ve sayma (Bölüm 7 önizlemesi)

**Tespit** her kareyi **bağımsız** işler: Bir arı 30 karede görünürse, tespit "30 arı" der. **Sayma** için aynı nesneyi kareler arasında **tanımak** gerekir: **Takip (tracking).**

```
Kare 1: [arı#1] [arı#2]        Tracker her tespite bir ID verir
Kare 2: [arı#1] [arı#2]        ve kareler arasında eşleştirir:
Kare 3: [arı#1]      [arı#3]   • IoU / hareket tahmini (Kalman filtresi)
                               • ByteTrack, BoT-SORT
```
```python
results = model.track(source=0, tracker="bytetrack.yaml", persist=True)
ids = results[0].boxes.id      # her kutunun takip ID'si
```
**Sayma yöntemleri:**
- **Benzersiz ID sayısı:** Kaç farklı arı görüldü?
- **Çizgi geçişi:** Bir ID'nin merkezi sanal bir çizgiyi hangi yönde geçti? (kovana giriş/çıkış)
- **Bölge sayımı:** Şu an bölgede kaç nesne var?

⚠️ Bugünkü imgsz dersinin sayma için anlamı: **Bir nesneye iki kutu = iki sayım.** Doğru imgsz ve doğru eşik, sayma doğruluğunun ön şartıdır.

---

## 📜 Seviye 16 — YOLO soy ağacı

| Sürüm | Yıl | Kim? | Ne getirdi? |
|---|---|---|---|
| **YOLOv1** | 2016 | Redmon et al. | Tek geçişte tespit; 7×7 ızgara |
| YOLOv2 (YOLO9000) | 2017 | Redmon & Farhadi | Anchor'lar, BatchNorm, çok ölçekli eğitim |
| YOLOv3 | 2018 | Redmon & Farhadi | Darknet-53, **3 ölçekte** tahmin |
| YOLOv4 | 2020 | Bochkovskiy et al. | CSP backbone, **mosaic**, "bag of freebies" |
| YOLOv5 | 2020 | Ultralytics | PyTorch, kolay kullanım, export ekosistemi |
| YOLOv6 / v7 | 2022 | Meituan / Wang et al. | Endüstriyel optimizasyon, re-parametrizasyon |
| **YOLOv8** | 2023 | Ultralytics | **Anchor-free**, decoupled head, C2f, DFL; detect/seg/pose/cls tek API |
| YOLOv9 | 2024 | Wang et al. | PGI, GELAN |
| YOLOv10 | 2024 | Tsinghua | **NMS'siz** eğitim (bire-bir atama) |
| **YOLO11** | 2024 | Ultralytics | C3k2, C2PSA (attention); daha az parametreyle daha yüksek mAP |
| YOLOv12 | 2025 | Tian et al. | Attention merkezli mimari |
| **YOLO26** | 2025–26 | Ultralytics | **End-to-end (NMS'siz)**, DFL kaldırıldı, **edge odaklı** |

> 💡 "YOLO" bir şirketin ürünü değil, bir **fikir ailesi**. Sürüm numaraları farklı ekiplerden gelir ve her zaman "daha büyük numara = daha iyi" demek değildir. Kendi verinde ve donanımında **ölç**.

---

## 🏆 Final

### Mülakat soruları (cevaplarıyla)
<details><summary>1. YOLO neden hızlıdır?</summary>Tespiti tek bir ağ geçişinde, ızgara tabanlı yoğun tahmin olarak yapar; ayrı bir bölge önerme aşaması yoktur.</details>
<details><summary>2. (1, 84, 8400) ne demek?</summary>Batch 1; 4 kutu + 80 sınıf skoru; 640 girişte stride 8/16/32 haritalarından 80²+40²+20² = 8400 aday.</details>
<details><summary>3. Anchor-free ne demek?</summary>Hazır kutu şablonları yok; her hücre merkezi bir nokta olarak kutunun dört kenarına mesafeyi tahmin eder.</details>
<details><summary>4. NMS nasıl çalışır, zayıflıkları neler?</summary>Skora göre sırala, en iyisini tut, IoU'su eşikten büyük olanları sil, tekrarla. Seri bir algoritma (edge'de gecikme), eşik ayarı ister, yakın nesneleri silebilir veya parça kutuları ayıramaz.</details>
<details><summary>5. mAP50 ile mAP50-95 farkı?</summary>mAP50 tek IoU eşiğinde "buldu mu?"; mAP50-95, 10 eşiğin ortalamasıyla "kutu ne kadar sıkı?" sorusunu ölçer.</details>
<details><summary>6. Precision yüksek, recall düşük. İlk ne yaparsın?</summary>Güven eşiğini düşür. Yetmezse kaçırılan örneklerin verisini topla.</details>
<details><summary>7. imgsz kuralı nedir, bozulursa ne olur?</summary>Eğitim = export = çıkarım imgsz. Bozulursa ölçek uyuşmazlığı: düşük güven, parça/hayalet kutular, yavaşlık. Sessiz bir hatadır.</details>
<details><summary>8. Letterbox neden kullanılır?</summary>En-boy oranını koruyarak kare girişe sığdırmak için; düz resize nesneleri ezer.</details>
<details><summary>9. DFL nedir?</summary>Kutu kenarını tek sayı yerine 16 kutucuklu bir olasılık dağılımı olarak tahmin edip beklenen değeri almak; belirsiz kenarları daha iyi modeller.</details>
<details><summary>10. Transfer learning log'da nasıl görünür?</summary>Transferred 493/499 items, nc=80 → nc=2 ve ilk epoch'ta yüksek mAP.</details>
<details><summary>11. Overfitting'i nasıl anlarsın, ne yaparsın?</summary>train loss düşerken val loss yükselir. best.pt, daha fazla veri, augmentation, erken durdurma, daha küçük model.</details>
<details><summary>12. Export sonrası neyi doğrularsın?</summary>Hız, aynı resimlerde aynı tespitler, metadata (imgsz, names, end2end).</details>
<details><summary>13. Model 7 kat hızlandı, sistem neden 7 kat hızlanmadı?</summary>Amdahl yasası: hızlandırılmayan pre/post, kamera, çizim, ağ.</details>
<details><summary>14. Tespit ile takip farkı?</summary>Tespit kareleri bağımsız işler; takip aynı nesneye kareler boyunca aynı ID'yi verir. Sayma için takip gerekir.</details>
<details><summary>15. Edge'de YOLO'yu hızlandırmanın yolları?</summary>Küçük imgsz ile eğit, verimli motor (NCNN/TensorRT/OpenVINO), küçük model, int8, pipeline paralelliği, profiling.</details>

### Hata ayıklama tablosu
| Belirti | Olası sebep |
|---|---|
| Hiç tespit yok | Normalize yok/yanlış, BGR/RGB, eşik çok yüksek, yanlış model |
| Kutular kaymış | Letterbox geri alınmadı, koordinat formatı karıştı |
| Bir nesneye birden çok kutu | imgsz uyuşmazlığı, NMS IoU eşiği çok yüksek, NMS yapılmamış |
| Yan yana nesnelerden biri kayıp | NMS IoU eşiği çok düşük |
| Etiketler yanlış sınıf | `names` sırası eğitimdekiyle farklı |
| Export sonrası sonuçlar farklı | imgsz, int8 kalibrasyonu, ön işleme farkı |
| Laboratuvarda iyi, sahada kötü | Domain shift: sahadan veri topla |
| Pi'de çok yavaş | PyTorch kullanılıyor, imgsz büyük, ısınma yok |

### 🎓 Kahramanlık projeleri (kolaydan zora)
1. **Kendi veri setin:** 2–3 sınıf, 150+ resim, Roboflow'da etiketle, Colab'da eğit, Pi'de NCNN (Proje 3!).
2. **Canlı tespit:** Picamera2 + NCNN + Flask akışı; ekranda FPS.
3. **Elle decode:** `manual_ncnn_decode.py`'yi C++'a taşı.
4. **int8 deneyi:** Aynı modelin fp32 ve int8 halini doğruluk/hız tablosunda karşılaştır.
5. **Sayma:** Takip + çizgi geçişi ile bir masadan geçen nesneleri say (Bölüm 7).
6. **AGV entegrasyonu:** Tespit sonuçlarını (sınıf, kutu merkezi, mesafe tahmini) ROS 2 mesajı olarak yayınla.

---

## 📚 Kaynaklar
- Redmon, Divvala, Girshick, Farhadi (2016) — *You Only Look Once: Unified, Real-Time Object Detection*
- Redmon & Farhadi (2017, 2018) — *YOLO9000*, *YOLOv3: An Incremental Improvement*
- Bochkovskiy, Wang, Liao (2020) — *YOLOv4: Optimal Speed and Accuracy of Object Detection*
- Wang et al. (2024) — *YOLOv10: Real-Time End-to-End Object Detection*
- Li et al. (2020) — *Generalized Focal Loss* (DFL)
- Zheng et al. (2020) — *Distance-IoU Loss* (DIoU/CIoU)
- [Ultralytics dokümantasyonu](https://docs.ultralytics.com)
- Marcelo Rovai — *Edge AI Engineering: Hands-On with the Raspberry Pi* (2026)

**Hazırlayan:** Ebrar Kılıç — Edge AI & Embedded Systems · [github.com/ebrarklc](https://github.com/ebrarklc)
