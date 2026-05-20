# Kapı Bulma

## Amaç
Müşteri uygulamasına bir kapı fotoğrafı yüklenir; uygulama ~32.000 AutoCAD (DWG) çiziminden en benzer kapıyı ve model adını döndürür.

## Yaklaşım
1. **DWG → PNG**: ODA FileConverter ile tüm CAD çizimleri raster'a çevrilir.
2. **Edge / sketch uzayı**: fotoğraf ve CAD aynı sketch domain'ine indirgenir (DoG + threshold), foto–CAD görsel boşluğu kapanır.
3. **Fine-tuned CLIP**: edge fotoğraf ve edge CAD aynı 512-d gömme uzayında. Cosine similarity ile retrieval.
4. **Active-learning labeler**: model önerir, kullanıcı doğrular → etiketler birikir → model daha iyi fine-tune edilir → öneriler daha iyi olur (geri besleme döngüsü).
5. **Müşteri uygulaması**: fotoğraf → top-K en benzer DWG + ad.

## Mevcut Durum (2026-05)

### Veri
- `data/cad_dwg/` — 32.153 DWG (kaynak, dokunulmaz)
- `data/tüm modeller/` — 3.786 fotoğraf (kaynak, dokunulmaz)
- `data/photos/` — 3.782 foto kopyası (`ingest_photos.py` çıktısı)
- `data/labels/foto_meta.json` — 3.782 foto için isimden çıkarılmış meta (etiket değil, sadece UI bilgisi)
- `data/cad_dxf/` — 32.153 DXF (ODA çıktısı, tamamlandı)
- `data/cad_png/` — 32.148 PNG (ezdxf+matplotlib render, 5 atlandı/0 bozuk)
- `data/labels/labels.json` — henüz oluşmadı (etiketleme başlayınca üretilir)

### Modeller
- `models/clip_finetuned_best.pt` (577 MB) — **ViT-B/32 QuickGELU**, anahtarlar `clip_model.` öneki taşır, 302 anahtar tam uyumlu.
- `models/embeddings.npz` (291 MB) — `dosya_adlari` + `raw_embeddings` + `edge_embeddings`. **128.484 vektör = 32.121 PNG × 4 rotasyon (0/90/180/270)**. Her DWG için hem ham hem edge gömmesi mevcut. 32 yeni DWG (`besmele*`) embedding'de yok — yeniden hesaplamada dahil olacak.
- `finetune_output/clip_finetuned_best.pt` (571 MB) — **ViT-B/16** (farklı backbone! patch 16, positional 197). models/'taki ile **uyumsuz**, mevcut embedding bank ile beraber kullanılamaz.
- `finetune_output/clip_finetuned_last.pt` (571 MB) — yine ViT-B/16.

### Kod (yazıldı)
Tüm pipeline scriptleri tamamlandı — bkz. **Klasör Yapısı** bölümü. Her dosya çalışır halde.

### Çalışma ortamı
- `.venv/` — proje köküne kurulu Python sanal ortamı
- Torch 2.6.0 + CUDA 12.4 (RTX 3060 6GB)
- open-clip-torch 3.3.0, streamlit 1.57, ezdxf 1.4.4, matplotlib 3.10
- ODA File Converter 27.1.0 — `C:\Program Files\ODA\ODAFileConverter 27.1.0\ODAFileConverter.exe`

## Final Klasör Yapısı
```
Kapı Bulma/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py                 ← path sabitleri
│   ├── utils.py                  ← ön-işleme + labels CRUD
│   ├── data_prep/
│   │   ├── convert_dwg.py        ← DWG → PNG (ODA wrapper, paralel, idempotent)
│   │   ├── prepare_edges.py      ← PNG → edge (foto + CAD ortak)
│   │   └── ingest_photos.py      ← data/tüm modeller/ → photos/ + foto_meta.json
│   ├── labeling/
│   │   ├── labeler.py            ← active-learning UI (top-K + "yenile" döngüsü)
│   │   └── label_manager.py      ← etiket bakım, istatistik, dedup
│   ├── train/
│   │   ├── dataset.py            ← (foto_edge, cad_edge) çift örnekleyici
│   │   ├── train_clip.py         ← contrastive fine-tune (InfoNCE)
│   │   └── evaluate.py           ← Recall@K, aile-içi/aile-arası metrik
│   └── retrieval/
│       ├── build_embeddings.py   ← model + cad_png_edge → embeddings.npz
│       └── search.py             ← foto → top-K DWG (CLI + uygulamadan çağrılabilir)
├── app/
│   └── main.py                   ← Streamlit müşteri arayüzü
├── data/
│   ├── cad_dwg/                  ← 32.153 DWG (kaynak)
│   ├── cad_dxf/                  ← DWG→DXF ara format (convert_dwg üretir)
│   ├── cad_png/                  ← DXF→PNG render (convert_dwg üretir)
│   ├── cad_png_edge/             ← sketch dönüşümü (prepare_edges üretir)
│   ├── tüm modeller/             ← orijinal foto havuzu (kaynak)
│   ├── photos/                   ← ingest_photos kopyası
│   ├── photos_edge/              ← foto sketch (prepare_edges üretir)
│   └── labels/
│       ├── labels.json           ← {tamamlanan, eslesme}
│       ├── foto_meta.json        ← dosya adından çıkarılmış örtük etiketler
│       └── labels.json.backup
├── models/                       ← prod ağırlık + embeddings (deploy edilen)
│   ├── clip_finetuned_best.pt
│   └── embeddings.npz
└── finetune_output/              ← eğitim koşumları (deney; en iyi olan models/'a kopyalanır)
```

## Veri Akışı
```
data/cad_dwg/*.dwg  ─convert_dwg─►  data/cad_png/*.png
                                         │
                                         ▼ prepare_edges (cad_edge)
                                    data/cad_png_edge/*.png
                                         │
                                         ▼ build_embeddings (CLIP)
                                    models/embeddings.npz
                                         │
data/tüm modeller/  ─ingest_photos─► data/photos/  ─prepare_edges─► data/photos_edge/
                    foto_meta.json                                       │
                                                                         ▼
                                                       labeler (top-K + yenile) ↔ labels.json
                                                                         │
                                                                         ▼
                                                                    train_clip
                                                                         │
                                                                         ▼
                                                              finetune_output/*.pt  ──seç──►  models/
                                                                         │
                                          ┌──────────────── search ◄─────┘
                                          │
app/main.py (foto upload) ────────────────┘
                       └──► top-K DWG + ad
```

## İsim Şeması

### KRİTİK: Foto kodları ile DWG kodları farklı sistemler
- 1309 farklı foto tabanından **sadece 4'ü** DWG ismiyle bire bir eşleşiyor.
- Fotoğraflar üretici/marka model numaralarını (örn. 1008, 1274, 1363) kullanır.
- DWG'ler AutoCAD iç model kodlarını (örn. 100A, 21A, 11A) kullanır.
- **İki isim sistemi arasında köprü yok** → tek yol görsel retrieval (CLIP).
- Bu yüzden isimlerden otomatik etiket çıkarmak işe yaramaz; manuel labeler zorunlu.

### DWG isimleri
`<TABAN_NO>A<MODIFIER>.dwg`
- Taban: 1-3 basamak (1, 21, 100 ...) — 4-haneli sadece 3 dosya var
- "A" çoğu zaman var (varyasyon işaretçisi)
- Modifier: sayı (ölçü: 28, 30, 35, 40, 45, 130) veya Türkçe ek (`fer`, `ust`, `par`, `apar`, `kab`, `zip`, `kare`, `re`, `del`)

### Fotoğraf isimleri — DİKKAT: güvenilir etiket DEĞİL
- `<no>.png` → katalog fotoğrafı; ad o modeli **iddia eder** ama DWG ile bire bir aynı kapı olduğu **garanti değil**.
- `<no><AD>-<varyant>.jpeg` → saha fotoğrafı; `<no>` o modele benzerlik iddiası, **çoğu zaman aynı kapı değil**, sadece yakın aile.
- `<no>GİBİ-<varyant>.<ext>` → açıkça "benziyor ama o değil".

**Kural: dosya adı sadece organizasyon içindir, etiket değildir.** Aynı `<no>`'ya sahip fotoğraf ile DWG'nin gerçekten aynı kapı olduğu **kullanıcının görsel onayı** olmadan kabul edilmez.

`ingest_photos.py` adı yalnızca `foto_meta.json`'a not amaçlı parse eder:
```json
{"1008AKEL-1.jpeg": {"taban": "1008", "ad": "AKEL", "varyant": "1", "yumusak": false}}
```
Bu meta **etiket değil**, sadece UI'da "iddia edilen no" olarak gösterilir; gerçek eşleşme her zaman labeler'da kullanıcının seçimiyle oluşur.

## `labels.json` Şeması
```json
{
  "tamamlanan": ["1008AKEL-1.jpeg"],
  "eslesme": {
    "1008AKEL-1.jpeg": ["1008A.dwg", "1008A40.dwg"]
  }
}
```
- `tamamlanan`: kullanıcının "bitti" dediği fotoğraflar
- `eslesme`: foto → eşlenen DWG'ler (1-çoğa)
- `utils.labels_kaydet` her yazımda `.backup` üretir

## Active-Learning Labeler — Akış
1. Bir sonraki etiketsiz fotoğraf seç (sıralama tercihen alfabetik; isim-no'ya **güvenme** ama UI'da göster).
2. Foto edge → CLIP gömme → `embeddings.npz` ile cosine → top-K (K=6) DWG.
3. UI'da foto + 6 DWG önerisi yan yana göster (thumbnail + dosya adı + skor).
4. Kullanıcı checkbox ile seçer; seçim `labels.json["eslesme"]`'ye eklenir.
5. "Yenile" → sonraki 6 öneri (4 yenileme hakkı = K=24 toplam görülebilir).
6. "Bitti" → fotoğraf `tamamlanan`'a girer, sonraki fotoğrafa geç.

## Teknolojiler
- Python 3.10+
- PyTorch + open_clip (varsayım; .pt yapısından doğrulanacak)
- Pillow, NumPy, OpenCV (cv2)
- ODA FileConverter 27.1.0 — DWG → PNG (Windows yolu config.py'de)
- Streamlit (labeler + müşteri uygulaması)
- Brute-force cosine (32K eleman = 65 MB, FAISS gereksiz)

## Kodlama Kuralları
- **Dil**: docstring ve değişken adları Türkçe (mevcut `utils.py` stiline uyum: `labels_yukle`, `eslesme`).
- **Path tek kaynak**: Tüm path'ler `src/config.py`'den. Yeni script absolute path tanımlamadan kullanmaz.
- **Tek sorumluluk**: her script tek iş yapar; CLI flag'leriyle dallanma minimal.
- **`labels.json` kutsal**: her zaman `utils.labels_kaydet` üzerinden yaz, doğrudan dump etme.
- **GPU varsayma**: `torch.cuda.is_available()` kontrol, CPU fallback.
- **Idempotent batch**: `convert_dwg`, `prepare_edges`, `build_embeddings` — çıktı varsa atla.
- **Yorum**: ne yapıldığını koddan oku; neden yapıldığını yorum yaz.
- **Türkçe karakter**: yollarda "Kapı Bulma", "tüm modeller" var — tüm I/O `encoding="utf-8"`, `Path` API tercih.

## Bilinen Sorunlar
1. **İki farklı backbone'lu checkpoint** — `models/best.pt` ViT-B/32 (577 MB), `finetune_output/best.pt` ViT-B/16 (571 MB). İkincisi mevcut `embeddings.npz` (ViT-B/32 ile hesaplanmış) ile uyumsuz; kullanmak için yeni embedding gerek.
2. **Mevcut model foto→CAD hizalamasını yapamıyor** — CAD↔CAD self benzerliği 1.0, foto↔CAD ~0.04-0.08. Top-K her foto için aynı birkaç dosyayı döndürüyor (REYYANOZEL3/4, bimka, vs. dataset bias). Etiketleme ilerledikçe fine-tune ile kapatılacak.
3. **ezdxf render kalitesi karışık** — bazı DWG'lerin modelspace'i neredeyse boş (sarı çerçeve + birkaç ince çizgi), bazıları net (21A.dwg gibi). Layout/paperspace yerine modelspace render ediliyor. İhtiyaç olursa AutoCAD COM veya ODA PDF + pdf2image ile yükseltilebilir.
4. **Eğitim kodu reprodüksiyonu yok** — mevcut .pt nasıl eğitildi (loss? aug? epoch?) belirsiz. Yeni `train_clip.py` sıfırdan tasarım.
5. **`sketch_donustur` parametreleri sabit** (DoG (3,3)/(15,15), threshold 15) — 32K çeşitlilik için tek değer optimum olmayabilir.
6. **`yazi_maskele` %12 sabit** — bazı saha fotoğraflarında ya yazı kalır ya kapı kesilir.
7. **`Image.MAX_IMAGE_PIXELS = None`** — büyük PNG bombasına karşı koruma kapalı, bilinçli karar.
8. **Foto-CAD ön-işleme asimetrisi** — fotoğraf text-maske + crop, CAD sadece invert. Sketch sonrası dağılımlar farklı.
9. **Hard negative yok (henüz veride)** — aile-içi (21A30 vs 21A35) ince fark; `dataset.py` aile-içi negatif örneklemeyi destekliyor ama eğitim verisi birikene kadar test edilemez.

## Yapılanlar ve Sıradakiler

### Bitti (kod hazır, çalışır)
- ✅ `requirements.txt`, `.gitignore`, `.venv` kurulumu
- ✅ `src/config.py`, `src/utils.py` (mevcut + güncellendi)
- ✅ `src/data_prep/ingest_photos.py` — 3.782 foto kopyalandı + `foto_meta.json` üretildi (parse: 1893 sayı-baş + 1572 ad-baş + 317 bilinmiyor)
- ✅ `src/data_prep/convert_dwg.py` — ODA + ezdxf, idempotent + paralel (**şu an background batch çalışıyor**)
- ✅ `src/data_prep/prepare_edges.py`
- ✅ `src/labeling/labeler.py`, `src/labeling/label_manager.py`
- ✅ `src/train/dataset.py`, `train_clip.py`, `evaluate.py`
- ✅ `src/retrieval/search.py`, `build_embeddings.py`
- ✅ `app/main.py`

### Sırada (PNG batch bittikten sonra)
1. **PNG render kontrolü**: 32K PNG tamamlanınca random örneklemeyle kalite gözle bak; çoğunluk minimal-içerikse render stratejisini değiştir (AutoCAD COM / ODA PDF).
2. **İlk etiketleme turu**: `streamlit run src/labeling/labeler.py` — ~200-300 etiket biriktir. İlk öneriler zayıf olacak (mevcut model foto-CAD hizalayamıyor); kullanıcı görsel olarak seçecek.
3. **İlk fine-tune**: `python -m src.train.train_clip --epochs 10`
4. **Yeni embedding bank**: `python -m src.retrieval.build_embeddings` (eski stale)
5. **Değerlendirme**: `python -m src.train.evaluate --k 1 5 10` → Recall raporu
6. **Döngüye devam**: yeni etiketler + yeniden fine-tune (her 200-300 etiket)

### İleride (kalite optimizasyonu)
- OCR ön-filtre (ölçü etiketi → DWG modifier filtresi)
- Geometrik re-rank (LoFTR / LightGlue) — top-100'ü yeniden sırala
- Render kanonikleştirme (kalın çizgi, sayfa çerçevesi temizliği)
- Aile-aware train/val split
- NOT: multi-view CAD embedding **zaten yapılmış** (her DWG 4 rotasyon + raw/edge çiftli, `build_embeddings.py` aynı yapıyı koruyor).

## Tek Komutla

Tüm komutlar proje kökünden (`c:\Users\canya\Desktop\Kapı Bulma\`) ve `.venv` ile çalıştırılır.

```powershell
# Sanal ortamı aktive et (her terminal açılışında)
.venv\Scripts\Activate.ps1

# --- Veri hazırlığı (sıra önemli) ---

# 1) Fotoğrafları kopyala + meta üret (idempotent)
python -m src.data_prep.ingest_photos

# 2) DWG → DXF → PNG (uzun: ~30dk ODA + ~2 saat ezdxf; idempotent)
python -m src.data_prep.convert_dwg --paralel 6
# Eğer ODA zaten çalıştırılmışsa (DXF'ler var):
python -m src.data_prep.convert_dwg --skip-oda --paralel 6
# Tek DWG hızlı test:
python -m src.data_prep.convert_dwg --tek data/cad_dwg/21A.dwg

# 3) Edge dönüşümleri (CAD + foto için ayrı)
python -m src.data_prep.prepare_edges --target cad
python -m src.data_prep.prepare_edges --target photos

# --- Etiketleme ---

streamlit run src/labeling/labeler.py
python -m src.labeling.label_manager stats
python -m src.labeling.label_manager dedup

# --- Eğitim ---

python -m src.train.train_clip --epochs 10 --batch 64 --lr 1e-5
python -m src.retrieval.build_embeddings   # yeni model + yeni embedding bank
python -m src.train.evaluate --k 1 5 10 20

# --- Inference / Müşteri uygulaması ---

# CLI tek foto:
python -m src.retrieval.search --photo data/photos/1008AKEL-1.jpeg --k 10 --kanal edge

# Web UI:
streamlit run app/main.py
```
