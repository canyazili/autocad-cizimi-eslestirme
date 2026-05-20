# Kapı Bulma

Bir kapı fotoğrafından, ~32.000 AutoCAD (DWG) çizimi arasından **en benzer kapıyı ve model adını** bulan görsel retrieval sistemi.

Fotoğraf ve CAD çizimi ortak bir **edge / sketch uzayında** karşılaştırılır; fine-tuned **CLIP** modeli ile 512-d gömme uzayında cosine similarity kullanılarak en yakın eşleşmeler döndürülür.

## Pipeline

```
DWG → PNG (siyah zemin, yazı temizlenmiş)
       │
       ▼
   CLIP edge encoding
       │
       ▼
embeddings.npz (32K × 4 rotasyon × 2 kanal)
       │
foto ──┴── cosine similarity ──► top-K DWG + ad
```

1. **DWG → PNG**: ODA File Converter ile DXF, ardından ezdxf + matplotlib ile siyah-zemin/beyaz-çizgi render (TEXT/MTEXT entity'leri temizlenmiş).
2. **Edge dönüşümü**: foto ve CAD aynı sketch domain'ine (DoG + threshold) indirgenir.
3. **CLIP fine-tune**: foto-CAD eşleşmeleri ile contrastive (InfoNCE) eğitim.
4. **Active-learning labeler**: model önerir → kullanıcı doğrular → etiketler birikir → periyodik fine-tune.
5. **Müşteri uygulaması**: foto yükle → top-K en benzer DWG.

## Kurulum

```powershell
# Sanal ortam
python -m venv .venv
.venv\Scripts\Activate.ps1

# Bağımlılıklar (CUDA 12.4 için)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Harici araç**: [ODA File Converter 27.1.0](https://www.opendesign.com/guestfiles/oda_file_converter) — DWG↔DXF dönüşümü için (Windows yolu `src/config.py`'de).

## Veri

Bu repo veri içermez (büyük boyut nedeniyle). Çalıştırmak için:

- `data/cad_dwg/` — DWG dosyaları (32K adet test edildi)
- `data/tüm modeller/` — kapı fotoğraf havuzu
- `models/clip_finetuned_best.pt` — fine-tune CLIP ağırlığı

Etiketler (`data/labels/labels.json`) repoda mevcut — kendi etiket havuzunla genişletebilirsin.

## Kullanım

```powershell
# --- Veri hazırlığı (tek seferlik, ~1-2 saat) ---
python -m src.data_prep.ingest_photos
python -m src.data_prep.convert_dwg --paralel 6
python -m src.retrieval.build_embeddings

# --- Etiketleme (Streamlit UI) ---
streamlit run src/labeling/labeler.py

# --- Eğitim (~200-300 etiket sonrası) ---
python -m src.train.train_clip --epochs 10
python -m src.retrieval.build_embeddings  # yeni modelle yeniden embed
python -m src.train.evaluate --k 1 5 10

# --- Müşteri uygulaması ---
streamlit run app/main.py

# --- CLI inference ---
python -m src.retrieval.search --photo data/photos/foo.jpg --k 10
```

## Mimari & Karar Defteri

Proje mimarisi, kodlama kuralları, bilinen sorunlar ve detaylı pipeline için: **[CLAUDE.md](CLAUDE.md)**

## Teknolojiler

- Python 3.10+, PyTorch 2.6 + CUDA 12.4
- open-clip-torch 3.3 (ViT-B/32 QuickGELU)
- Pillow, OpenCV, NumPy
- ezdxf 1.4 + matplotlib (DXF→PNG render)
- Streamlit (labeler + müşteri arayüzü)
- ODA File Converter 27.1.0 (DWG↔DXF)
