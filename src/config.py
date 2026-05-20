"""
Merkezi konfigürasyon — tüm path ve sabitler buradan.
Path değiştirmek için sadece bu dosyayı düzenle.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent  # src/ -> proje kökü

# ── Ham veri (kaynak, dokunulmaz) ─────────────────────────────
CAD_DWG       = ROOT / "data" / "cad_dwg"             # 32K DWG
PHOTOS_RAW    = ROOT / "data" / "tüm modeller"        # orijinal foto havuzu

# ── Üretilmiş veri ────────────────────────────────────────────
CAD_PNG       = ROOT / "data" / "cad_png"             # ODA çıktısı
CAD_PNG_EDGE  = ROOT / "data" / "cad_png_edge"        # sketch dönüşümlü CAD
PHOTOS        = ROOT / "data" / "photos"              # ingest edilmiş fotolar
PHOTOS_EDGE   = ROOT / "data" / "photos_edge"         # sketch dönüşümlü fotolar

# ── Etiket / meta ─────────────────────────────────────────────
LABELS_DIR       = ROOT / "data" / "labels"
LABELS_FILE      = str(LABELS_DIR / "labels.json")
FOTO_META_FILE   = str(LABELS_DIR / "foto_meta.json")
AUTO_LABELS_FILE = str(LABELS_DIR / "auto_labels.json")   # geriye dönük uyum

# ── Model + embedding ─────────────────────────────────────────
MODELS_DIR = ROOT / "models"
MODEL_BEST = str(MODELS_DIR / "clip_finetuned_best.pt")
EMBEDDINGS = str(MODELS_DIR / "embeddings.npz")

# ── Fine-tune çıktısı (eğitim koşumları) ──────────────────────
FINETUNE_OUTPUT  = ROOT / "finetune_output"
MODEL_BEST_NAME  = "clip_finetuned_best.pt"
MODEL_LAST_NAME  = "clip_finetuned_last.pt"

# ── Harici araçlar ────────────────────────────────────────────
ODA_CONVERTER = r"C:\Program Files\ODA\ODAFileConverter 27.1.0\ODAFileConverter.exe"

# ── Model parametreleri ───────────────────────────────────────
CLIP_BACKBONE = "ViT-B-32-quickgelu"  # OpenAI checkpoint'leri QuickGELU kullanır; .pt incelemesinden doğrulandı
CLIP_PRETRAINED = "openai"
IMG_SIZE = 224                        # CLIP girdi boyutu
EMBED_DIM = 512                       # ViT-B/32 için
