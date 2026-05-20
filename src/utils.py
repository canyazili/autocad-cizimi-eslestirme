"""
Ortak yardımcı fonksiyonlar.
Tüm scriptler (labeler, train, label_manager) buradan import eder.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# src/ dizinini path'e ekle — config import için
sys.path.insert(0, str(Path(__file__).parent))
from config import LABELS_FILE, AUTO_LABELS_FILE  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

# =========================================================
# SABİTLER
# =========================================================
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
YAZI_SERIT_ORAN = 0.12

# =========================================================
# DOSYA
# =========================================================
def is_image_file(name: str) -> bool:
    return name.lower().endswith(SUPPORTED_EXTS)


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


# =========================================================
# GÖRÜNTÜ ÖN İŞLEME
# =========================================================
def yazi_maskele(img: Image.Image, oran: float = YAZI_SERIT_ORAN) -> Image.Image:
    if oran <= 0:
        return img
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, int(h * (1 - oran)), w, h], fill=(0, 0, 0))
    return img


def merkez_crop(img: Image.Image, alt_oran: float = 0.95) -> Image.Image:
    w, h = img.size
    yeni_h = int(h * alt_oran)
    ust = max((h - yeni_h) // 2, 0)
    return img.crop((0, ust, w, ust + yeni_h))


def foto_raw(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img = yazi_maskele(img)
    img = merkez_crop(img, 0.95)
    return img


def cad_raw(img: Image.Image) -> Image.Image:
    return ImageOps.invert(img.convert("RGB"))


def sketch_donustur(img: Image.Image) -> Image.Image:
    import cv2
    gray = np.array(img.convert("L"))
    blur1 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
    dog = cv2.normalize(cv2.subtract(blur1, blur2), None, 0, 255, cv2.NORM_MINMAX)
    _, t = cv2.threshold(dog, 15, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(t)
    return Image.fromarray(inv).convert("RGB")


def foto_edge(img: Image.Image) -> Image.Image:
    return sketch_donustur(foto_raw(img))


def cad_edge(img: Image.Image) -> Image.Image:
    return sketch_donustur(cad_raw(img))


# =========================================================
# LABELS (JSON)
# =========================================================
def labels_yukle() -> dict:
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                d.setdefault("tamamlanan", [])
                d.setdefault("eslesme", {})
                return d
        except Exception as e:
            print(f"[UYARI] labels.json okunamadı: {e}. Yedek aranıyor...")
            backup = LABELS_FILE + ".backup"
            if os.path.exists(backup):
                try:
                    with open(backup, "r", encoding="utf-8") as f:
                        d = json.load(f)
                        d.setdefault("tamamlanan", [])
                        d.setdefault("eslesme", {})
                        print("[UYARI] Yedekten yüklendi.")
                        return d
                except Exception:
                    pass
    return {"tamamlanan": [], "eslesme": {}}


def labels_kaydet(tamamlanan, eslesme=None):
    mevcut = labels_yukle()
    mevcut["tamamlanan"] = tamamlanan

    if eslesme:
        for foto, autocad_listesi in eslesme.items():
            autocad_listesi = list(dict.fromkeys(autocad_listesi))
            if foto in mevcut["eslesme"]:
                onceki = mevcut["eslesme"][foto]
                mevcut["eslesme"][foto] = list(dict.fromkeys(onceki + autocad_listesi))
            else:
                mevcut["eslesme"][foto] = autocad_listesi

    # Kaydetmeden önce mevcut dosyayı yedekle
    if os.path.exists(LABELS_FILE):
        shutil.copy2(LABELS_FILE, LABELS_FILE + ".backup")

    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(mevcut, f, ensure_ascii=False, indent=2)


def labels_autocad_sil(foto_adi, autocad_adi):
    data = labels_yukle()
    if foto_adi in data["eslesme"]:
        if autocad_adi in data["eslesme"][foto_adi]:
            data["eslesme"][foto_adi].remove(autocad_adi)
        if not data["eslesme"][foto_adi]:
            del data["eslesme"][foto_adi]
    if os.path.exists(LABELS_FILE):
        shutil.copy2(LABELS_FILE, LABELS_FILE + ".backup")
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
