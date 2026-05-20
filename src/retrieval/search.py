"""
Foto → top-K DWG retrieval.

Mevcut models/embeddings.npz (32K DWG × 4 rotasyon × {raw, edge}) ile çalışır.
Verilen bir fotoğraf için:
  1. foto_raw + foto_edge versiyonlarını hazırla
  2. CLIP ile her iki gömme vektörünü hesapla
  3. embeddings.npz'deki raw/edge gömmelerle ayrı ayrı cosine
  4. Her DWG için 4 rotasyon arasından en yüksek skoru al
  5. Raw + edge skorlarını birleştir (ortalama veya max), top-K döndür

Kullanım:
    python -m src.retrieval.search --photo data/photos/1001.png --k 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    MODEL_BEST, EMBEDDINGS, CLIP_BACKBONE, CLIP_PRETRAINED, IMG_SIZE,
)
from utils import foto_raw, foto_edge, load_rgb  # noqa: E402


# ──────────────────────────────────────────────────────────────
# MODEL YÜKLEME
# ──────────────────────────────────────────────────────────────
def model_yukle(ckpt_path: str, backbone: str = CLIP_BACKBONE, pretrained: str = CLIP_PRETRAINED):
    """
    Fine-tuned CLIP checkpoint'i yükle. open_clip formatı bekleniyor.
    Birden çok kayıt şekline (full state_dict, {"model": ...}, {"state_dict": ...}) toleranslı.
    """
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break

    # Önekleri sıyır (custom wrapper veya DataParallel'den)
    def sıyır(k: str) -> str:
        for ön in ("module.", "clip_model.", "model."):
            if k.startswith(ön):
                return k[len(ön):]
        return k
    state = {sıyır(k): v for k, v in state.items()}

    eksik, fazla = model.load_state_dict(state, strict=False)
    if eksik:
        print(f"[UYARI] eksik anahtarlar (ilk 5): {eksik[:5]} / toplam {len(eksik)}")
    if fazla:
        print(f"[UYARI] fazladan anahtarlar (ilk 5): {fazla[:5]} / toplam {len(fazla)}")

    model.eval().to(device)
    return model, preprocess, device


# ──────────────────────────────────────────────────────────────
# FOTO GÖMME
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def foto_embed(model, preprocess, img: Image.Image, device: str) -> np.ndarray:
    """Tek bir PIL görüntüsünü CLIP ile gömüp L2-normalize edilmiş 512-d vektör döndür."""
    x = preprocess(img).unsqueeze(0).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy().astype(np.float32)


def foto_iki_gomme(model, preprocess, foto_yolu: str, device: str):
    """Foto için (raw, edge) gömme çifti üret."""
    img = load_rgb(foto_yolu)
    raw_img = foto_raw(img)
    edge_img = foto_edge(img)
    raw_vec = foto_embed(model, preprocess, raw_img, device)
    edge_vec = foto_embed(model, preprocess, edge_img, device)
    return raw_vec, edge_vec


# ──────────────────────────────────────────────────────────────
# RETRIEVAL
# ──────────────────────────────────────────────────────────────
def embeddings_yukle(yol: str = EMBEDDINGS):
    d = np.load(yol, allow_pickle=True)
    return d["raw_embeddings"].astype(np.float32), d["edge_embeddings"].astype(np.float32), d["dosya_adlari"]


def cosine_topk(query: np.ndarray, bank: np.ndarray, k: int):
    """query: (512,), bank: (N, 512) — ikisi de L2 normalize. Top-K indeks ve skor döner."""
    # bank zaten normalize varsayılıyor; değilse normalize et
    norms = np.linalg.norm(bank, axis=1)
    if not np.allclose(norms.mean(), 1.0, atol=1e-2):
        bank = bank / np.clip(norms[:, None], 1e-8, None)
    sims = bank @ query
    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx_unsorted = np.argpartition(-sims, k)[:k]
        idx = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    return idx, sims[idx]


def rotasyon_birlestir(dosya_adlari: np.ndarray, idx: np.ndarray, sims: np.ndarray):
    """
    Top-K rotasyon-spesifik eşleşmeleri DWG-bazına indir.
    "1.png__0", "1.png__90" → "1.png" (en yüksek skor alınır).
    """
    en_iyi: dict[str, float] = {}
    for i, s in zip(idx, sims):
        ad = str(dosya_adlari[i])
        baz = ad.rsplit("__", 1)[0]
        if baz not in en_iyi or s > en_iyi[baz]:
            en_iyi[baz] = float(s)
    return sorted(en_iyi.items(), key=lambda kv: -kv[1])


def ara(foto_yolu: str, k: int = 10, ckpt: str = MODEL_BEST, embeddings_yolu: str = EMBEDDINGS,
        kanal: str = "edge"):
    """
    kanal: "raw" | "edge" | "ortalama" | "max"
      - raw: sadece raw_embeddings ile
      - edge: sadece edge_embeddings ile
      - ortalama: (raw_sim + edge_sim) / 2
      - max: max(raw_sim, edge_sim)
    """
    print(f"[yükleniyor] model: {ckpt}")
    model, preprocess, device = model_yukle(ckpt)

    print(f"[yükleniyor] embeddings: {embeddings_yolu}")
    raw_bank, edge_bank, adlar = embeddings_yukle(embeddings_yolu)
    print(f"  bank: {raw_bank.shape}, {edge_bank.shape}  |  {len(adlar)} vektör")

    print(f"[gömme] foto: {foto_yolu}")
    q_raw, q_edge = foto_iki_gomme(model, preprocess, foto_yolu, device)

    # Tüm bank'ı tara, ardından kanala göre birleştir
    raw_sims = raw_bank @ q_raw
    edge_sims = edge_bank @ q_edge

    if kanal == "raw":
        sims = raw_sims
    elif kanal == "edge":
        sims = edge_sims
    elif kanal == "ortalama":
        sims = (raw_sims + edge_sims) / 2
    elif kanal == "max":
        sims = np.maximum(raw_sims, edge_sims)
    else:
        raise ValueError(f"bilinmeyen kanal: {kanal}")

    # En iyi 4*k rotasyon-spesifik sonuç al, sonra DWG bazına indir, k tane döndür
    aday_k = min(len(sims), 4 * k * 4)
    idx_unsorted = np.argpartition(-sims, aday_k - 1)[:aday_k]
    idx = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    birlesik = rotasyon_birlestir(adlar, idx, sims[idx])[:k]
    return birlesik


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True, help="Fotoğraf dosya yolu")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ckpt", default=MODEL_BEST)
    ap.add_argument("--embeddings", default=EMBEDDINGS)
    ap.add_argument("--kanal", default="edge", choices=["raw", "edge", "ortalama", "max"])
    args = ap.parse_args()

    sonuc = ara(args.photo, k=args.k, ckpt=args.ckpt, embeddings_yolu=args.embeddings, kanal=args.kanal)
    print(f"\nTop-{args.k} (kanal={args.kanal}):")
    for sira, (ad, skor) in enumerate(sonuc, 1):
        print(f"  {sira:2d}. {skor:.4f}  {ad}")


if __name__ == "__main__":
    main()
