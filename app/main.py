"""
Müşteri uygulaması — fotoğraf yükle, en yakın AutoCAD çizimlerini gör.

Kullanım:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    CAD_PNG, EMBEDDINGS, MODEL_BEST,
    CLIP_BACKBONE, CLIP_PRETRAINED,
)
from utils import foto_edge  # noqa: E402


GOSTER_K = 12  # ilk sayfada gösterilen sonuç sayısı


@st.cache_resource
def model_yukle():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_PRETRAINED)
    ckpt = torch.load(MODEL_BEST, map_location="cpu", weights_only=False)
    state = {k.replace("clip_model.", "", 1) if k.startswith("clip_model.") else k: v
             for k, v in ckpt.items()}
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), preprocess, device


@st.cache_resource
def embeddings_yukle():
    d = np.load(EMBEDDINGS, allow_pickle=True)
    edge = d["edge_embeddings"].astype(np.float32)
    edge /= np.clip(np.linalg.norm(edge, axis=1, keepdims=True), 1e-8, None)
    adlar = d["dosya_adlari"]
    return edge, adlar


@torch.no_grad()
def gomme(model, preprocess, img: Image.Image, device: str):
    e = foto_edge(img.convert("RGB"))
    x = preprocess(e).unsqueeze(0).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy().astype(np.float32)


def ara(query: np.ndarray, edge_bank, adlar, k: int):
    sims = edge_bank @ query
    en_iyi = {}
    aday_idx = np.argpartition(-sims, min(k * 8, len(sims) - 1))[:k * 8]
    for i in aday_idx:
        ad = str(adlar[i])
        baz = ad.rsplit("__", 1)[0]
        if baz not in en_iyi or sims[i] > en_iyi[baz]:
            en_iyi[baz] = float(sims[i])
    return sorted(en_iyi.items(), key=lambda kv: -kv[1])[:k]


st.set_page_config(page_title="Kapı Bulucu", layout="wide")
st.title("Kapı Bulucu")
st.write("Bir kapı fotoğrafı yükleyin, en yakın AutoCAD çizimlerini görün.")

yuklenen = st.file_uploader("Fotoğraf", type=["jpg", "jpeg", "png", "webp", "bmp"])
if not yuklenen:
    st.info("Fotoğraf bekleniyor…")
    st.stop()

img = Image.open(yuklenen)
sol, sag = st.columns([1, 3])
with sol:
    st.image(img, caption="Yüklenen", use_container_width=True)

model, preprocess, device = model_yukle()
edge_bank, adlar = embeddings_yukle()
q = gomme(model, preprocess, img, device)
sonuclar = ara(q, edge_bank, adlar, GOSTER_K)

with sag:
    st.subheader(f"En yakın {len(sonuclar)} kapı")
    sutunlar = st.columns(4)
    for i, (ad, skor) in enumerate(sonuclar):
        with sutunlar[i % 4]:
            png_yolu = CAD_PNG / ad
            if png_yolu.exists():
                st.image(str(png_yolu), use_container_width=True)
            else:
                st.write("⏳ thumbnail hazır değil")
            # ".png" sıyır → muhtemel DWG adı
            dwg_adi = ad.replace(".png", ".dwg")
            st.caption(f"**{dwg_adi}**  ·  skor={skor:.3f}")
