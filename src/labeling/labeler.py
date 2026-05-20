"""
Active-learning etiketleme UI — tek ekran tasarım.

Üst header:  ilerleme + DWG ara + ◀ Geri / 🔄 Yenile / ✅ Bitti / ▶ İleri
Sol:         foto thumbnail + meta
Sağ:         3×2 = 6 öneri grid (her birinin altında "Eşleşir" checkbox)
Arama dolu ise sağdaki grid arama sonuçlarıyla değişir.

Kullanım:
    streamlit run src/labeling/labeler.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    PHOTOS, CAD_PNG, EMBEDDINGS, MODEL_BEST,
    CLIP_BACKBONE, CLIP_PRETRAINED, LABELS_FILE, FOTO_META_FILE,
)
from utils import load_rgb, foto_edge, labels_yukle  # noqa: E402


GRUP_BOYUTU = 6
YENILEME_HAKKI = 4
THUMB_W = 200          # öneri thumbnail genişliği (px)
FOTO_W = 280           # ana foto genişliği (px)
ARAMA_MAX = 60         # arama modunda en fazla gösterilecek sonuç
ARAMA_SUTUN = 4        # arama grid sütun sayısı
ARAMA_THUMB_W = 150    # arama thumbnail biraz küçük


# ──────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────
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
    return edge, d["dosya_adlari"]


@st.cache_resource
def foto_meta_yukle():
    p = Path(FOTO_META_FILE)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def tum_fotolar_listesi():
    return sorted(p for p in PHOTOS.iterdir()
                  if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"))


@st.cache_resource
def tum_dwg_pngler():
    return sorted(p.name for p in CAD_PNG.glob("*.png"))


# ──────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def foto_gomme(model, preprocess, foto_yolu, device):
    img = foto_edge(load_rgb(str(foto_yolu)))
    x = preprocess(img).unsqueeze(0).to(device)
    v = model.encode_image(x)
    v = v / v.norm(dim=-1, keepdim=True)
    return v[0].cpu().numpy().astype(np.float32)


def oner_dwg_ler(query, edge_bank, adlar, toplam=GRUP_BOYUTU * YENILEME_HAKKI):
    sims = edge_bank @ query
    aday_k = min(len(sims), toplam * 8)
    idx_unsorted = np.argpartition(-sims, aday_k - 1)[:aday_k]
    idx = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    en_iyi = {}
    for i in idx:
        ad = str(adlar[i])
        baz, rot = ad.rsplit("__", 1) if "__" in ad else (ad, "0")
        if baz not in en_iyi or sims[i] > en_iyi[baz][0]:
            en_iyi[baz] = (float(sims[i]), rot)
    return sorted(en_iyi.items(), key=lambda kv: -kv[1][0])[:toplam]


# ──────────────────────────────────────────────────────────────
# labels.json yazımı — foto bazlı overwrite (utils.labels_kaydet merge ediyordu)
# ──────────────────────────────────────────────────────────────
def labels_overwrite_foto(d: dict, foto_ad: str, secimler: set, tamamlanan: list):
    d["tamamlanan"] = tamamlanan
    if secimler:
        d["eslesme"][foto_ad] = sorted(secimler)
    else:
        d["eslesme"].pop(foto_ad, None)
    if Path(LABELS_FILE).exists():
        shutil.copy2(LABELS_FILE, LABELS_FILE + ".backup")
    Path(LABELS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# Sayfa kurulumu + CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Kapı Etiketleyici", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container { padding-top: 0.7rem; padding-bottom: 0.5rem; max-width: 100%; }
header[data-testid="stHeader"] { display: none; }
[data-testid="stImage"] img { max-height: 240px; object-fit: contain; }
.stButton button { padding: 0.25rem 0.5rem; }
.stCheckbox { margin-bottom: 0; }
.element-container { margin-bottom: 0.3rem; }
[data-testid="stCaptionContainer"] { font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# State + kaynaklar
# ──────────────────────────────────────────────────────────────
model, preprocess, device = model_yukle()
edge_bank, adlar = embeddings_yukle()
meta = foto_meta_yukle()
tum_fotolar = tum_fotolar_listesi()
tum_dwgler = tum_dwg_pngler()
labels = labels_yukle()

if not tum_fotolar:
    st.error("Fotoğraf yok. `python -m src.data_prep.ingest_photos` çalıştırın.")
    st.stop()

ss = st.session_state
ss.setdefault("foto_idx", 0)
ss.setdefault("yenileme_no", 0)
ss.setdefault("secimler", set())
ss.setdefault("oneriler", None)
ss.setdefault("son_foto", None)
ss.setdefault("oto_arama", True)   # foto no'sundan otomatik DWG araması
ss.setdefault("arama_override", None)  # foto değişince otomatik dolduracak

ss.foto_idx = max(0, min(ss.foto_idx, len(tum_fotolar) - 1))
foto = tum_fotolar[ss.foto_idx]

# Foto değiştiğinde yeniden hesapla + mevcut seçimleri labels'tan yükle
if ss.son_foto != foto.name:
    ss.son_foto = foto.name
    ss.yenileme_no = 0
    ss.secimler = set(labels.get("eslesme", {}).get(foto.name, []))
    q = foto_gomme(model, preprocess, foto, device)
    ss.oneriler = oner_dwg_ler(q, edge_bank, adlar)
    # Otomatik arama: foto meta'sından taban no varsa o ile başla
    m = meta.get(foto.name, {})
    if ss.oto_arama and m.get("taban"):
        ss.arama_override = m["taban"]
    else:
        ss.arama_override = ""

tamamlanan = list(labels.get("tamamlanan", []))
bitti_isaret = foto.name in tamamlanan


# ──────────────────────────────────────────────────────────────
# Yardımcı — DWG kartı (checkbox + thumbnail + caption)
# ──────────────────────────────────────────────────────────────
def dwg_karti(slot, dwg_ad: str, etiket: str, anahtar: str, thumb_w: int = THUMB_W):
    with slot:
        p = CAD_PNG / dwg_ad
        if p.exists():
            st.image(str(p), width=thumb_w)
        else:
            st.markdown("⏳ _thumbnail_")
        st.caption(etiket)
        secili = st.checkbox("Eşleşir", key=anahtar, value=(dwg_ad in ss.secimler))
        if secili and dwg_ad not in ss.secimler:
            ss.secimler.add(dwg_ad)
            labels_overwrite_foto(labels, foto.name, ss.secimler, tamamlanan)
        elif not secili and dwg_ad in ss.secimler:
            ss.secimler.discard(dwg_ad)
            labels_overwrite_foto(labels, foto.name, ss.secimler, tamamlanan)


# ──────────────────────────────────────────────────────────────
# HEADER BAR
# ──────────────────────────────────────────────────────────────
hdr = st.columns([2, 1, 3, 1, 1, 1, 1])

with hdr[0]:
    durum = "✅ bitmiş" if bitti_isaret else "⏳ etiketsiz"
    st.markdown(
        f"**Kapı Etiketleyici**  ·  tamam: {len(tamamlanan)}  ·  "
        f"yenile {ss.yenileme_no + 1}/{YENILEME_HAKKI}  ·  {durum}"
    )

with hdr[1]:
    yeni_no = st.number_input(
        "foto no",
        min_value=1, max_value=len(tum_fotolar),
        value=ss.foto_idx + 1, step=1,
        label_visibility="collapsed",
        key="foto_no_input",
        help=f"İstediğin foto no'suna git (1-{len(tum_fotolar)})",
    )
    if yeni_no - 1 != ss.foto_idx:
        ss.foto_idx = int(yeni_no) - 1
        ss.son_foto = None
        st.rerun()

with hdr[2]:
    # Foto değiştiğinde arama kutusu otomatik dolar; kullanıcı boşaltabilir
    if ss.arama_override is not None:
        st.session_state["arama_kutu"] = ss.arama_override
        ss.arama_override = None
    arama_kol1, arama_kol2 = st.columns([4, 1])
    with arama_kol1:
        arama = st.text_input(
            "DWG ara", key="arama_kutu",
            label_visibility="collapsed",
            placeholder="DWG adı ara (örn. 100A, 21Afer)…",
        )
    with arama_kol2:
        ss.oto_arama = st.toggle("oto", value=ss.oto_arama, help="Yeni fotoğrafa geçince adın taban numarasıyla otomatik arama")

with hdr[3]:
    if st.button("◀ Geri", use_container_width=True, disabled=(ss.foto_idx == 0)):
        ss.foto_idx -= 1
        ss.son_foto = None
        st.rerun()

with hdr[4]:
    if st.button("🔄 Yenile", use_container_width=True,
                 disabled=(ss.yenileme_no >= YENILEME_HAKKI - 1)):
        ss.yenileme_no += 1
        st.rerun()

with hdr[5]:
    if st.button("✅ Bitti", type="primary", use_container_width=True):
        if foto.name not in tamamlanan:
            tamamlanan.append(foto.name)
        labels_overwrite_foto(labels, foto.name, ss.secimler, tamamlanan)
        ss.foto_idx = min(ss.foto_idx + 1, len(tum_fotolar) - 1)
        ss.son_foto = None
        st.rerun()

with hdr[6]:
    if st.button("İleri ▶", use_container_width=True,
                 disabled=(ss.foto_idx >= len(tum_fotolar) - 1)):
        ss.foto_idx += 1
        ss.son_foto = None
        st.rerun()

st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# ANA İÇERİK — sol foto, sağ grid
# ──────────────────────────────────────────────────────────────
sol, sag = st.columns([1, 3], gap="small")

with sol:
    st.image(str(foto), width=FOTO_W)
    m = meta.get(foto.name, {})
    iddia = []
    if m.get("taban"):   iddia.append(f"no={m['taban']}")
    if m.get("ad"):      iddia.append(f"ad={m['ad']}")
    if m.get("varyant"): iddia.append(f"var={m['varyant']}")
    if m.get("yumusak"): iddia.append("GİBİ")
    bilgi = f"`{foto.name}`"
    if iddia:
        bilgi += "  ·  " + " · ".join(iddia) + " ⚠"
    st.caption(bilgi)
    st.markdown(f"**Seçilen DWG:** `{len(ss.secimler)}`")
    if ss.secimler:
        st.caption("  ·  ".join(f"`{d}`" for d in sorted(ss.secimler)))

with sag:
    if arama.strip():
        # ── Arama sonuçları (öneri grid'inin yerine) ──
        q_str = arama.strip().lower()
        tum_eslesen = [d for d in tum_dwgler if q_str in d.lower()]
        eslesen = tum_eslesen[:ARAMA_MAX]
        if not eslesen:
            st.info(f"`{arama}` ile başlayan/içeren DWG yok.")
        else:
            ek = ""
            if len(tum_eslesen) > ARAMA_MAX:
                ek = f" (toplam {len(tum_eslesen)}, ilk {ARAMA_MAX} gösteriliyor — aramayı daralt)"
            st.caption(f"🔎 **Arama:** `{arama}` — {len(eslesen)} sonuç{ek}")
            satir_sayisi = (len(eslesen) + ARAMA_SUTUN - 1) // ARAMA_SUTUN
            for sira in range(satir_sayisi):
                sutunlar = st.columns(ARAMA_SUTUN)
                for i in range(ARAMA_SUTUN):
                    idx = sira * ARAMA_SUTUN + i
                    if idx >= len(eslesen):
                        continue
                    d = eslesen[idx]
                    dwg_karti(
                        sutunlar[i], d,
                        etiket=f"`{d}`",
                        anahtar=f"ara_{foto.name}_{d}",
                        thumb_w=ARAMA_THUMB_W,
                    )
    else:
        # ── Normal öneri grid ──
        baslangic = ss.yenileme_no * GRUP_BOYUTU
        grup = ss.oneriler[baslangic:baslangic + GRUP_BOYUTU]
        for sira in range(2):
            sutunlar = st.columns(3)
            for i in range(3):
                idx = sira * 3 + i
                if idx >= len(grup):
                    continue
                dwg_ad, (skor, rot) = grup[idx]
                dwg_karti(
                    sutunlar[i], dwg_ad,
                    etiket=f"`{dwg_ad}`  ·  skor={skor:.3f}  ·  rot={rot}",
                    anahtar=f"sec_{foto.name}_{dwg_ad}_{ss.yenileme_no}",
                )
